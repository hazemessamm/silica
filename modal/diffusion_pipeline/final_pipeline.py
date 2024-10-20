import modal 
import os
import subprocess
from pathlib import Path
import shutil

GPU_CONFIG = modal.gpu.H100()
modal_app = modal.App()
def build_mpnn_image():
    return (
        modal.Image.debian_slim(python_version="3.11.5")
        .apt_install("git")
        .pip_install(
            "torch==2.0.0",
            "biopython",
            "ml-collections",
            "pyrosetta-installer",
            "numpy<2.0.0"
        )
        .run_commands(
            "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'",
            "git clone https://github.com/nrbennet/dl_binder_design.git /app/dl_binder_design",
            "git clone https://github.com/dauparas/ProteinMPNN.git /app/dl_binder_design/mpnn_fr/ProteinMPNN",
            "export PATH=$PATH:/app/dl_binder_design/include/silent_tools"
        )#.workdir("/app/dl_binder_design")
    )
def build_af2_image():
    return (
        # modal.Image.debian_slim()
        modal.Image.from_registry(
            # "tensorflow/tensorflow:2.13.0-gpu", add_python="3.11"
            "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04", add_python="3.11"
        ).apt_install("git", "wget", gpu=GPU_CONFIG)  # Added wget for fetching AlphaFold weights
        .pip_install(
            "--find-links=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html", 
            "jax[cuda12_pip]==0.4.20", 
            "jaxlib==0.4.20+cuda12.cudnn89",
            "dm-haiku",
            "dm-tree",
            "biopython==1.81",
            "ml-collections",
            "pyrosetta-installer",
            "ml_dtypes",
            "tensorflow==2.13.0",
            "mock", 
            "pandas",
            "flax",
            gpu=GPU_CONFIG,
        )
        .run_commands(
            "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'",  # Keep the PyRosetta installation process the same
            "git clone https://github.com/nrbennet/dl_binder_design.git",  # Clone dl_binder_design repo
            # Download and extract AlphaFold model weights
            "mkdir -p dl_binder_design/af2_initial_guess/model_weights/params",  # Create directory for AlphaFold weights
            "wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -O dl_binder_design/af2_initial_guess/model_weights/params/alphafold_params_2022-12-06.tar",  # Download weights
            "tar -xvf dl_binder_design/af2_initial_guess/model_weights/params/alphafold_params_2022-12-06.tar -C dl_binder_design/af2_initial_guess/model_weights/params"  # Extract weights
        )
    )
mpnn_image = build_mpnn_image()
af2_image = build_af2_image()
vol = modal.Volume.from_name("diffusion-volume-3", create_if_missing=True)

# Define the Dockerfile image
rf_diffusion_image = modal.Image.from_dockerfile("Dockerfile", 
            context_mount=modal.Mount.from_local_dir(
                local_path="./RFdiffusion/",
                remote_path="."
            ))

models_mount = modal.Mount.from_local_dir(local_path="RFdiffusion/models", remote_path="/models")
inputs_mount = modal.Mount.from_local_dir(local_path="RFdiffusion/inputs", remote_path="/inputs")
outputs_mount = modal.Mount.from_local_dir(local_path="RFdiffusion/outputs", remote_path="/outputs")
print('IMAGES BUILT')

modal_app = modal.App()

# #RF diffusion class
@modal_app.cls(
    image=af2_image,
    gpu="A100",
    timeout=24*60*60,
    volumes={"/diffusion_output": vol},  # Mount the volume at "/vol"
)
class Af2Runner:

    def read_sc_file(self, sc_file_path):
        """
        Reads the .sc file and returns it as a pandas DataFrame.
        The first line is assumed to be the header.
        """
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1

        #check if the file exists; if it doesn't just print missing: file
        # if not os.path.exists(sc_file_path):
        #     print(f"Missing: {sc_file_path}")
        #     #save name to a missing files tracker file. Append and create the missing files tracker file if it doesn't exist
        #     with open("missing_files_tracker.txt", "a") as f:
        #         f.write(f"{sc_file_path}\n")
        #     return None
        with open(sc_file_path, 'r') as file:
            # Extract columns from the first line starting after 'SCORE:'
            header_line = file.readline().strip().split()[1:]  # Skip 'SCORE:' label
            data = []
        
            # Read the remaining lines and split them into columns
            for line in file:
                if line.startswith("SCORE:"):
                    data_line = line.strip().split()[1:]  # Skip 'SCORE:' label
                    data.append(data_line)
    
        # Create DataFrame
        df = pd.DataFrame(data, columns=header_line)
    
        return df

    def copy_file(self,source_file, destination_file):
        try:
            # Copy the source file to the destination
            shutil.copy(source_file, destination_file)
            print(f"File copied successfully from {source_file} to {destination_file}")
        except FileNotFoundError:
            print(f"Source file not found: {source_file}")
        except PermissionError:
            print(f"Permission denied when copying to: {destination_file}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def extract_chain_a_sequence(self, pdb_file):
        """
        Extract the amino acid sequence of chain A from a PDB file.
        """
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)

        chain_a_sequence = ""
        for model in structure:
            for chain in model:
                if chain.id == 'A':  # Select only chain A
                    for residue in chain:
                        # Filter standard amino acids and convert to one-letter code
                        if PDB.is_aa(residue, standard=True):
                            chain_a_sequence += seq1(residue.resname)
                    break  # Exit after first model's chain A

        return chain_a_sequence

    def add_chain_a_sequences_to_df(self, df, pdb_dir):
        """
        Adds a new column 'Chain_A_Sequence' to the DataFrame with the sequence of Chain A
        for each corresponding PDB file.
        """
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1

        sequences = []
    
        for description in df['description']:
            pdb_file_path = os.path.join(pdb_dir, description + ".pdb")
            if os.path.exists(pdb_file_path):
                sequence = self.extract_chain_a_sequence(pdb_file_path)
            else:
                sequence = None  # If the PDB file does not exist
            sequences.append(sequence)
    
        df['sequence'] = sequences
        return df

    def extract_silent_files(self, silent_file):
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1
        import subprocess

        os.environ['PATH'] += ':/dl_binder_design/include/silent_tools'
        cmd = [ 'cd','/diffusion_output/af2','&&','/dl_binder_design/include/silent_tools/silentextract', f'{silent_file}']
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(" ".join(cmd), shell=True, check=True)
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred:")
            print("Return code:", e.returncode)
            print("Output:", e.output)
            print("Error:", e.stderr)
            raise

    def find_all_silent_files(self, directory):
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1

        matches_silent = []
        matches_sc = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".silent"):
                    matches_silent.append(os.path.join(root, file))
                    matches_sc.append(os.path.join(root, file.replace('.silent', '.sc')))
                    print(os.path.join(root, file))
        return matches_silent, matches_sc

    @modal.method()
    def run_extract(self, silent_file, score_file):
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1

        # a, b = self.find_all_silent_files('/diffusion_output/af2')
        # print(f'a: {a}', f'b: {b}')
        # df_list = []   
        # for silent_file, score_file in vals:  
        #check that score and silent files exist
        if os.path.exists(silent_file) and os.path.exists(score_file): 
            print(f"Silent file found at: {silent_file}")
            print(f"Score file found at: {score_file}")
        else:
            print(f"Silent file '{silent_file}' or score file '{score_file}' not found.")
            #save name to a missing files tracker file. Append and create the missing files tracker file if it doesn't exist
            with open("/diffusion_output/missing_files_tracker.txt", "a") as f:
                f.write(f"{silent_file}\n")
                f.write(f"{score_file}\n")  
            return None     
        self.extract_silent_files(silent_file)        
        df = self.read_sc_file(score_file)
        if df is not None:
            df = self.add_chain_a_sequences_to_df(df, '/diffusion_output/af2')
            # df_list.append(df)
        #load progress tracker file and add the silent file to it
        with open("/diffusion_output/progress_tracker.txt", "a") as f:
            f.write(f"{silent_file}\n")
        return df
        # return pd.concat(df_list)

    def find_all_silent_files_2(self, starting_directory: str, target_filename: str):
        """
        Recursively searches for all files matching the target filename starting from the specified directory.

        :param starting_directory: The directory to start searching from.
        :param target_filename: The name of the file to find.
        :return: List of full paths to the target file if found, otherwise an empty list.
        """
        matching_files = []

        for root, dirs, files in os.walk(starting_directory):
            if target_filename in files:
                # Append the full path to the matching file
                matching_files.append(os.path.join(root, target_filename))

        return matching_files

    @modal.method()
    def run_af2(self, af2_input):
        import os
        import pandas as pd
        from Bio import PDB
        from Bio.SeqUtils import seq1

        #print the input file directory structure
        print(f'af2_input: {af2_input}')

        #get the input file directory structure
        vol.reload()
        #get the base name of the pdb file
        file_path_parts = af2_input.replace('mpnn', 'af2').split("/")[:-1]  # Extract the directory path as a list
        file_name = af2_input.split("/")[-1] # Extract the file name

        # Join the parts back to form the full directory path
        af2_output_path = "/".join(file_path_parts) + "/" + file_name
        sc_output_path = af2_output_path.replace('.silent', '.sc')

        # Confirm that directory exists, if not, create it
        if not os.path.exists("/".join(file_path_parts)):
            os.makedirs("/".join(file_path_parts))

        print("Environment variables:", os.environ)
        try:
            gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            print("GPU Info:", gpu_info)
        except:
            print("No GPU found or nvidia-smi not available")

        print(os.environ.get('LD_LIBRARY_PATH'))
        print(os.environ.get('CUDA_HOME'))
        print(os.environ)

        cmd = [
                "/dl_binder_design/af2_initial_guess/predict.py",
                "-silent",
                af2_input,
                "-outsilent",
                af2_output_path,
                "-scorefilename",
                sc_output_path
            ]
        
        print(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(" ".join(cmd), shell=True, check=True)
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred:")
            print("Return code:", e.returncode)
            print("Output:", e.output)
            print("Error:", e.stderr)
            raise
        
        # silent_file_out = self.find_all_silent_files_2('/','out.silent')[0]
        # silent_file_score = self.find_all_silent_files_2('/','out.sc')[0]
        # self.copy_file(silent_file_out, af2_output_path)
        # sc_output_path = af2_output_path.replace('.silent', '.sc')
        # self.copy_file(silent_file_score, sc_output_path)

        return af2_output_path, sc_output_path

    # Get the path to the silent file

#MPNN class
@modal_app.cls(
    image=mpnn_image,
    timeout=24*60*60,
    volumes={"/diffusion_output": vol},  # Mount the volume at "/vol",
)
class MPNN:
    """
    MPNN class that handles ...
    """
    import shutil

    def find_all_silent_files_2(self, starting_directory: str, target_filename: str):
        """
        Recursively searches for all files matching the target filename starting from the specified directory.

        :param starting_directory: The directory to start searching from.
        :param target_filename: The name of the file to find.
        :return: List of full paths to the target file if found, otherwise an empty list.
        """
        matching_files = []

        for root, dirs, files in os.walk(starting_directory):
            if target_filename in files:
                # Append the full path to the matching file
                matching_files.append(os.path.join(root, target_filename))

        return matching_files

    def find_all_silent_files(self, starting_directory: str, target_filename: str):
        """
        Recursively searches for all files matching the target filename starting from the specified directory.
    
        :param starting_directory: The directory to start searching from.
        :param target_filename: The name of the file to find.
        :return: List of full paths to the target file if found, otherwise an empty list.
        """
        matching_files = []
    
        # Walk the directory tree
        for root, dirs, files in os.walk(starting_directory):
            if target_filename in files:
                # Append the full path to the matching file
                matching_files.append(os.path.join(root, target_filename))
    
        return matching_files
    
    def copy_file(self, source_file, destination_file):
        try:
            # Copy the source file to the destination
            shutil.copy(source_file, destination_file, )
            print(f"File copied successfully from {source_file} to {destination_file}")
        except FileNotFoundError:
            print(f"Source file not found: {source_file}")
        except PermissionError:
            print(f"Permission denied when copying to: {destination_file}")
        except Exception as e:
            print(f"An error occurred: {e}")

    @modal.method()
    def add_fixed_labels(self, pdb_dir: str) -> str:
        """
        Add FIXED labels to PDB files based on RFdiffusion trajectories.

        Args:
            pdb_dir (str): Path to the directory containing PDB files.
            trb_dir (str): Path to the directory containing trajectory files.

        Returns:
            str: Path to the directory containing PDB files with FIXED labels.
        """
        vol.reload()
        cmd = [
            "python", "/app/dl_binder_design/helper_scripts/addFIXEDlabels.py",
            "--pdbdir", pdb_dir,
            "--trbdir", pdb_dir,
            "--verbose"
        ]
        subprocess.run(cmd)

        return pdb_dir

    @modal.method()
    def create_silent_file(self, pdb_file: str) -> str:
        """
        Create a silent file from PDB files.

        Args:
            pdb_dir (str): Path to the directory containing the fixed PDB files.
            silent_file (str): Path to the output silent file.

        Returns:
            str: Path to the created silent file.
        """

        vol.reload()
        #get the base name of the pdb file
        file_path_parts = pdb_file.replace('rf_diffusion', 'mpnn').split("/")[:-1]  # Extract the directory path as a list
        silent_file_name = pdb_file.split("/")[-1].replace('.pdb','.silent')  # Extract the file name

        # Join the parts back to form the full directory path
        mpnn_input_path = "/".join(file_path_parts) + "/" + silent_file_name

        # Confirm that directory exists, if not, create it
        if not os.path.exists("/".join(file_path_parts)):
            os.makedirs("/".join(file_path_parts))

        os.environ['PATH'] += ':/app/dl_binder_design/include/silent_tools'
        cmd = [ '/app/dl_binder_design/include/silent_tools/silentfrompdbs', f'{pdb_file}']
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)#, shell=True)

        with open(silent_file_name, 'wb') as outfile: # Use 'w' to overwrite the file 
            subprocess.run(cmd, stdout=outfile)

        vol.commit()
        vol.reload()
        
        # Start the search from the root directory
        silent_file_path = self.find_all_silent_files("/", silent_file_name)
        
        if silent_file_path:
            print(f"Silent file (compressed from pdbs) found at: {silent_file_path} Expected /app/my_designs.silent")
        else:
            print(f"Silent file '{silent_file_name}' not found.")

        # Example usage
        source = silent_file_path[0]  # Replace with your actual source file path
        destination = mpnn_input_path # Replace with your actual destination path
        
        self.copy_file(source, destination)

        print(f"Creating silent file at: {destination}")

        if os.path.getsize(destination) == 0:
                raise ValueError(f"But silent file is empty: {destination}")

        vol.commit()
        return destination

    @modal.method()
    def run_mpnn_fastrelax(self, mpnn_input: str) -> str:
        """
        Run MPNN-FastRelax on the silent file. 

        This will use the default of 1 FastRelax cycle of 1 ProteinMPNN sequence per round
        (NOTE: running with -relax_cycles > 0 and -seqs_per_struct > 1 is disabled as it leads 
        to an explosion in the amount of FastRelax trajectories being run and is probably bad idea)

        It is not recommended to run ProteinMPNN with a GPU as it is only marginally faster 
        than running on CPU and FastRelax cannot take advantage of the GPU anyway.

        Args:
            silent_file (str): Path to the input silent file.
            output_file (str): Path to the output silent file.

        Returns:
            str: Path to the output silent file containing designs.
        """
        import shutil
        vol.reload()

        os.environ['PATH'] += ':/app/dl_binder_design/include/silent_tools'

        print(f"Passing silent file to final function from: {mpnn_input}")

        cmd = [
            "/app/dl_binder_design/mpnn_fr/dl_interface_design.py",
            "-silent", mpnn_input
        ]


        print(f"Running command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check = True, text = True)

        #print error if the silent file is not found
        vol.commit()
        vol.reload()

        # print("Standard Output:\n", result.stdout)
        silent_file_path = self.find_all_silent_files_2("/", "out.silent")

        vol.commit()
        vol.reload()
        if silent_file_path:
            print('############################################################################################################')
            print('############################################################################################################')
            print(f"Silent file found at: {silent_file_path}")
            print('############################################################################################################')
            print('############################################################################################################')
        else:
            print('############################################################################################################')
            print('############################################################################################################')
            print(f"Silent file  not found.")
            print('############################################################################################################')
            print('############################################################################################################')

        if isinstance(silent_file_path, list) and len(silent_file_path) > 0:
            source = silent_file_path[0]
        else:
            source = silent_file_path
        destination = mpnn_input.replace(".silent", "_mpnn.silent")
        self.copy_file(source, destination)
    
        return destination  

# #RF diffusion class
@modal_app.cls(
    image=rf_diffusion_image,
    gpu="A100",  # Use L4 GPU
    timeout=24*60*60,
    volumes={"/diffusion_output": vol},  # Mount the volume at "/vol"
    mounts=[models_mount, inputs_mount, outputs_mount]
)
class RF_Diffusion:
    """
    RF_Diffusion class that handles ...
    """
    def generate_path(self, pdb_path, round_num, contigs, hotspots, contig_name):
        import os
        import random
        import subprocess

        vol.reload()
        # Extract the base name from the pdb_path
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]

        # Build the output parts
        output_parts = []

        if contigs:
            output_parts.append(contig_name)
        if hotspots:
            output_parts.append("hs")

        # Add the round number
        output_parts.append(f"round{round_num}")

        # Join the parts with underscores
        output = "_".join(output_parts)

        # Create the directory path
        # dir_path = os.path.join("/diffusion_output", "rf_diffusion", output, f"{output}_test")
        dir_path = os.path.join("/diffusion_output", "rf_diffusion", output)

        return dir_path, output

    @modal.method()
    def run_diffusion(self, pdb_path, round_num, contigs, hotspots, contig_name, num_designs=2):
        
        vol.reload()
        # Generate the directory path and output
        dir_path, output = self.generate_path(pdb_path, round_num, contigs, hotspots, contig_name)

        #check if the directory exists
        if os.path.exists(dir_path):
            print(f"Directory {dir_path} already exists.")
            return dir_path
        else:
            # Create the directory
            os.makedirs(dir_path, exist_ok=True)

        # Construct the save_path using the directory path and output
        save_path = os.path.join(dir_path, output)
        print(f"Save path: {save_path}")

        # Run the inference Python script with the same arguments
        command = [
            "python3.9", "scripts/run_inference.py",
            f"inference.input_pdb={pdb_path}",
            f"inference.output_prefix={save_path}",
            "inference.model_directory_path=/models",
            'denoiser.noise_scale_ca=0',  # leave alone unless you have a good reason
            'denoiser.noise_scale_frame=0',  # leave alone unless you have a good reason
            f'inference.num_designs={num_designs}',  # increase number of designs
        ]

        if hotspots:
            command = command + [f'ppi.hotspot_res={hotspots}']
        if contigs:
            command = command + [f'contigmap.contigs={contigs}']

        result = subprocess.run(command, check=True)
        print("Subprocess completed successfully.")
        vol.commit()

        print(f"Save path: {dir_path}")
        return dir_path
    
    def generate_design_params(self, pdb_path, num_designs = 2, run_diffusion = True):

        # Define your contigs and hotspots
        d12_diff_var = '[B307-511/0 A1-26/6-12/A36-98/13-17/A114-122]'
        d12_fro_diff_var = '[B307-511/0 A1-26/3-5/A30-30/3-6/A36-98/1-3/A100-100/12-18/A114-122]'
        d12_hs = '[B355,B353,B384,B357,B420]'

        d12_fro_diff = '[B307-511/0 A1-26/3-3/A30-30/5-5/A36-98/1-1/A100-100/13-13/A114-122]'
        d12_diff = '[B307-511/0 A1-26/9-9/A36-98/15-15/A114-122]'

        d12_fro_diff_cdr2_var = '[B307-511/0 A1-26/3-5/A30-30/3-8/A36-49/8-12/A58-98/1-3/A100-100/13-17/A114-122]'
        d12_diff_cdr2_var = '[B307-511/0 A1-26/8-12/A36-49/8-12/A58-98/12-18/A114-122]'

        d12_fro_diff_cdr2 = '[B307-511/0 A1-26/3-3/A30-30/5-5/A36-49/8-8/A58-98/1-1/A100-100/13-13/A114-122]'
        d12_diff_cdr2 = '[B307-511/0 A1-26/9-9/A36-49/8-8/A58-98/15-15/A114-122]'

        # List of all contigs and corresponding names
        contigs_list = [
            (d12_diff_var, "diff_var"),
            (d12_fro_diff_var, "fro_diff_var"),
            (d12_diff, "diff"),
            (d12_fro_diff, "fro_diff"),
            (d12_diff_cdr2_var, "diff_cdr2_var"),
            (d12_fro_diff_cdr2_var, "fro_diff_cdr2_var"),
            (d12_diff_cdr2, "diff_cdr2"),
            (d12_fro_diff_cdr2, "fro_diff_cdr2")
        ]

        # Generate the list of design parameters using a list comprehension
        design_params = [
            ((pdb_path, 4, contigs, d12_hs if hotspots else None, contig_name, num_designs),run_diffusion)
            for contigs, contig_name in contigs_list
            for hotspots in [True, False]
        ]

        #function to read results.py and find the design parameters that have already been run
        def read_results():
            results = []
            with open("results.csv", "r") as f:
                for line in f:
                    results.append(line.strip())
            return results

        return design_params

@modal_app.function(volumes={"/diffusion_output": vol})
def find_mpnn_silent_files(directory):
    import os

    matches_silent = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_mpnn.silent"):
                matches_silent.append(os.path.join(root, file))
                print(os.path.join(root, file))
    return matches_silent

@modal_app.function(volumes={"/diffusion_output": vol}, image=af2_image, timeout=24*60*60, gpu=None)
def af2_pipeline(matches_silent):
    import pandas as pd
    import pickle
    vol.commit()
    vol.reload()
    # matches_silent = find_mpnn_silent_files.remote(directory)
    # matches_silent = matches_silent.result()
    print('############################################################################################################')
    print('############################################################################################################')
    print(f'Matches_silent: {matches_silent}')
    print(f'Len of matches_silent: {len(matches_silent)}')
    print(f'First element of matches_silent: {matches_silent[0]}')
    print('############################################################################################################')
    print('############################################################################################################')
    vol.commit()
    vol.reload()
    # # Work horse
    vals = list(Af2Runner().run_af2.map(matches_silent))
    # #save to csv
    with open("/diffusion_output/vals.pkl", "wb") as f:
        pickle.dump(vals, f)

    return None
    # with open("/diffusion_output/vals.pkl", "rb") as f:
    #     vals = pickle.load(f)
    # vals = [('/diffusion_output/af2/diff_var_hs_round4/diff_var_hs_round4_0_mpnn.silent', '/diffusion_output/af2/diff_var_hs_round4/diff_var_hs_round4_0_mpnn.sc'), ('/diffusion_output/af2/diff_var_hs_round4/diff_var_hs_round4_1_mpnn.silent', '/diffusion_output/af2/diff_var_hs_round4/diff_var_hs_round4_1_mpnn.sc'), ('/diffusion_output/af2/diff_var_round4/diff_var_round4_0_mpnn.silent', '/diffusion_output/af2/diff_var_round4/diff_var_round4_0_mpnn.sc'), ('/diffusion_output/af2/diff_var_round4/diff_var_round4_1_mpnn.silent', '/diffusion_output/af2/diff_var_round4/diff_var_round4_1_mpnn.sc')]
    print('############################################################################################################')
    print('############################################################################################################')
    print(f'Vals: {vals}')
    print('############################################################################################################')
    print('############################################################################################################')
    vol.commit()
    vol.reload()
    # result = Af2Runner().run_af2.remote(af2_input='/diffusion_output/mpnn/fro_diff_cdr2_round4/fro_diff_cdr2_round4_9_mpnn.silent')
    
    #get first twp tupes in vals. Corrupt them slightly by changing the silent file path

    df = list(Af2Runner().run_extract.starmap(vals))
    #remove None values
    df = [x for x in df if x is not None]
    df = pd.concat(df)
    vol.commit()
    vol.reload()
    return df

@modal_app.function(volumes={"/diffusion_output": vol}, timeout=24*60*60)
def run_pipeline(design_param, only_rf_diffusion=False):
    """
    Run the MPNN pipeline.
    """
    ############################################################
    # Input to RFdiffusion
    ############################################################
    if only_rf_diffusion:
        return RF_Diffusion().run_diffusion.remote(*design_param)
    pdb_dir = RF_Diffusion().run_diffusion.remote(*design_param)
    pdb_dir = MPNN().add_fixed_labels.remote(pdb_dir)
    vol.commit()
    print(f"fixed labels pdb_dir: {pdb_dir}")

    alphafold_input_ls = []
    vol.reload()
    #iterate over the files in the result directory
    pdb_paths = [os.path.join(pdb_dir, file) for file in os.listdir(pdb_dir) if file.endswith(".pdb")]
    print("############################################################################################################")
    print(f"pdb_paths: {pdb_paths}")
    print("############################################################################################################")  
    mpnn_input = list(MPNN().create_silent_file.map(pdb_paths))
    vol.commit()
    vol.reload()
    print("############################################################################################################")
    print(f"mpnn_input: {mpnn_input }")
    print("############################################################################################################") 
    alphafold_input = list(MPNN().run_mpnn_fastrelax.map(mpnn_input))
    vol.commit()
    vol.reload()
    print("############################################################################################################")
    print(f"alphafold_input: {alphafold_input }")
    print("############################################################################################################")
    return alphafold_input

def main():
    with modal_app.run():
        pdb_path = "/inputs/4krl_chain_a.pdb"
        num_designs = 30

        #run RFDiffusion and MPNN pipeline
        design_params = RF_Diffusion().generate_design_params(pdb_path, num_designs=num_designs, run_diffusion = True)
        results = list(run_pipeline.starmap(design_params))
        print(f"Design parameters: {design_params}")
        design_params = RF_Diffusion().generate_design_params(pdb_path, num_designs=num_designs, run_diffusion = False)
        results = list(run_pipeline.starmap(design_params))
        print(f"Design parameters: {design_params}")

        #print results
        print(f"MPNN results: {results}")

        #save results to csv
        with open("MPNN_results.txt", "w") as f:
            for result in results:
                f.write(f"{result}\n")

        # af2_df = af2_pipeline.remote('/diffusion_output/mpnn')
#         results = [['/diffusion_output/mpnn/diff_var_hs_round4/diff_var_hs_round4_0_mpnn.silent', 
# '/diffusion_output/mpnn/diff_var_hs_round4/diff_var_hs_round4_1_mpnn.silent'], ['/diffusion_output/mpnn/diff_var_round4/diff_var_round4_0_mpnn.silent', 
# '/diffusion_output/mpnn/diff_var_round4/diff_var_round4_1_mpnn.silent']]
        flat_results = [item for sublist in results for item in sublist]
        # flat_results = None
        af2_df = af2_pipeline.remote(flat_results)
        #save the dataframe to a csv file
        # af2_df.to_csv('af2_results.csv', index=False)
        # print(af2_df)

if __name__ == "__main__":
    main()





