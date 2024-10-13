import modal
import os
import tempfile
import shutil
import re



# Build image and dependencies. Will be cached after build
antifold_image = (
    modal.Image.micromamba(python_version="3.10")\
    .apt_install("libopenblas-dev")\
    .micromamba_install(spec_file='environment.yml')\
    .apt_install("git")\
    .copy_local_dir('.', 'AntiFoldLocal')\
    .run_commands("cd AntiFoldLocal && python -m pip install .")\
    .run_commands("python -c \"import antifold.main; antifold.main.load_model()\"")\
)
    
app = modal.App(name="antifold", image=antifold_image)


@app.cls(cpu=1, image=antifold_image)
class AntiFold:
    @modal.enter()
    def run_this_on_container_startup(self):
        import antifold.main
        self.model = antifold.main.load_model()

    @modal.method()
    def run_antifold(self, pdb_contents, hchain, lchain, regions_to_mut,
                     num_seqs, t, nanobody=False,
                     batch_size=1, logits=False, logprobs=False):
        """
        Returns
        -------
        {'sequences': OrderedDict([('temp_pdb_H', SeqRecord(seq=Seq('EVQLVESGGGLLSCAASGRTFSSYAMGWFRQAPGKEREFVVAINWSSGSTYYAD...QVT'),
        id='temp_pdb_H', name='', description=", score=1.1935, global_score=1.1935,
        regions=['CDR3'], model_name=AntiFold, seed=42", dbxrefs=[])),
        ('temp_pdb_H__1', SeqRecord(seq=Seq('EVQLVESGGGLLSCAASGRTFSSYAMGWFRQAPGKEREFVVAINWSSGSTYYAD...QVT'), id='', name='',
        description='T=2.50, sample=1, score=2.9150, global_score=1.3810, seq_recovery=0.8621, mutations=16', dbxrefs=[]))])}
        """

        import antifold.main
        import pandas as pd

        # Create a temporary directory to store the PDB file and results
        tmpdirname, tmp_pdb_path, tmp_csv_path = None, None, None
        tmpdirname = tempfile.mkdtemp()

        try:
            # Write PDB contents to a temporary file
            tmp_pdb_path = os.path.join(tmpdirname, "temp_pdb.pdb")
            with open(tmp_pdb_path, "w") as tmp_pdb_file:
                tmp_pdb_file.write(pdb_contents)

            # Write CSV content directly to a temporary file
            tmp_csv_path = os.path.join(tmpdirname, "temp_pdb.csv")
            with open(tmp_csv_path, 'w') as f:
                if nanobody is True:
                    f.write(f"pdb,Hchain\n{os.path.basename(tmp_pdb_path).split('.')[0]},{hchain}")
                else:
                    f.write(f"pdb,Hchain,Lchain\n{os.path.basename(tmp_pdb_path).split('.')[0]},{hchain},{lchain}")

            # Load DataFrame from the temporary CSV
            df_pdbs = pd.read_csv(tmp_csv_path)

            # Running AntiFold on the temporary PDB and CSV files
            output = antifold.main.sample_pdbs(
                model=self.model,
                pdbs_csv_or_dataframe=df_pdbs,
                regions_to_mutate=regions_to_mut.split(" "),
                pdb_dir=tmpdirname,  # Directory where the temporary PDB file is located
                out_dir=tmpdirname,  # Directory to store output files
                sample_n=int(num_seqs),
                sampling_temp=[float(t)],
                save_flag=True,
                batch_size=int(batch_size),
                seed=None,
                custom_chain_mode=True if nanobody is True else False,
            )
            if not logits:
                output['temp_pdb_H'].pop('logits')
            if not logprobs:
                output['temp_pdb_H'].pop('logprobs')
            return output['temp_pdb_H']
        except Exception as e:
            raise
        finally:
            # Cleanup temporary files and directory
            if tmpdirname and os.path.exists(tmpdirname):
                shutil.rmtree(tmpdirname)

    @modal.method()
    def score_single_seq(self, sequence):
        f = modal.Function.from_name('nanobodybuilder2', 'predict_structure')
        pdb_str = f.remote(sequence)
        results = AntiFold.run_antifold.remote(
            pdb_contents=pdb_str,
            hchain='H',
            lchain=None,
            regions_to_mut='CDR1',
            num_seqs=1,
            t=0.2,
            nanobody=True,
            batch_size=1,
            logits=False,
            logprobs=False
        )
        parent = results['sequences']['temp_pdb_H']
        score, global_score = self.extract_scores(parent.description)
        return score, global_score

    @modal.method()
    def score_single_pdb_str(self, pdb_contents):
        results = AntiFold.run_antifold.remote(
            pdb_contents=pdb_contents,
            hchain='H',
            lchain=None,
            regions_to_mut='CDR1',
            num_seqs=1,
            t=0.2,
            nanobody=True,
            batch_size=1,
            logits=False,
            logprobs=False
        )
        parent = results['sequences']['temp_pdb_H']
        score, global_score = self.extract_scores(parent.description)
        return score, global_score

    def extract_scores(self, description):
        score_match = re.search(r'score=([\d.]+)', description)
        global_score_match = re.search(r'global_score=([\d.]+)', description)

        score = float(score_match.group(1)) if score_match else None
        global_score = float(global_score_match.group(1)) if global_score_match else None

        return score, global_score




@app.local_entrypoint()
def nanobody_entry(pdb_path, hchain, regions, num_seqs, t, logits=False, logprobs=False):
    # Check if PDB exists
    if not os.path.exists(pdb_path):
        raise AssertionError(f'PDB {pdb_path} not found!')
    # Get PDB as string
    with open(pdb_path) as f:
        pdb_contents = f.read()
    # Run AntiFold
    results = AntiFold.run_antifold.remote(
        pdb_contents=pdb_contents,
        hchain=hchain,
        lchain=None,
        regions_to_mut=regions,
        num_seqs=num_seqs,
        t=t,
        nanobody=True,
        batch_size=1,
        logits=logits,
        logprobs=logprobs
    )
    print(results)
