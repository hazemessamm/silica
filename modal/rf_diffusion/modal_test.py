import modal

# Build the image as defined above
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04",
        add_python="3.9"
    )
    .copy_local_dir("./RFdiffusion/", "/app/RFdiffusion")
    .apt_install(
        "cuda-nvtx-11-6",
        "git",
        "libgomp1"
    )
    .run_commands(
        "python -m pip install -q -U --no-cache-dir pip"
    )
    .pip_install(
        "dgl==1.0.2+cu116",
        find_links="https://data.dgl.ai/wheels/cu116/repo.html"
    )
    .pip_install(
        "e3nn==0.3.3",
        "wandb==0.12.0",
        "pynvml==11.0.0",
        "git+https://github.com/NVIDIA/dllogger#egg=dllogger",
        "decorator==5.1.0",
        "hydra-core==1.3.2",
        "pyrsistent==0.19.3",
        "numpy==1.21.2",
    )
    .pip_install("/app/RFdiffusion/env/SE3Transformer")
    .pip_install("/app/RFdiffusion", extra_options="--no-deps")
    .pip_install(
        "torch==1.13.0+cu116",
        extra_index_url="https://download.pytorch.org/whl/cu116"
    )
    .env({"DGLBACKEND": "pytorch"})
    .workdir("/app/RFdiffusion")
)

app = modal.App("get_cdr_indices_test", image=image)

@app.function(image=image, concurrency_limit=5, gpu="A100")
def run_diffusion(pdb_path, round, contigs, hotspots, contig_name):
    import subprocess
    num_designs = 1
    save_path = "/app/RFdiffusion/outputs"

    command = [
        "python3.9", "scripts/run_inference.py",
        f"inference.input_pdb={pdb_path}",
        f"inference.output_prefix={save_path}",
        "inference.model_directory_path=/app/RFdiffusion/models",
        'denoiser.noise_scale_ca=0',  # leave alone unless you have a good reason
        'denoiser.noise_scale_frame=0',  # leave alone unless you have a good reason
        f'inference.num_designs={num_designs}',  # increase number of designs
    ]

    # if hotspots:
    #     command = command + [f'ppi.hotspot_res={hotspots}']
    if contigs:
        command = command + [f'contigmap.contigs={contigs}']
        print("Running command:", ' '.join(command))

    try:
        # Run the subprocess and capture output
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        # Print the error details
        print("An error occurred while running the command:")
        print("Return code:", e.returncode)
        print("Command:", ' '.join(e.cmd))
        print("Standard Output:\n", e.stdout)
        print("Standard Error:\n", e.stderr)
        # Optionally re-raise the exception if you want the function to fail
        # raise e
        return None

@app.local_entrypoint()
def main():
    pdb_path = "/app/RFdiffusion/inputs/4krl_chain_a.pdb"

    # Define your contigs and hotspots
    d12_diff_var = '[B307-511/0 A1-26/6-12/A36-98/13-17/A114-122]'
    d12_fro_diff_var = '[B307-511/0 A1-26/3-5/A30/3-6/A36-98/1-3/A100/12-15/A114-122]'
    d12_hs = '[B355,B353,B384,B357,B420]'

    # Generate the list of design parameters using a list comprehension
    design_params = [
        (pdb_path, 3, contigs, d12_hs if hotspots else None, contig_name)
        for contigs, contig_name in zip([d12_diff_var, d12_fro_diff_var], ["var", "fixed_var"])
        for hotspots in [True, False]
    ]

    # Use starmap to run run_diffusion over design_params
    results = list(run_diffusion.starmap(design_params[0:1]))
    print("Results:")
    for result in results:
        print(result)
    print(results)