import modal

app = modal.App("get_cdr_indices")

# Define the Dockerfile image
dockerfile_image = modal.Image.from_dockerfile("Dockerfile", 
            context_mount=modal.Mount.from_local_dir(
                local_path="./RFdiffusion/",
                remote_path="."
            ))

models_mount = modal.Mount.from_local_dir(local_path="RFdiffusion/models", remote_path="/models")
inputs_mount = modal.Mount.from_local_dir(local_path="RFdiffusion/inputs", remote_path="/inputs")
outputs_mount = modal.Mount.from_local_dir(local_path="RFdiffusion/outputs", remote_path="/outputs")

app = modal.App("get_cdr_indices", mounts=[models_mount, inputs_mount, outputs_mount])
vol = modal.Volume.from_name("diffusion-volume", create_if_missing=True)

@app.function(image=dockerfile_image, concurrency_limit=5, gpu="A100", volumes={"/diffusion_output": vol})
def run_diffusion(pdb_path, round_num, contigs, hotspots, contig_name, num_designs=2):
    import subprocess
    import random
    import os

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
    dir_path = os.path.join("/diffusion_output", f"{base_name}_test")

    # Ensure the directory exists
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
        # f'contigmap.contigs=[B307-511/0 A1-26/6-12/A36-98/13-17/A114-122]',
        # f'ppi.hotspot_res=[B355,B353,B384,B357,B420]',
    ]

    if hotspots:
        command = command + [f'ppi.hotspot_res={hotspots}']
    if contigs:
        command = command + [f'contigmap.contigs={contigs}']

    result = subprocess.run(command, check=True)
    print("Subprocess completed successfully.")
    volume.commit()

    return result


@app.local_entrypoint()
def main():
    pdb_path = "/inputs/4krl_chain_a.pdb"

    # Define your contigs and hotspots
    d12_diff_var = '[B307-511/0 A1-26/6-12/A36-98/13-17/A114-122]'
    d12_fro_diff_var = '[B307-511/0 A1-26/3-5/A30-30/3-6/A36-98/1-3/A100-100/12-15/A114-122]'
    d12_hs = '[B355,B353,B384,B357,B420]'

    # Generate the list of design parameters using a list comprehension
    design_params = [
        (pdb_path, 4, contigs, d12_hs if hotspots else None, contig_name)
        for contigs, contig_name in zip([d12_diff_var, d12_fro_diff_var], ["var", "fixed_var"])
        for hotspots in [True, False]
    ]

    # Use starmap to run run_diffusion over design_params
    results = list(run_diffusion.starmap(design_params))
    print("Results:")
    for result in results:
        print(result)
    print(results)
