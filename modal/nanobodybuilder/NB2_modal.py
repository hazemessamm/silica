import modal
from pathlib import Path


# Build image and dependencies. Will be cached after build
immune_builder_image = (
    modal.Image.micromamba(python_version="3.12").micromamba_install("openmm", "pdbfixer", channels=["conda-forge"]).apt_install("git", "wget").run_commands("git clone https://github.com/oxpig/ANARCI.git").micromamba_install("biopython", channels=["conda-forge"]).micromamba_install("hmmer=3.3.2", channels=["bioconda"]).run_commands("cd ANARCI && python setup.py install")
    .pip_install("ImmuneBuilder").apt_install("libopenblas-dev").run_commands("mkdir -p /app/NB2_weights").workdir("/app").run_commands("wget -O NB2_weights/nanobody_model_1 https://zenodo.org/record/7258553/files/nanobody_model_1?download=1", "wget -O NB2_weights/nanobody_model_2 https://zenodo.org/record/7258553/files/nanobody_model_2?download=1","wget -O NB2_weights/nanobody_model_3 https://zenodo.org/record/7258553/files/nanobody_model_3?download=1","wget -O NB2_weights/nanobody_model_4 https://zenodo.org/record/7258553/files/nanobody_model_4?download=1"))
    
app = modal.App(name="nanobodybuilder2", image=immune_builder_image)


# Remote function that predicts the PDBs
@app.function(image=immune_builder_image)
def predict_structure(sequences):
    from ImmuneBuilder import NanoBodyBuilder2
    import os
    # Model class
    predictor = NanoBodyBuilder2(weights_dir = "NB2_weights")
    results=[]
    assert isinstance(sequences, (str, list, tuple))

    # Iteratively Predict
    # TODO: Check batching
    sequences_mod = [sequences] if isinstance(sequences, str) else sequences
    for seq in sequences_mod:
        output_file = f"{seq}.pdb"
        nanobody = predictor.predict({'H': seq})
        nanobody.save(output_file)
        # Return PDB string
        with open(output_file,"r") as f:
            pdb_str=f.read()
            results.append(pdb_str)
        if os.path.isfile(output_file):
            os.remove(output_file)
    return results[0] if isinstance(sequences, str) else results


# Local function that saves PDBs
@app.local_entrypoint()
# Make remote function call and save predicted PDBs  
def save_pdbs(sequences, out_dir, ids):
    if isinstance(sequences, str):
        sequences=[sequences]
    if isinstance(ids, str):
        ids=[ids]
    results = predict_structure.remote(sequences)
    assert len(results) == len(sequences)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ids:
        assert len(ids) == len(sequences)
        for id, pdb_string in zip(ids, results):
            with open(out_dir / Path(f"{id}.pdb"), "w") as f:
                f.write(pdb_string)
    else:
        for i, pdb_string in enumerate(results):
            with open(out_dir / Path(f"{i}.pdb"), "w") as f:
                f.write(pdb_string)
                
