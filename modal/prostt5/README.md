# Setup

## Install Modal client
```pip install --upgrade modal-client```

## Authenticate Modal client
```modal token new```

## Difference between `app.py` and `app_with_local_foldseek.py`
* In `app.py`, you will pass the PDB ID and it will be downloaded via Alphafold database.
* In `app_with_local_foldseek.py`, you will pass the local path of the PDB from your machine, you will have to have foldseek installed locally, this app will run foldseek locally on your machine on the target file then it will run ProstT5 on Modal.


## Run the script
```modal run app.py --query-id B1PN84```
OR
```modal run app_with_local_foldseek.py --pdb path/to/the/pdb/file/4krl.py```
