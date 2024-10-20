# RFdiffusion Setup Guide

## Getting Started

To start using RFdiffusion, begin by cloning the repository:

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
```

After cloning, you will need to download the model weights into the `RFdiffusion` directory.

### Step 1: Set Up the Directory

Navigate to the `RFdiffusion` directory:

```bash
cd RFdiffusion
```

Create a `models` directory and navigate into it:

```bash
mkdir models && cd models
```

### Step 2: Download Model Weights

Use the following commands to download the necessary model weights:

```bash
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt
```

#### Optional Weight

You may also choose to download an additional optional checkpoint:

```bash
wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt
```

#### Original Structure Prediction Weights

Additionally, download the original structure prediction weights:

```bash
wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt
```

#### Add an initial structure

Create an 'input/' directory in RFDiffusion and copy in your starting file. Update db_path in the main function of 'final_pipeline.py' to match the path of the input pdb.

### Step 3: Environment Setup with Modal

Instantiate and activate the conda environment using Modal:

```bash
python final_pipeline.py
```

> **Note:** The `final_pipeline.py` script is intended to demonstrate how to set up and run RFdiffusion. Ensure you have Modal properly configured before executing this step.Now you are ready to start using RFdiffusion.