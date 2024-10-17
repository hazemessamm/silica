# Usage: 
# git clone https://github.com/RosettaCommons/RFdiffusion.git
# cd RFdiffusion
# docker build -f docker/Dockerfile -t rfdiffusion .
# mkdir $HOME/inputs $HOME/outputs $HOME/models
# bash scripts/download_models.sh $HOME/models
# wget -P $HOME/inputs https://files.rcsb.org/view/5TPN.pdb

# docker run -it --rm --gpus all \
#   -v $HOME/models:$HOME/models \
#   -v $HOME/inputs:$HOME/inputs \
#   -v $HOME/outputs:$HOME/outputs \
#   rfdiffusion \
#   inference.output_prefix=$HOME/outputs/motifscaffolding \
#   inference.model_directory_path=$HOME/models \
#   inference.input_pdb=$HOME/inputs/5TPN.pdb \
#   inference.num_designs=3 \
#   'contigmap.contigs=[10-40/A163-181/10-40]'

FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Copy the entire project to /app/RFdiffusion
COPY . /app/RFdiffusion/

# Print the working directory after setting WORKDIR and list its contents
RUN echo "Working directory after setting WORKDIR:" && pwd && echo "Contents of the directory:" && ls -la

#print the contents of the /app/RFdiffusion directory
RUN ls -la /app/RFdiffusion

# Install CUDA Developer Tools, which include NVTX support
RUN apt-get update && apt-get install -y \
  cuda-nvtx-11-6

RUN apt-get -q update \ 
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  git \
  python3.9 \
  python3-pip \
  libgomp1 \
  && python3.9 -m pip install -q -U --no-cache-dir pip \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean \
  && pip install -q --no-cache-dir \
  dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
  # torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
  e3nn==0.3.3 \
  wandb==0.12.0 \
  pynvml==11.0.0 \
  git+https://github.com/NVIDIA/dllogger#egg=dllogger \
  decorator==5.1.0 \
  hydra-core==1.3.2 \
  pyrsistent==0.19.3 \
  /app/RFdiffusion/env/SE3Transformer \
  && pip install --no-cache-dir /app/RFdiffusion --no-deps

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /app/RFdiffusion

#list the contents of the /app/RFdiffusion directory
RUN ls -la

ENV DGLBACKEND="pytorch"

# Create symbolic links to ensure python3.9 and pip are default
RUN ln -sf /usr/bin/python3.9 /usr/bin/python \
  && ln -sf /usr/bin/pip3 /usr/bin/pip

# Verify Python and pip installations
RUN python --version \
  && pip --version



# ENTRYPOINT ["python3.9", "scripts/run_inference.py"]
