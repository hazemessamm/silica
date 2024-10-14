# Use the official Miniconda3 image as a base
FROM continuumio/miniconda3

# Step 1: Update and install any necessary build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Install Python 3.10 using Conda
RUN conda install -y python=3.10

# Step 3: Set Python 3.10 as the system default
RUN ln -sf /opt/conda/bin/python3.10 /usr/bin/python && \
    ln -sf /opt/conda/bin/python3.10 /usr/bin/python3

# Step 4: Verify Python version and install pip
RUN python --version && python3 --version && \
    /opt/conda/bin/pip install --upgrade pip

# Step 5: Clean up
RUN conda clean -a -y

# Install libgcc-ng from conda-forge
RUN conda install -c conda-forge libgcc-ng libstdcxx-ng=12.2.0 -y

# Install hmmer and anarci from bioconda
RUN conda install -c bioconda hmmer -y && \
    conda install -c bioconda anarci -y && \
    conda install -c bioconda abnumber

# Install abnumber using pip
# RUN /bin/bash -c "source activate abnumber_env && pip install git+https://github.com/prihoda/abnumber.git"

# # Auto-activate the Conda environment by modifying .bashrc
# RUN echo "conda activate abnumber_env" >> ~/.bashrc

# Step 6: Set PYTHONPATH to ensure the system recognizes Conda-installed packages
ENV PYTHONPATH=/opt/conda/lib/python3.10/site-packages

# Step 7: Verify installation of abnumber
RUN python -c "import abnumber; print('abnumber installed successfully')"

# Set the default command to run bash
CMD ["/bin/bash"]

