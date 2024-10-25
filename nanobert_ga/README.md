# GA with nanoBERT 
### Project Objectives
1. **Key Goals:**  
   The primary objective was to develop a Genetic Algorithm that directs initial parent sequences towards functional and diverse CDR3 variants.

### Therapeutic Focus
1. **Understanding EGFR:**  
   EGFR (Epidermal Growth Factor Receptor) is a cell receptor that plays a critical role in certain cancers. Targeting EGFR can disrupt signaling pathways that lead to uncontrolled cell growth.

2. **Challenges in Targeting EGFR:**  
   EGFR mutations can lead to therapeutic resistance, necessitating rapid redesign of therapeutics. Additionally, navigating tumor microenvironments (TME) is complex, as solid tumors often possess dense extracellular matrices (ECM) and abnormal blood vessel architecture that hinder drug delivery. Nanobodies can exhibit enhanced diffusion in these environments.


## Data Resources
1. **Data Sources:**  
   The primary data source used was an EGFR-targeting parent nanobody, with PDB accession code **4KRL**.



3. **Data Splitting for Analysis:**  
   Due to the scarcity of diverse representative EGFR nanobody data, zeroshot methods were employed for analysis but surrogate models and other specifically trained methods can be readily integrated.

## Methods

### Computational Infrastructure
1. **Compute Resources:**  
   The project utilized **Modal** and a local **GPU** for computational tasks although everything can run on  **CPU**.

2. **Computational Expense:**  
   The pipeline was not computationally expensive. NanoBERT, which runs efficiently on both GPU and CPU, was fully operable on Modal. The ESM3 model was accessed via the Forge API. GAs are generally less resource-intensive compared to combinatorial approaches.

### Design Pipeline Architecture
The following outlines the steps of the pipeline, detailing inputs, outputs, and key considerations:

1. **GA Implementation**
   - **Input:** Parent sequence.
   - **Mutation Operator:** Utilized NanoBERT to preferentially mask variable CDR regions of the input nanobody. The algorithm iteratively searched for higher-scoring mutations before applying random mutations, preferentially targeting on CDR loop sampling.
   - **Scoring:** Mutated variants were scored using NanoBERT and ESM3 log probabilities, which demonstrated some association with binding in previous EGFR data assays.
   - **Crossover and Mutation:** Implemented two-point crossovers and additional mutations to the best individuals determined by tournament selection to generate the next generation.
   - **Repetition:** The process was repeated for a specified number of generations and runs.
   - **Output:** All mutated sequence variants were produced for subsequent filtering and scoring.

   **Note:** Many parameters and configurations for this algorithm were not fully explored due to time constraints.

```
Johannes Thorling Hadsund, Tadeusz Satława, Bartosz Janusz, Lu Shan, Li Zhou, Richard Röttger, Konrad Krawczyk, nanoBERT: a deep learning model for gene agnostic navigation of the nanobody mutational space, Bioinformatics Advances, Volume 4, Issue 1, 2024, vbae033, https://doi.org/10.1093/bioadv/vbae033


Simulating 500 million years of evolution with a language model
Thomas Hayes, Roshan Rao, Halil Akin, Nicholas J. Sofroniew, Deniz Oktay, Zeming Lin, Robert Verkuil, Vincent Q. Tran, Jonathan Deaton, Marius Wiggert, Rohil Badkundri, Irhum Shafkat, Jun Gong, Alexander Derry, Raul S. Molina, Neil Thomas, Yousuf Khan, Chetan Mishra, Carolyn Kim, Liam J. Bartie, Matthew Nemeth, Patrick D. Hsu, Tom Sercu, Salvatore Candido, Alexander Rives
bioRxiv 2024.07.01.600583; doi: https://doi.org/10.1101/2024.07.01.600583 
```