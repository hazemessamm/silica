# silica
Silica team repo for the Evolved-2024 BioML hackathon

## Projects

Projects were split over lipase enzyme and EGFR binding nanobody design.
Both projects consisted of two main components with the ultimate goal of synthesizing top sequences and providing more wet lab data for various ML methods.
Candidate Generation:
This included techniques such as constrained diffusion, inversefolding and genetic algorithms including the use of ESM3. These generated seqs can be found [here](./generated_seqs)
Candidate ranking/scoring
This included log probabilites computed by several pretrained methods as well as some structural based metrics over predicted binding compelexes and embedding based distance measures. 

### Lipase
Methods
[ProstT5 generation](./modal/prostt5) and [Variant stability prediction](./silica/stability_training.py)

### EGFR Nanobody Binders

Methods
[ESM3 Example](./esm3) and [ESM3](./modal/esm3_nanobodybuilder) for generation and both a [ESM3 generation and filtering method with nanoBERT embeddings](https://github.com/hazemessamm/silica/tree/vhh-fixed-fr/nanobody_fixed_fr). [AntiFold generation and scoring](./modal/antifold_nanobody). [MPNN inversefolding](./mpnn). [Motif scaffolded diffusion generation](./modal/diffusion_pipeline). [GA candidate generation](./nanobert_ga). [Nanobody structure prediction for self consistency](./modal/nanobodybuilder). [NanoBERT log probability for scoring](./modal/nanobert)

### Scoring for Both Projects
[ESM2 LP](./modal/esm2) and [ESM3 LP](./modal/esm3)

## Data

### EGFR Affinity Data

[`source_data/adaptyv_biolm_egfr_data.csv`](source_data/adaptyv_biolm_egfr_data.csv) 
contains data from Adaptyv Bio's EGFR Competition Round 1 (202 sequences), plus 
36 additional nanobody sequences ordered by BioLM.

  * 1 ScFV control Cetuximab
  * 3 nanobody control therapeutics
  * Approximately 10 antibody binders
  * 2 peptide binders

Columns are described as:

  * sequence: the AA sequence ordered
  * replicate: the number of replicate affinity assays performed
  * expression: Adaptyv's assessment of protein expression (use combined_expression)
  * binding: Adaptyv's call on antibody-antigen binding (use combined_expression)
  * confidence: Adaptyv's assessment of confidence in calls
  * kd: Kd measurement for antibody-antigen
  * kon/koff: Kon and Koff for the same
  * normalized_sequence_lp: ESM2 sequence log probability normalized by length
  * source: Adaptyv Bio competition or BioLM order
  * metadata: temperature, recovery, scores, other data from BioLM sequence design
  * parent: closest known therapeutic antibody for BioLM design
  * sequence_lp: ESM2 sequence LP
  * tm_prediction: predicted Tms of sequence
  * aliphatic_index through mz: computed physiochemical properties from sequence
  * combined_expression and combined_binding: BioLM's assessment of sequence across replicates
  * iptm_tm through pi_score: computed multimer confidence, docking, other scores



