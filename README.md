# silica
Silica team repo for BioML hackathon

## Data

### EGFR Affinity Data

`[source_data/adaptyv_biolm_egfr_data.csv](source_data/adaptyv_biolm_egfr_data.csv)` 
contains data from Adaptyv Bio's EGFR Competition Round 1 (202 sequences), plus 
36 additional nanobody sequences ordered by BioLM.

  * 1 ScFV control Cetuximab
  * 3 nanobody control therapeutics
  * Approximately 10 antibody binders
  * 2 peptide binders

Columns are described as:

  * sequence: the AA sequence ordered
  * replicate: the number of replicate affinity assays performed
  * expression: Adaptyv's assessment of protein expression
  * binding: Adaptyv's call on antibody-antigen binding
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
  * iptm_tm through pi_score: computed multimer confidence, docking, other scores



