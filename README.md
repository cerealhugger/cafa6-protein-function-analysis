# 🧬 CAFA-6 Protein Function Prediction — Data Analysis

This repository supports exploratory data analysis and model development for the **CAFA-6 Protein Function Prediction** competition on Kaggle.

Main work is performed in **`data_analysis.ipynb`**, using training and testing data from the competition dataset.

---

## 📁 Project Structure


---

## Dataset Description

| File / Folder | Description |
|----------------|-------------|
| **`Train/train_sequences.fasta`** | Contains **amino acid sequences** for proteins in the training set. Each sequence represents a known protein whose function (GO terms) has been annotated. |
| **`Train/train_terms.tsv`** | Maps each training protein to its corresponding **Gene Ontology (GO) terms**, representing experimentally verified protein functions. |
| **`Train/train_taxonomy.tsv`** | Provides **taxon IDs** (organism identifiers) for each protein in the training set. Useful for incorporating phylogenetic or organism-level features. |
| **`Train/go-basic.obo`** | The **Gene Ontology graph structure** in OBO format. Defines relationships between GO terms (e.g., “is_a”, “part_of”) and provides the hierarchical organization used for semantic similarity and evaluation. |
| **`Test/testsuperset.fasta`** | Contains **amino acid sequences** for proteins on which predictions should be made (test set). These sequences lack GO annotations. |
| **`Test/testsuperset-taxon-list.tsv`** | Lists **taxon IDs** corresponding to proteins in the test set. Allows integration of taxonomy-aware prediction strategies. |
| **`IA.tsv`** | Provides **information accretion (IA)** values for each GO term. Used during evaluation to weight precision and recall, giving more importance to specific or informative terms. |
| **`sample_submission.tsv`** | Template file showing the **correct submission format** required for Kaggle. It lists protein IDs and example predicted GO terms. |

---

## Usage in Analysis

| Component | Typical Use in `data_analysis.ipynb` |
|------------|--------------------------------------|
| **FASTA files** | Parsed with `Bio.SeqIO` from Biopython to extract protein IDs and sequences. |
| **`train_terms.tsv`** | Joined with sequence IDs to build the training label set for supervised learning. |
| **`train_taxonomy.tsv` / `testsuperset-taxon-list.tsv`** | Used for grouping proteins by organism or exploring evolutionary signal. |
| **`go-basic.obo`** | Loaded via `goatools` to compute semantic similarity, term depth, and ontology-based metrics. |
| **`IA.tsv`** | Used for evaluation metrics that apply information-content weighting (e.g., weighted Fmax). |
| **`sample_submission.tsv`** | Used to verify correct output formatting before submitting predictions. |

---
Quick Reference Summary
Variable	Type	Description
train_terms	DataFrame	Protein → GO term mapping
train_taxonomy	DataFrame	Protein → Taxonomy mapping
IA	DataFrame	GO term → Information Accretion
aa_df	DataFrame	Protein → Amino acid frequency profile
extract_uniprot_id()	Function	Parses UniProt ID from FASTA record string

## Notebooks
| File | Description |
|----------------|-------------|
| **`sequence.ipynb`** | Baseline Model with ESM2 8M Embeddings + MLP |
| **`CV.ipynb`** | Cross Validation Experiment on ESM2 650M and ProtBERT Embeddings with Linear Regression and MLP Model Only|
| **`hybridnet.ipynb`** | All embeddings trained on HybridNet |
| **`data_analysis.ipynb`** | exploration of dataset|
| **`embeddings.ipynb`** | embeddings genearation|
| **`cafa6_train_cnn.ipynb`** | experiment of esm2 8m embeddings with CNN |



---
