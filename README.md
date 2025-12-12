# 🧬 CAFA-6 Protein Function Prediction — Data Analysis

This repository supports exploratory data analysis and model development for the **CAFA-6 Protein Function Prediction** competition on Kaggle. Here is the link for the dataset: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data

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
## Notebooks
| File | Description | Requirement | Execution Time | 
|----------------|-------------|-------------|-------------|
| **`sequence.ipynb`** | Baseline Model with ESM2 8M Embeddings + MLP | train_esm2_embeddings.parquet and test_esm2_embeddings.parquet| / | 
| **`CV.ipynb`** | Cross Validation Experiment on ESM2 650M and ProtBERT Embeddings with Linear Regression and MLP Model Only| esm2 650 embeddings and probert embeddings| 5hrs with full batch |
| **`hybridnet.ipynb`** | All embeddings trained on HybridNet |all emebddings file| 2 hours | 
| **`data_analysis.ipynb`** | exploration of dataset| cafa dataset | / |
| **`embeddings.ipynb`** | embeddings genearation| sequence fasta file |  2 - 12 hours|
| **`cafa6_train_cnn.ipynb`** | experiment of esm2 8m embeddings with CNN | train_esm2_embeddings.parquet | / | 
| **`demo.ipynb`** | demonstration on the subset of the data | train_demo.parquet and test_demo.parquet| / | 

## data_analysis.ipynb: Usage in Analysis

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

## How to Run `demo.ipynb`

This demo notebook runs a complete, lightweight version of our protein function prediction pipeline using precomputed ESM2-8M embeddings.

---

### 1. Ensure Required Files Are Present

Before running the notebook, confirm that the following files are in the **root directory** of the repository:

- `train_demo.parquet` — labeled demo training data  
- `test_demo.parquet` — unlabeled demo test data  
- `demo.ipynb`

`train_demo.parquet` contains protein IDs, ESM2-8M embeddings (320 dimensions), and a list of GO terms per protein.  
`test_demo.parquet` contains protein IDs and embeddings only.

---

### 2. Install Required Python Packages

If the required packages are not already installed, run:

```bash
pip install pandas torch scikit-learn numpy tqdm
```
Open demo.ipynb and run all cells from top to bottom. The notebook will automatically:
- Load the demo datasets
- Split the data into training and validation sets
- Train a 3-layer MLP classifier on ESM2-8M embeddings
- Evaluate performance using micro-F1
- Run inference on the demo test set
- No additional configuration is required.

---

## embeddings.ipynb: Embeddings Generation

We use Google Colab for embeddings generation.

---
1. Run first cell to install all dependencies
2. Upload sequence fasta file for embeddings generation
3. Change model name for Protein Lanaguage Model you want to use. Or you can choose to download embeddings from google drive: https://drive.google.com/drive/folders/11msGNZCKy4jFYsQDGpdO90rqMudv23zB?usp=sharing
   
| Model Name | Execution time on test set |
|----------------|-------------|
| **`facebook/esm2_t6_8M_UR50D`** | 3 hours with GPU |
| **`Rostlab/prot_bert`** | 6 hours with GPU |
| **`facebook/esm2_t33_650M_UR50D`** | 8 hours with A100 |

---

