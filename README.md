# HCGT-PL: A Heterogeneous Contrastive Graph Transformer Unifying Protein–Ligand Affinity Prediction and Structure-Based Virtual Screening

Structure-based virtual screening and binding affinity prediction remain challenging due to solvation/entropy effects, protein flexibility, and induced fit. We present a Heterogeneous Contrastive Graph Transformer for Protein-Ligand (HCGT-PL) framework. Each complex is represented as a directed heterogeneous graph with multiple node and relation types; relation-specific multi-head attention enables message passing and aggregation. Unsupervised augmentations yield transferable interaction representations that are fine-tuned for affinity regression and virtual screening. Across diverse benchmarks and hold-out evaluations, the approach delivers robust accuracy, strong ranking capability, and pronounced early-enrichment, with consistent generalization over varied protein families and pocket conditions. Interpretability visualizations indicate that the model prioritizes ligand functional groups and contacting receptor side chains within the binding pocket. By unifying heterogeneous graph modeling, graph Transformers, and contrastive learning, this framework provides a general, transferable, and interpretable solution for protein–ligand modeling.

## Getting Started

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- DGL >= 0.7.0
- CUDA (recommended for GPU acceleration)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Shaoruisun/HCGT-PL.git
cd HCGT-PL
```

2. Create a conda environment:
```bash
conda create -n hcgt-pl python=3.8
conda activate hcgt-pl
```

3. Install dependencies:
```bash
# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install DGL
conda install -c dglteam dgl-cuda11.3

# Install other dependencies
pip install rdkit-pypi
pip install prody
pip install prolif
pip install scikit-learn
pip install scipy
pip install pandas
pip install matplotlib
pip install seaborn
```

## Data Preparation

### Dataset Overview

HCGT-PL uses the following datasets:

1. **Pre-training**: 371,458 unlabeled protein-ligand complexes from BioLiP
2. **Affinity Prediction**:
   - PDBbind v2016 General Set + Refined Set (~13,000 complexes)
   - CASF-2016 Core Set (285 complexes)
   - CASF-2013 Core Set (195 complexes)
   - CSAR-HiQ (176 complexes)
3. **Virtual Screening**:
   - DUD-E (102 targets)
   - DEKOIS 2.0 (81 targets)
   - TrueDecoy & TrueDecoygap benchmarks

### Downloading Datasets

1. **PDBbind**: Download from [PDBbind website](http://www.pdbbind.org.cn/)
2. **BioLiP**: Download from [BioLiP database](https://zhanggroup.org/BioLiP/)
3. **DUD-E**: Download from [DUD-E website](http://dude.docking.org/)
4. **DEKOIS**: Download from [DEKOIS 2.0](https://www.zbh.uni-hamburg.de/forschung/amd/datasets/dekois.html)

### Graph Construction

The heterogeneous graph construction involves:

1. **Pocket Extraction**: Extract binding pocket residues within a cutoff radius (default: 5Å)
2. **Graph Nodes**:
   - Ligand atoms (node type: 'l')
   - Protein atoms (node type: 'p')
3. **Graph Edges**:
   - Intra-ligand covalent bonds ('ll')
   - Intra-protein covalent bonds ('pp')
   - Ligand→Protein contacts ('lp')
   - Protein→Ligand contacts ('pl')
4. **Node Features**: Atom type, degree, valence, hybridization, aromaticity, hydrogen count (35-dim)
5. **Edge Features**: Bond type, conjugation, ring membership, geometric features (17-dim)

Run the graph construction script:

```bash
cd Dataset
python graph_constructor.py \
    --dir /path/to/complexes \
    --outprefix output_graphs \
    --parallel
```

This generates `.bin` files containing DGL heterogeneous graphs and corresponding PDB IDs.

---

## Usage

### Dataset Processing

Before training, prepare your datasets by constructing heterogeneous graphs:

```python
from Dataset.graph_constructor import construct_graphs

# Example: Process PDBbind dataset
construct_graphs(
    complex_dir='Dataset/1.pdbbind_v2016/refined-set',
    output_prefix='out_pdbbind_2016',
    pocket_radius=5.0,
    parallel=True
)
```
### Pre-training

Pre-train the model using contrastive learning on unlabeled BioLiP data:

```bash
cd Pre-train

python train.py \
    --data_type BioLiP \
    --model_type DTI \
    --epochs 5000 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_dim0 128 \
    --hidden_dim 4096 \
    --temperature 0.05 \
    --save_prefix contrastive_pretrain
```

**Key Parameters**:
- `--temperature`: Temperature parameter for NT-Xent loss (default: 0.05)
- `--hidden_dim0`: Initial hidden dimension (128)
- `--hidden_dim`: Final embedding dimension (4096)
- `--epochs`: Number of pre-training epochs

Pre-trained checkpoints will be saved in `checkpoints/DTI_contrastive_pretrain/`.

### Fine-tuning for Affinity Prediction

Fine-tune the pre-trained model for binding affinity regression:

```bash
cd Affinity-model

python train.py \
    --data_type PDBbind2016 \
    --model_type DTI \
    --load_model_path ../Pre-train/checkpoints/DTI_contrastive_pretrain/best_model.pth \
    --epochs 2000 \
    --batch_size 64 \
    --lr 0.001 \
    --save_prefix affinity_finetune
```

### Virtual Screening

Train the model for structure-based virtual screening (binary classification):

```bash
cd Virtual_Screening_Model

python train.py \
    --data_type DUDE \
    --model_type DTI \
    --load_model_path ../Pre-train/checkpoints/DTI_contrastive_pretrain/best_model.pth \
    --epochs 1000 \
    --batch_size 32 \
    --lr 0.0005 \
    --save_prefix virtual_screening
```
