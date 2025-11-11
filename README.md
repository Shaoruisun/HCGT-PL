# HCGT-PL: A Heterogeneous Contrastive Graph Transformer Unifying Protein–Ligand Affinity Prediction and Structure-Based Virtual Screening

Structure-based virtual screening and binding affinity prediction remain challenging due to solvation/entropy effects, protein flexibility, and induced fit. We present a Heterogeneous Contrastive Graph Transformer for Protein-Ligand (HCGT-PL) framework. Each complex is represented as a directed heterogeneous graph with multiple node and relation types; relation-specific multi-head attention enables message passing and aggregation. Unsupervised augmentations yield transferable interaction representations that are fine-tuned for affinity regression and virtual screening. Across diverse benchmarks and hold-out evaluations, the approach delivers robust accuracy, strong ranking capability, and pronounced early-enrichment, with consistent generalization over varied protein families and pocket conditions. Interpretability visualizations indicate that the model prioritizes ligand functional groups and contacting receptor side chains within the binding pocket. By unifying heterogeneous graph modeling, graph Transformers, and contrastive learning, this framework provides a general, transferable, and interpretable solution for protein–ligand modeling.

## Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install dgl dgllife
pip install pandas numpy scikit-learn
pip install rdkit  # For advanced molecular metrics
```

## 📊 Datasets

The models are designed to work with:
- **PDBbind 2016**: Primary dataset for affinity prediction
- **CASF 2013/2016**: Benchmark datasets for evaluation
- **DUD-E Dataset**: For virtual screening tasks

### Data Format
- **Graph Files**: Binary DGL graph files (`.bin`)
- **ID Files**: NumPy arrays containing PDB IDs (`.npy`)
- **Labels**: CSV files with binding affinities or binary labels

## 🔧 Configuration

### Key Parameters
- `--epochs`: Number of training epochs (default: 2000)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 256)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--dropout`: Dropout rate (default: 0.05)
- `--weight_decay`: Weight decay for regularization (default: 1e-6)

### Model Selection
- `--model_type`: Model architecture (default: 'DTI')
- `--data_type`: Dataset type (default: 'PDBbind2016')
- `--load_model_path`: Path to pre-trained model for transfer learning

## 📈 Evaluation Metrics

### Regression Metrics (Affinity Prediction)
- **RMSE**: Root Mean Square Error
- **Pearson Correlation**: Linear correlation coefficient
### Classification Metrics (Virtual Screening)
- **Enrichment Factor**: Early recognition capability

