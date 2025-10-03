# Deep Ocean Analysis Pipeline

A bioinformatics pipeline for analyzing deep-sea sequencing data using AI-powered embedding models, similarity search, and clustering to identify both known organisms and discover novel species clusters.

## 🎯 Project Goals

Given deep-sea sequencing reads or contigs, this pipeline produces:

1. **Similarity Analysis (Known Species)**: Per-sequence scores/labels identifying the most similar known sequences or taxa with confidence metrics
2. **Unknown Grouping**: Coherent clusters of sequences not confidently matched to known records, enabling researchers to inspect candidate novel groups

> **Note**: This is a clustering and representation pipeline, not an automated discovery system. All candidate novel clusters require manual expert review.

---

## 🏗️ Architecture Overview

### Core Technologies

- **FAISS**: Fast similarity search and clustering of sequence embeddings (efficient nearest-neighbor search)
- **Cerebras**: Heavy model training, fine-tuning, and high-throughput inference for large DNA transformer models
- **Exa**: Programmatic web search and literature lookup to enrich clusters with taxonomic context and recent research

### Pipeline Stages

```
Input Data → Preprocessing → Embedding → Similarity Branch (Known)
                                      ↓
                                Clustering Branch (Unknown)
                                      ↓
                            Reports & Visualizations
```

---

## 📋 Detailed Pipeline

### 1. Data Input

- **Sequences**: Raw reads or assembled contigs (FASTQ/FASTA format)
- **Metadata** (optional): Location, depth, sample ID, collection date

### 2. Preprocessing

- **Quality Control**: 
  - Adapter trimming using `fastp` or `cutadapt`
  - Quality filtering and chimera removal (for amplicon data)
- **Assembly** (if needed):
  - Assemble short reads into contigs using `SPAdes` or `metaSPAdes`
- **Target Extraction**:
  - Extract target regions (e.g., 16S/18S rRNA) for amplicon data
  - Use whole sequences for metagenomic analysis

### 3. Reference Database Indexing

Pull and index authoritative databases:
- **GenBank/NCBI**: General genomic sequences
- **SILVA**: Ribosomal RNA sequences (16S/18S)

Build labeled reference embedding index for similarity matching.

### 4. Embedding Generation

Convert nucleotide sequences → dense vector embeddings using DNA foundation models:

**Recommended Models**:
- **DNABERT / DNABERT-2**: Specialized for DNA sequence representation
- **Nucleotide Transformer**: Alternative foundation model family

**Process**:
- Tokenize sequences (k-mer or character-level)
- Use sliding windows for long sequences
- Pool embeddings (CLS token or mean pooling) to create per-sequence vectors

### 5. Similarity Branch (Known Species)

For each query sequence:

1. **Index Search**: Compute cosine similarity / k-nearest neighbors in FAISS embedding index
2. **Threshold Check**:
   - If `max_similarity > threshold` → report match (taxon/accession) + confidence
   - If `max_similarity < threshold` → route to clustering branch
3. **Verification** (optional): Run BLAST alignment for high-confidence matches

**FAISS Index Choices**:
- **Development**: `IndexFlatIP` (exact inner-product search)
- **Production**: `IVF,PQ` indices for >1M vectors (memory-efficient)

### 6. Clustering Branch (Unknowns)

For sequences below similarity threshold:

1. **Dimensionality Reduction**: UMAP (or PCA) for noise reduction and visualization
2. **Density Clustering**: HDBSCAN for robust, variable-density clusters
3. **Cluster Summaries**:
   - Consensus k-mers
   - Representative sequences (centroid/medoid)
   - Abundance across samples
   - Nearest-known neighbors (even if below threshold)

### 7. Post-processing & Outputs

**Reports**:
- Per-sequence predictions (match/no-match + confidence scores)
- Cluster assignments for unknown sequences
- Abundance tables

**Visualizations**:
- UMAP scatter plots (colored by cluster/taxon)
- Cluster dendrograms
- Confidence distribution plots

**Exports**:
- Representative sequences (FASTA)
- Cluster metadata (CSV/JSON)
- Ready for phylogenetic analysis (MAFFT alignment + IQ-TREE/FastTree)

---

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

### Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core packages
pip install torch transformers
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install umap-learn hdbscan
pip install biopython pandas numpy scipy matplotlib seaborn

# Preprocessing tools (install via conda or system package manager)
conda install -c bioconda fastp cutadapt spades blast

# Optional: Exa API client
pip install exa_py
```

### API Keys

Create a `.env` file in the project root:

```env
# Cerebras Cloud (for training/inference)
CEREBRAS_API_KEY=your_cerebras_api_key_here

# Exa (for literature lookup)
EXA_API_KEY=your_exa_api_key_here
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Place your FASTA/FASTQ files in the `data/` directory:

```
data/
├── raw_sequences.fasta
└── metadata.csv
```

### 2. Run Quality Control

```bash
# Quality trimming
fastp -i data/raw_sequences.fastq \
      -o data/clean_sequences.fastq \
      --html data/qc_report.html
```

### 3. Generate Embeddings

```python
python scripts/generate_embeddings.py \
    --input data/clean_sequences.fasta \
    --model zhihan1996/DNABERT-2-117M \
    --output embeddings/query_embeddings.npy
```

### 4. Build Reference Index

```python
python scripts/build_faiss_index.py \
    --reference data/references/silva_database.fasta \
    --output indexes/silva_index.faiss
```

### 5. Run Similarity Search

```python
python scripts/similarity_search.py \
    --query embeddings/query_embeddings.npy \
    --index indexes/silva_index.faiss \
    --threshold 0.85 \
    --output results/similarity_matches.csv
```

### 6. Cluster Unknowns

```python
python scripts/cluster_unknowns.py \
    --embeddings embeddings/unknown_embeddings.npy \
    --output results/clusters/ \
    --visualize
```

---

## 📊 Configuration

### Similarity Threshold Calibration

Calibrate the similarity threshold using a validation set:

```python
python scripts/calibrate_threshold.py \
    --validation data/validation_labeled.fasta \
    --index indexes/silva_index.faiss \
    --output config/threshold_roc.png
```

Recommended starting threshold: `0.80 - 0.85` (adjust based on ROC curve)

### FAISS Index Configuration

```python
# config/faiss_config.yaml
index_type: "IVF,PQ"  # For large-scale production
nlist: 1024           # Number of clusters for IVF
m: 16                 # PQ code size
nbits: 8              # Bits per sub-quantizer
```

### Clustering Parameters

```python
# config/clustering_config.yaml
umap:
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
  metric: "cosine"

hdbscan:
  min_cluster_size: 10
  min_samples: 5
  cluster_selection_epsilon: 0.0
```

---

## 🧪 Cerebras Integration

### Fine-tuning DNA Models

Use Cerebras for training/fine-tuning large DNA transformers:

```python
# Train on Cerebras Cloud
python scripts/train_dnabert.py \
    --config config/cerebras_training.yaml \
    --platform cerebras-cloud \
    --output models/dnabert_finetuned/
```

**When to use Cerebras**:
- Fine-tuning on >1M sequences
- Long-context sequences (>1000bp)
- Large batch inference
- Low-latency production serving

**Deployment Options**:
- Cerebras Cloud (pay-as-you-go)
- On-premises CS-3 / CS-2 systems

### Batch Embedding Generation

```python
# High-throughput inference on Cerebras
python scripts/cerebras_inference.py \
    --input data/large_dataset.fasta \
    --model models/dnabert_finetuned/ \
    --batch-size 1024 \
    --output embeddings/
```

---

## 🔍 Exa Integration

### Literature Lookup for Clusters

Enrich cluster results with recent taxonomic literature:

```python
python scripts/exa_literature_lookup.py \
    --clusters results/clusters/cluster_summary.csv \
    --output results/clusters/literature_context.json
```

**Example API Usage**:

```python
from exa_py import Exa

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# Search for recent papers on a cluster's nearest-known taxon
results = exa.search_and_contents(
    query=f"taxonomy {taxon_name} deep sea novel species",
    type="neural",
    num_results=5,
    text=True
)

# Attach citations to cluster report
for result in results.results:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Summary: {result.text[:200]}...")
```

---

## 📈 Evaluation & Validation

### Known-Similarity Branch

Metrics:
- Top-1 / Top-5 accuracy on held-out labeled sequences
- Precision / Recall vs. manual BLAST results
- ROC curve for threshold calibration

```python
python scripts/evaluate_similarity.py \
    --validation data/validation_labeled.fasta \
    --predictions results/similarity_matches.csv \
    --output results/evaluation_report.pdf
```

### Clustering Branch

Metrics:
- Silhouette score (cluster cohesion)
- Cluster stability across bootstrap samples
- Phylogenetic coherence (MAFFT + tree reconstruction)

**Manual Review**:
- Every candidate cluster requires expert examination
- Check representative alignments
- Verify sample co-occurrence patterns

---

## 📁 Project Structure

```
DeepOceanAnalysis/
├── README.md                 # This file
├── main.py                   # Main pipeline orchestrator
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── config/                   # Configuration files
│   ├── faiss_config.yaml
│   ├── clustering_config.yaml
│   └── cerebras_training.yaml
├── data/                     # Input data (not committed)
│   ├── raw_sequences.fasta
│   ├── references/
│   └── metadata.csv
├── scripts/                  # Pipeline scripts
│   ├── generate_embeddings.py
│   ├── build_faiss_index.py
│   ├── similarity_search.py
│   ├── cluster_unknowns.py
│   ├── calibrate_threshold.py
│   ├── exa_literature_lookup.py
│   └── train_dnabert.py
├── embeddings/               # Generated embeddings (not committed)
├── indexes/                  # FAISS indices (not committed)
├── results/                  # Output reports and visualizations
│   ├── similarity_matches.csv
│   ├── clusters/
│   └── plots/
└── notebooks/                # Jupyter notebooks
    └── analysis_walkthrough.ipynb
```

---

## 🔬 Example Workflow

### Complete Analysis Pipeline

```bash
# 1. Quality control
fastp -i data/raw_reads.fastq -o data/clean_reads.fastq

# 2. (Optional) Assembly
metaspades.py -1 data/clean_reads.fastq -o data/assembly/

# 3. Generate embeddings
python scripts/generate_embeddings.py \
    --input data/assembly/contigs.fasta \
    --model zhihan1996/DNABERT-2-117M \
    --output embeddings/query_emb.npy

# 4. Similarity search
python scripts/similarity_search.py \
    --query embeddings/query_emb.npy \
    --index indexes/genbank_silva_index.faiss \
    --threshold 0.82 \
    --output results/matches.csv

# 5. Cluster unknowns
python scripts/cluster_unknowns.py \
    --embeddings embeddings/unknown_emb.npy \
    --output results/clusters/ \
    --visualize

# 6. Enrich with literature
python scripts/exa_literature_lookup.py \
    --clusters results/clusters/summary.csv \
    --output results/clusters/literature.json

# 7. Generate final report
python scripts/generate_report.py \
    --results results/ \
    --output final_report.html
```

---

## 📚 Key References

### Core Technologies

- **FAISS**: [GitHub - facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **Cerebras**: [Cerebras Cloud Platform](https://www.cerebras.net/)
- **Exa**: [Exa.ai - AI Search API](https://exa.ai/)

### DNA Language Models

- **DNABERT**: Ji et al., "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers for DNA-language in Genome"
- **DNABERT-2**: Zhou et al., "DNABERT-2: Efficient Foundation Model for Multi-Species Genome"
- **Nucleotide Transformer**: Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics"

### Clustering & Visualization

- **UMAP**: McInnes et al., "UMAP: Uniform Manifold Approximation and Projection"
- **HDBSCAN**: McInnes et al., "hdbscan: Hierarchical density based clustering"

### Reference Databases

- **GenBank**: [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)
- **SILVA**: [SILVA rRNA Database](https://www.arb-silva.de/)

---

## 🤝 Contributing

This pipeline is designed for deep-sea genomics research. Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -m 'Add new analysis module'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

---

## ⚠️ Important Notes

### Manual Review Required

**No automated novelty claims**: All "unknown" clusters represent candidate groups that require:
- Expert taxonomic review
- Phylogenetic validation (alignment + tree reconstruction)
- Cross-reference with recent literature
- Sample metadata correlation analysis

### Data Privacy

- Never commit raw sequencing data or API keys to version control
- Use `.gitignore` to exclude `data/`, `embeddings/`, and `.env`
- Follow institutional data-sharing policies

### Computational Requirements

- **Minimum**: 16GB RAM, 4-core CPU for small datasets (<10k sequences)
- **Recommended**: 64GB RAM, GPU (CUDA-enabled) for large-scale analysis
- **Production**: Cerebras Cloud or CS-3 for training/fine-tuning

---

## 📝 License

This project is provided as-is for research purposes. Please cite appropriately when using this pipeline in publications.

---

## 📧 Contact & Support

For questions, issues, or collaboration:
- Open an issue on GitHub
- Contact the bioinformatics team

**Happy sequencing! 🌊🧬**
