# Usage Examples

## Quick Start

### 1. Basic Analysis (Local Data)

```bash
# Analyze a FASTA file against a reference database
python scripts/analyze_pipeline.py \
    --input data/raw/deep_sea_samples.fasta \
    --reference data/raw/silva_16S.fasta \
    --output data/results/
```

### 2. Fetch Data from NCBI First

```bash
# Search and download deep-sea 16S sequences
python scripts/fetch_ncbi.py \
    --query "deep sea 16S rRNA" \
    --database nucleotide \
    --max-results 1000 \
    --output data/raw/ncbi_deepsea.fasta

# Run analysis
python scripts/analyze_pipeline.py \
    --input data/raw/ncbi_deepsea.fasta \
    --output data/results/
```

## Step-by-Step Workflow

### Step 1: Generate Embeddings

```bash
# Generate embeddings for query sequences
python scripts/embed_sequences.py \
    --input data/raw/queries.fasta \
    --output data/embeddings/queries.npy \
    --config config.yaml
```

### Step 2: Build FAISS Index (One-Time)

```bash
# Build index from reference sequences
python scripts/build_faiss.py \
    --build \
    --embeddings data/embeddings/references.npy \
    --index data/faiss_index/reference.faiss
```

### Step 3: Search for Similar Sequences

```bash
# Search index for matches
python scripts/build_faiss.py \
    --search \
    --query data/embeddings/queries.npy \
    --index data/faiss_index/reference.faiss \
    --k 10
```

### Step 4: Cluster Unknown Sequences

```bash
# Cluster sequences with low similarity
python scripts/cluster_unknowns.py \
    --embeddings data/embeddings/unknowns.npy \
    --output data/clusters/ \
    --visualize
```

### Step 5: Generate Report

```bash
# Create HTML report
python scripts/report_generator.py \
    --results data/results/ \
    --output data/results/final_report.html
```

## Advanced Usage

### Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
model:
  name: "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
  max_length: 1024
  
similarity:
  known_threshold: 0.90  # Stricter threshold
  
clustering:
  hdbscan:
    min_cluster_size: 20  # Larger clusters
```

Run with custom config:

```bash
python scripts/analyze_pipeline.py \
    --input data.fasta \
    --config my_config.yaml
```

### Using Cerebras for Large-Scale Inference

Enable Cerebras in config:

```yaml
# config.yaml
model:
  use_cerebras: true
```

Set API key:

```bash
export CEREBRAS_API_KEY="your_key_here"
```

### BLAST Verification

Enable BLAST verification in config:

```yaml
# config.yaml
blast:
  enabled: true
  e_value: 1e-10
  max_target_seqs: 5
```

Download BLAST database:

```bash
python scripts/fetch_ncbi.py \
    --download-blast-db nt
```

### Literature Enrichment with Exa

Enable Exa in config:

```yaml
# config.yaml
exa:
  enabled: true
  num_results: 10
```

Set API key:

```bash
export EXA_API_KEY="your_key_here"
```

## Example Workflows

### Workflow 1: Deep-Sea Microbiome Analysis

```bash
# 1. Fetch reference 16S sequences
python scripts/fetch_ncbi.py \
    --query "marine microbiome 16S rRNA" \
    --max-results 5000 \
    --output data/raw/marine_refs.fasta

# 2. Generate reference embeddings and build index
python scripts/embed_sequences.py \
    --input data/raw/marine_refs.fasta \
    --output data/embeddings/marine_refs.npy

python scripts/build_faiss.py \
    --build \
    --embeddings data/embeddings/marine_refs.npy \
    --index data/faiss_index/marine.faiss

# 3. Analyze your deep-sea samples
python scripts/analyze_pipeline.py \
    --input data/raw/my_deep_sea_samples.fasta \
    --output data/results/microbiome_analysis/
```

### Workflow 2: Novel Species Discovery

```bash
# 1. Use existing SILVA database as reference
python scripts/embed_sequences.py \
    --input data/raw/SILVA_database.fasta \
    --output data/embeddings/silva.npy

# 2. Build index
python scripts/build_faiss.py \
    --build \
    --embeddings data/embeddings/silva.npy \
    --index data/faiss_index/silva.faiss

# 3. Analyze with strict threshold for unknowns
python scripts/analyze_pipeline.py \
    --input data/raw/deep_ocean_vent_samples.fasta \
    --output data/results/novel_discovery/ \
    --config config_strict.yaml  # Use higher threshold

# 4. Review cluster summaries
cat data/results/novel_discovery/clusters/cluster_summary.csv
```

### Workflow 3: Time-Series Analysis

```bash
# Process multiple time points
for timepoint in T1 T2 T3 T4; do
    echo "Processing $timepoint..."
    
    python scripts/analyze_pipeline.py \
        --input data/raw/samples_${timepoint}.fasta \
        --output data/results/${timepoint}/ \
        --config config.yaml
done

# Compare results across time points
python scripts/compare_timepoints.py \
    --results data/results/T*/ \
    --output data/results/timeseries_comparison.html
```

## Jupyter Notebook Examples

### Interactive Exploration

```python
# notebooks/00_exploration.ipynb
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt

# Load sequences
sequences = list(SeqIO.parse('data/raw/samples.fasta', 'fasta'))

# Basic stats
lengths = [len(seq) for seq in sequences]
plt.hist(lengths, bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Sequence Length Distribution')
plt.show()
```

### Custom Analysis Pipeline

```python
# notebooks/01_embedding.ipynb
from scripts.embed_sequences import SequenceEmbedder
import numpy as np

# Initialize embedder
embedder = SequenceEmbedder('config.yaml')

# Embed sequences
result = embedder.embed_fasta('data/raw/queries.fasta')

# Inspect embeddings
print(f"Shape: {result['embeddings'].shape}")
print(f"Sequences: {len(result['ids'])}")

# Compute pairwise similarities
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(result['embeddings'])
print(f"Mean similarity: {sim_matrix.mean():.4f}")
```

### Visualize Clusters

```python
# notebooks/03_clustering.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cluster results
labels = pd.read_csv('data/clusters/cluster_labels.csv')
summary = pd.read_csv('data/clusters/cluster_summary.csv')

# Plot cluster size distribution
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x='cluster_id', y='size')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Sequences')
plt.title('Cluster Size Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cluster_sizes.png', dpi=300)
```

## Performance Tuning

### For Small Datasets (<1K sequences)

```yaml
# config.yaml
model:
  batch_size: 16
  device: "cpu"

faiss:
  index_type: "IndexFlatIP"  # Exact search
```

### For Medium Datasets (1K-100K sequences)

```yaml
# config.yaml
model:
  batch_size: 64
  device: "cuda"

faiss:
  index_type: "IVF,Flat"
  nlist: 100
  nprobe: 10
```

### For Large Datasets (>100K sequences)

```yaml
# config.yaml
model:
  batch_size: 128
  device: "cuda"
  use_cerebras: true  # If available

faiss:
  index_type: "IVF,PQ"
  nlist: 1024
  nprobe: 20
  use_gpu: true
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
# Edit config.yaml:
#   model.batch_size: 8

# Or process in chunks
split -l 1000 data/raw/large.fasta data/raw/chunk_
for chunk in data/raw/chunk_*; do
    python scripts/analyze_pipeline.py --input $chunk --output data/results/
done
```

### Slow FAISS Search

```bash
# Use approximate search
# Edit config.yaml:
#   faiss.index_type: "IVF,PQ"
#   faiss.nprobe: 20
```

### BLAST Not Found

```bash
# Install BLAST+ toolkit
conda install -c bioconda blast

# Or disable BLAST verification
# Edit config.yaml:
#   blast.enabled: false
```

## Best Practices

1. **Start small**: Test pipeline on subset before full dataset
2. **Version control**: Track config files and results metadata
3. **Reproducibility**: Fix random seeds in config
4. **Validation**: Use held-out data to calibrate thresholds
5. **Documentation**: Keep notes on parameter choices and results
6. **Backups**: Save embeddings and indices (expensive to recompute)

## Getting Help

- Check logs: `logs/pipeline.log`
- Enable debug logging: `logging.level: "DEBUG"` in config
- Review test cases: `tests/`
- Consult documentation: `docs/architecture.md`
