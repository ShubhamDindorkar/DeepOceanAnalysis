# Setup Guide

Complete setup instructions for the Deep-Sea NCBI Analysis Pipeline.

## Prerequisites

- **Python**: 3.8 or higher
- **Git**: For version control
- **Conda** (recommended): For managing bioinformatics tools

## Installation Steps

### 1. Clone Repository (if applicable)

```bash
git clone <repository-url>
cd DeepOceanAnalysis
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate

# Or using conda
conda create -n deepsea python=3.9
conda activate deepsea
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Bioinformatics Tools (Optional)

```bash
# Using conda (recommended)
conda install -c bioconda fastp cutadapt blast spades

# Or install individually from source
# See: 
# - fastp: https://github.com/OpenGene/fastp
# - BLAST+: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download
```

### 5. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# Get keys from:
# - Cerebras: https://cloud.cerebras.ai/
# - Exa: https://exa.ai/
# - NCBI (optional): https://www.ncbi.nlm.nih.gov/account/
```

Example `.env` file:
```
CEREBRAS_API_KEY=your_cerebras_key_here
EXA_API_KEY=your_exa_key_here
NCBI_API_KEY=your_ncbi_key_here  # Optional
```

### 6. Configure Pipeline

Edit `config.yaml` to customize:
- Model selection
- FAISS index type
- Similarity thresholds
- Clustering parameters
- File paths

```yaml
# config.yaml
model:
  name: "zhihan1996/DNABERT-2-117M"
  device: "cuda"  # or "cpu"
  
similarity:
  known_threshold: 0.82
  
# ... more settings
```

### 7. Test Installation

```bash
# Run tests
pytest tests/

# Or test individual components
python scripts/fetch_ncbi.py --help
python scripts/embed_sequences.py --help
```

## Quick Start Test

### Option 1: Using Sample Data

```bash
# Create sample FASTA file
cat > data/raw/test.fasta << EOF
>seq1
ATCGATCGATCGATCGATCG
>seq2
GCTAGCTAGCTAGCTAGCTA
EOF

# Generate embeddings
python scripts/embed_sequences.py \
    --input data/raw/test.fasta \
    --output data/embeddings/test.npy

# Build index
python scripts/build_faiss.py \
    --build \
    --embeddings data/embeddings/test.npy \
    --index data/faiss_index/test.faiss

echo "Setup successful!"
```

### Option 2: Fetch from NCBI

```bash
# Fetch small test dataset
python scripts/fetch_ncbi.py \
    --query "marine bacteria 16S rRNA" \
    --max-results 50 \
    --output data/raw/ncbi_test.fasta

# Run full pipeline
python scripts/analyze_pipeline.py \
    --input data/raw/ncbi_test.fasta \
    --output data/results/test_run/
```

## GPU Setup (Optional but Recommended)

### CUDA Installation

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
# Visit: https://pytorch.org/get-started/locally/

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FAISS GPU version
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Verify GPU

```python
import torch
import faiss

print(f"PyTorch CUDA: {torch.cuda.is_available()}")
print(f"FAISS GPU: {faiss.get_num_gpus()}")
```

## Cerebras Cloud Setup

### 1. Sign Up

Visit: https://cloud.cerebras.ai/

### 2. Get API Key

Navigate to API Keys section and create a new key.

### 3. Configure

Add to `.env`:
```
CEREBRAS_API_KEY=your_key_here
```

Enable in `config.yaml`:
```yaml
model:
  use_cerebras: true
```

### 4. Test Connection

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('CEREBRAS_API_KEY')
print(f"API Key configured: {api_key is not None}")
```

## Exa Setup

### 1. Sign Up

Visit: https://exa.ai/

### 2. Get API Key

Create an API key from the dashboard.

### 3. Configure

Add to `.env`:
```
EXA_API_KEY=your_key_here
```

Enable in `config.yaml`:
```yaml
exa:
  enabled: true
  num_results: 5
```

### 4. Test

```python
from exa_py import Exa
import os

exa = Exa(api_key=os.getenv('EXA_API_KEY'))
results = exa.search("deep sea bacteria taxonomy", num_results=3)
print(f"Found {len(results.results)} results")
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
pip install -r requirements.txt
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config
```yaml
model:
  batch_size: 8  # Reduce from default 32
```

### Issue: BLAST not found

**Solution**: Install BLAST+ or disable in config
```bash
conda install -c bioconda blast
# OR
# Edit config.yaml: blast.enabled = false
```

### Issue: Slow embedding generation

**Solution**: Use GPU or Cerebras
```yaml
model:
  device: "cuda"  # Use GPU
  # OR
  use_cerebras: true  # Use Cerebras Cloud
```

### Issue: NCBI rate limiting

**Solution**: Get NCBI API key or reduce request frequency
```bash
# Add to .env
NCBI_API_KEY=your_key_here
```

## Directory Structure Verification

Your project should look like this:

```
DeepOceanAnalysis/
├── README.md
├── SETUP.md (this file)
├── requirements.txt
├── config.yaml
├── .env
├── .gitignore
├── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   ├── faiss_index/
│   ├── blast_db/
│   ├── results/
│   └── clusters/
├── scripts/
│   ├── __init__.py
│   ├── fetch_ncbi.py
│   ├── embed_sequences.py
│   ├── build_faiss.py
│   ├── run_blast.py
│   ├── cluster_unknowns.py
│   ├── analyze_pipeline.py
│   └── report_generator.py
├── notebooks/
│   ├── 00_exploration.ipynb (to be created)
│   ├── 01_embedding.ipynb
│   ├── 02_similarity.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_reporting.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_faiss.py
│   ├── test_blast.py
│   ├── test_embeddings.py
│   └── test_pipeline.py
├── logs/
└── docs/
    ├── architecture.md
    ├── usage_examples.md
    └── pipeline_flow.md
```

## Next Steps

1. **Read the documentation**:
   - `README.md` - Project overview
   - `docs/architecture.md` - System design
   - `docs/usage_examples.md` - Example workflows

2. **Try example workflows**:
   - See `docs/usage_examples.md`
   - Run Jupyter notebooks

3. **Run your analysis**:
   - Prepare your FASTA files
   - Configure thresholds
   - Execute pipeline

4. **Join the community**:
   - Report issues on GitHub
   - Share your findings
   - Contribute improvements

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Open a GitHub issue
- **Email**: [Your contact email]

---

Happy analyzing! 🌊🧬
