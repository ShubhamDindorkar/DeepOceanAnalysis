# Project Structure - Deep Ocean Analysis

## 📁 Complete Directory Tree

```
DeepOceanAnalysis/
│
├── 📄 README.md                    # Comprehensive project documentation
├── 📄 SETUP.md                     # Installation and setup guide
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.yaml                  # Pipeline configuration
├── 📄 .env                         # API keys (DO NOT COMMIT)
├── 📄 .gitignore                   # Git ignore rules
├── 📄 main.py                      # Main entry point
│
├── 📂 data/                        # All data files (gitignored)
│   ├── 📂 raw/                     # Raw input sequences
│   ├── 📂 processed/               # QC'd and cleaned sequences
│   ├── 📂 embeddings/              # Generated embeddings (.npy)
│   ├── 📂 faiss_index/             # FAISS index files
│   ├── 📂 blast_db/                # Downloaded BLAST databases
│   ├── 📂 results/                 # Analysis results
│   └── 📂 clusters/                # Cluster assignments and summaries
│
├── 📂 scripts/                     # Core pipeline scripts
│   ├── __init__.py
│   ├── fetch_ncbi.py               # Fetch sequences from NCBI
│   ├── embed_sequences.py          # Generate embeddings (Cerebras/FAISS)
│   ├── build_faiss.py              # Build and query FAISS index
│   ├── run_blast.py                # BLAST verification
│   ├── cluster_unknowns.py         # UMAP + HDBSCAN clustering
│   ├── analyze_pipeline.py         # Full end-to-end pipeline
│   └── report_generator.py         # Report generation with Exa
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── README.md
│   ├── 00_exploration.ipynb        # Data exploration
│   ├── 01_embedding.ipynb          # Embedding generation
│   ├── 02_similarity.ipynb         # Similarity search
│   ├── 03_clustering.ipynb         # Clustering experiments
│   └── 04_reporting.ipynb          # Report generation
│
├── 📂 tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_faiss.py               # FAISS index tests
│   ├── test_blast.py               # BLAST wrapper tests
│   ├── test_embeddings.py          # Embedding generation tests
│   └── test_pipeline.py            # Full pipeline tests
│
├── 📂 logs/                        # Pipeline execution logs
│   └── README.md
│
└── 📂 docs/                        # Documentation
    ├── architecture.md             # System architecture
    ├── pipeline_flow.md            # Pipeline flow diagram
    └── usage_examples.md           # Usage examples and workflows
```

## 🎯 Key Components

### Main Entry Point

**`main.py`**
- CLI interface for running the complete pipeline
- Usage: `python main.py --input data.fasta --output results/`

### Core Scripts

| Script | Purpose | Key Technologies |
|--------|---------|------------------|
| `fetch_ncbi.py` | Download sequences from NCBI | BioPython, Entrez |
| `embed_sequences.py` | Generate DNA embeddings | DNABERT-2, Cerebras, PyTorch |
| `build_faiss.py` | Index & search embeddings | FAISS |
| `run_blast.py` | Verify matches with alignment | BLAST+ |
| `cluster_unknowns.py` | Cluster unknown sequences | UMAP, HDBSCAN |
| `analyze_pipeline.py` | Orchestrate full pipeline | All above |
| `report_generator.py` | Generate reports | Exa, HTML |

### Configuration Files

**`config.yaml`**
- Model selection (DNABERT-2, Nucleotide Transformer, etc.)
- FAISS index type and parameters
- Similarity thresholds
- Clustering parameters (UMAP, HDBSCAN)
- BLAST settings
- Exa integration settings

**`.env`**
- `CEREBRAS_API_KEY` - For Cerebras Cloud inference
- `EXA_API_KEY` - For literature lookup
- `NCBI_API_KEY` - For faster NCBI downloads (optional)
- `HF_TOKEN` - For HuggingFace gated models (optional)

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Configure NCBI email (required)
# Edit config.yaml: ncbi.email = "your@email.com"
```

### 2. Run Pipeline

```bash
# Full analysis
python main.py --input data/raw/sequences.fasta --output data/results/

# With verbose logging
python main.py -i data/raw/sequences.fasta -o results/ --verbose
```

### 3. View Results

```bash
# Open HTML report
open data/results/analysis_report.html  # Mac
start data/results/analysis_report.html  # Windows

# Or explore in Jupyter
jupyter notebook notebooks/04_reporting.ipynb
```

## 📊 Data Flow

```
Input FASTA
    ↓
Preprocessing (QC, dedup)
    ↓
Embedding Generation (DNABERT-2 / Cerebras)
    ↓
FAISS Indexing & Search
    ↓
    ├── High Similarity → Known Branch → BLAST Verify → Report
    │
    └── Low Similarity → Unknown Branch → UMAP + HDBSCAN → Clusters
                                                              ↓
                                                         Exa Literature
                                                              ↓
                                                         Final Report
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_faiss.py -v

# Run with coverage
pytest --cov=scripts tests/
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `README.md` | Complete project overview with goals and architecture |
| `SETUP.md` | Detailed installation and configuration guide |
| `docs/architecture.md` | System design and component details |
| `docs/pipeline_flow.md` | Visual pipeline flow diagrams |
| `docs/usage_examples.md` | Example workflows and commands |

## 🔧 Customization

### Change Embedding Model

Edit `config.yaml`:
```yaml
model:
  name: "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
```

### Adjust Similarity Threshold

Edit `config.yaml`:
```yaml
similarity:
  known_threshold: 0.85  # Stricter threshold
```

### Enable Cerebras

Edit `config.yaml`:
```yaml
model:
  use_cerebras: true
```

And set API key in `.env`:
```
CEREBRAS_API_KEY=your_key_here
```

## 📝 Important Notes

### Files to Never Commit

- `.env` (contains API keys)
- `data/` directories (large datasets)
- `logs/*.log` (log files)
- `__pycache__/` (Python cache)

These are already in `.gitignore`.

### Required Before Running

1. ✅ Install dependencies (`pip install -r requirements.txt`)
2. ✅ Configure `.env` with API keys
3. ✅ Set `ncbi.email` in `config.yaml`
4. ✅ Prepare input FASTA file

### Optional Enhancements

- Install BLAST+ for verification: `conda install -c bioconda blast`
- Use GPU: Install `faiss-gpu` and set `device: "cuda"` in config
- Enable Exa: Set `EXA_API_KEY` and `exa.enabled: true`

## 🎓 Learning Path

1. **Start here**: `README.md` - Understand the project goals
2. **Setup**: `SETUP.md` - Get everything installed
3. **Architecture**: `docs/architecture.md` - Learn system design
4. **Try it**: `docs/usage_examples.md` - Run example workflows
5. **Explore**: `notebooks/` - Interactive analysis
6. **Customize**: Edit `config.yaml` for your needs

## 🆘 Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `docs/usage_examples.md`
- **Logs**: Review `logs/pipeline.log` for errors
- **Tests**: Run `pytest tests/ -v` to verify setup
- **Issues**: Open a GitHub issue (if applicable)

## 📦 Deliverables

After running the pipeline, you'll have:

1. **Similarity matches** (`data/results/similarity_matches.csv`)
2. **Cluster assignments** (`data/results/clusters/cluster_labels.csv`)
3. **Cluster summaries** (`data/results/clusters/cluster_summary.csv`)
4. **Visualizations** (`data/results/clusters/clusters_umap.png`)
5. **HTML Report** (`data/results/analysis_report.html`)

---

## 🌊 Ready to analyze deep-sea sequences!

Run `python main.py --help` to get started.
