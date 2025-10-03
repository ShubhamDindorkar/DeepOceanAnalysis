# 🌊 AquaGenomeAI - Deep-Sea Genomic Analysis Platform

**AquaGenomeAI** is an AI-powered platform for analyzing deep-sea genomic sequences, integrating state-of-the-art DNA language models ([DNABERT-2](https://arxiv.org/abs/2306.15006)), vector similarity search (FAISS), and graph-based knowledge retrieval (ArangoDB) to identify known organisms and discover novel species in deep-ocean environments.

The platform combines **automated sequence analysis**, **similarity-based taxonomic identification**, **clustering of unknown sequences**, and **interactive exploration** through an AI agent interface powered by LangChain and Google Gemini.

AquaGenomeAI supports both **end-to-end genomic pipeline automation** and **interactive, researcher-guided analysis workflows**.

## 🚀 Quick Start

### Prerequisites

- **Linux or MacOS** (Windows with WSL2)
- **Docker** (for ArangoDB)
- **Conda** or Python 3.9+
- **BLAST+** (optional, for verification)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/AquaGenomeAI.git
cd AquaGenomeAI
```

2. **Setup ArangoDB with Docker**
```bash
docker compose up -d
```
This creates an ArangoDB instance at `localhost:8529` with password `openSesame`.

3. **Create Python environment**
```bash
# Using conda (recommended)
conda create --name aquagenome python=3.9
conda activate aquagenome

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt

# Optional: Install BLAST+ for verification
conda install -c bioconda blast
```

5. **Configure API keys**

Create a `.env` file in the project root:
```bash
# Google Gemini API (for AI agent)
GOOGLE_API_KEY=your_gemini_api_key_here

# ArangoDB connection
ARANGO_HOST=http://localhost:8529
ARANGO_USER=root
ARANGO_PASS=openSesame

# Optional: NCBI API key (for faster downloads)
NCBI_API_KEY=your_ncbi_api_key_here

# Optional: Exa API (for literature search)
EXA_API_KEY=your_exa_api_key_here
```

6. **Initialize the database**

Run the setup notebook to populate the database with reference sequences:
```bash
jupyter notebook setup_database.ipynb
```

7. **Create vector index in ArangoDB** (for DNABERT embeddings)
```bash
docker exec -it <container_name> arangosh
```

In ArangoShell:
```javascript
db._useDatabase("AquaGenome");
db.sequence.ensureIndex({
    name: "dnabert_cosine",
    type: "vector",
    fields: ["embedding"],
    params: { metric: "cosine", dimension: 768, nLists: 100 }
});
```

8. **Launch the platform**
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 🏗️ Architecture

AquaGenomeAI is built as a modular, multi-model system for deep-sea genomic analysis, integrating AI, bioinformatics, and graph databases.

### 1. **Core Analysis Modules**

   - **DNABERT-2**: DNA language model (117M parameters) trained on genomic sequences to generate contextual embeddings (768-dim vectors) for similarity search and classification.
   - **FAISS**: Facebook AI Similarity Search for ultra-fast nearest-neighbor lookup in embedding space, enabling real-time identification of similar sequences.
   - **BLAST**: Traditional sequence alignment for verification of high-confidence matches and detailed alignment analysis.
   - **UMAP + HDBSCAN**: Dimensionality reduction and density-based clustering for grouping unknown sequences into candidate novel species.

### 2. **Knowledge Graph & Database**

   - **ArangoDB**: Multi-modal graph database storing:
     - **Sequences**: DNA/RNA sequences with metadata (location, depth, date)
     - **Taxa**: Taxonomic classifications (domain → species)
     - **Samples**: Deep-sea sample collection information
     - **Relationships**: Sequence-taxa links, evolutionary relationships, sample provenance
   - **GraphRAG**: Retrieval-augmented generation using graph traversal to answer complex biological queries
   - **Vector Indexing**: Native vector search in ArangoDB for embedding-based similarity

### 3. **Similarity & Classification**

   - **Embedding-based similarity**: DNABERT embeddings + cosine similarity for semantic sequence matching
   - **Threshold-based classification**: 
     - High similarity (>0.85) → Known species
     - Medium similarity (0.70-0.85) → Potential match (requires verification)
     - Low similarity (<0.70) → Unknown/novel candidate
   - **Multi-level verification**: FAISS → BLAST → manual review

### 4. **AI Agent & Reasoning**

   - **LangChain Agent**: Tool-calling agent with 11+ specialized genomic tools
   - **Google Gemini**: Large language model for natural language understanding and query interpretation
   - **GraphRAG framework**: Combines graph queries with LLM reasoning for complex biological questions
   - **Memory system**: SQLite-based conversation history for context-aware interactions

## 🔬 Features

### Current Capabilities

- ✅ **Sequence Upload & Analysis**: FASTA/FASTQ file support with metadata
- ✅ **DNABERT-2 Embeddings**: Generate 768-dim contextual sequence representations
- ✅ **FAISS Similarity Search**: Ultra-fast nearest-neighbor lookup in embedding space
- ✅ **BLAST Verification**: Traditional alignment for high-confidence matches
- ✅ **Unknown Clustering**: UMAP + HDBSCAN for grouping novel sequences
- ✅ **Graph Database**: ArangoDB for sequences, taxa, samples, and relationships
- ✅ **AI Chatbot**: Natural language interface for genomic queries
- ✅ **Interactive Visualizations**: Sequence alignments, phylogenetic trees, cluster plots
- ✅ **NCBI Integration**: Fetch reference sequences from GenBank/SILVA
- ✅ **Literature Search**: Exa API for recent taxonomy papers

### Genomic Analysis Tools

| Tool | Description | Status |
|------|-------------|--------|
| **FindSequence** | Query sequences from database | ✅ |
| **FindTaxaFromSequence** | Get taxonomic classification | ✅ |
| **TextToAQL** | Natural language → AQL graph queries | ✅ |
| **PlotSequenceAlignment** | Visualize alignments | ✅ |
| **PlotPhylogeneticTree** | Generate phylogenetic trees | ✅ |
| **PredictSimilarity** | FAISS-based similarity search | ✅ |
| **GetDNABERTEmbedding** | Generate sequence embeddings | ✅ |
| **PrepareSequenceData** | Process FASTA/FASTQ files | ✅ |
| **ClusterUnknowns** | UMAP + HDBSCAN clustering | ✅ |
| **FetchNCBISequences** | Download from NCBI | ✅ |
| **SearchLiterature** | Find relevant papers (Exa) | ✅ |

## 🚧 Future Roadmap

- **Enhanced Phylogenetic Analysis**:
  - Automated phylogenetic tree construction (IQ-TREE, FastTree)
  - Bootstrap support calculation
  - Interactive tree visualization with metadata overlays

- **Multi-Omics Integration**:
  - Metagenomics support (16S/18S rRNA, whole-genome)
  - Proteomics data integration
  - Metabolomics correlation

- **Real-time Monitoring**:
  - Live sequencing data pipeline (MinION integration)
  - Streaming analysis for shipboard research
  - Automated alerts for novel sequences

- **Advanced AI Features**:
  - Protein structure prediction (AlphaFold2 integration)
  - Functional annotation with large language models
  - Automated hypothesis generation

- **Collaboration Tools**:
  - Multi-user workspaces
  - Shared annotations and comments
  - Export to publication-ready formats

## 🤝 Contribute

We welcome contributions from bioinformaticians, marine biologists, and AI researchers! 

**Areas for contribution:**
- New genomic analysis tools
- Integration with other databases (Pfam, KEGG, etc.)
- Improved visualization methods
- Performance optimizations
- Documentation and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue or contact the maintainers.

---

**AquaGenomeAI** - Exploring the genomic frontier of Earth's final frontier 🌊🧬
