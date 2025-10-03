# System Architecture

## Overview

The Deep-Sea NCBI Analysis pipeline is designed as a modular, scalable system for analyzing genomic sequences using modern AI and bioinformatics tools.

## Architecture Diagram

```
┌─────────────────┐
│   Input Data    │
│  (FASTA/FASTQ)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - QC (fastp)   │
│  - Deduplication│
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Embedding     │─────▶│   Cerebras   │
│  Generation     │      │  (Optional)  │
│  (DNABERT-2)    │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │
│   Building &    │
│    Querying     │
└────────┬────────┘
         │
         ├────────────────┬─────────────────┐
         ▼                ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Similarity │  │    BLAST    │  │  Clustering │
│   Search    │  │Verification │  │  (Unknowns) │
│   (Known)   │  │ (Optional)  │  │   HDBSCAN   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
                        ▼
                ┌───────────────┐      ┌──────────┐
                │   Report      │─────▶│   Exa    │
                │  Generation   │      │Literature│
                └───────────────┘      └──────────┘
```

## Components

### 1. Data Input Layer

**Files**: `scripts/fetch_ncbi.py`

- Fetches sequences from NCBI databases
- Downloads BLAST databases
- Handles user-uploaded FASTA/FASTQ files
- Validates and normalizes input data

### 2. Preprocessing Layer

**Files**: `scripts/` (preprocessing functions in various modules)

- Quality control using `fastp` or `cutadapt`
- Sequence deduplication
- Length filtering
- Adapter trimming

### 3. Embedding Layer

**Files**: `scripts/embed_sequences.py`

**Key Technologies**:
- **DNABERT-2**: Primary embedding model
- **Cerebras**: Optional accelerated inference
- **HuggingFace Transformers**: Model loading and inference

**Process**:
1. Load pre-trained DNA language model
2. Tokenize sequences (k-mer or character-level)
3. Generate embeddings via forward pass
4. Pool representations (CLS token or mean pooling)
5. Normalize for similarity search

### 4. Indexing & Search Layer

**Files**: `scripts/build_faiss.py`

**Key Technologies**:
- **FAISS**: Facebook AI Similarity Search
- Support for multiple index types (Flat, IVF, PQ)
- GPU acceleration support

**Index Types**:
- `IndexFlatIP`: Exact inner-product search (development)
- `IndexFlatL2`: Exact L2 distance search
- `IVF,PQ`: Approximate search with quantization (production)

### 5. Similarity Branch (Known Sequences)

**Files**: `scripts/build_faiss.py`, `scripts/run_blast.py`

**Process**:
1. Query FAISS index with sequence embeddings
2. Retrieve top-k nearest neighbors
3. Apply similarity threshold
4. Optional: Verify with BLAST alignment
5. Report matches with confidence scores

**Outputs**:
- Per-sequence similarity scores
- Top-k matches with metadata
- Confidence classifications (high/medium/low)

### 6. Clustering Branch (Unknown Sequences)

**Files**: `scripts/cluster_unknowns.py`

**Process**:
1. Filter sequences below similarity threshold
2. Dimensionality reduction with UMAP
3. Density-based clustering with HDBSCAN
4. Compute cluster summaries
5. Visualize with matplotlib/seaborn

**Outputs**:
- Cluster assignments
- Cluster centroids and representatives
- Intra-cluster statistics
- UMAP visualizations

### 7. Verification Layer

**Files**: `scripts/run_blast.py`

**Process**:
- Run traditional BLAST alignment
- Parse XML/TSV results
- Compare with FAISS results
- Identify discrepancies

### 8. Reporting Layer

**Files**: `scripts/report_generator.py`

**Key Technologies**:
- **Exa**: Literature search and enrichment
- HTML/PDF report generation
- Interactive visualizations

**Process**:
1. Aggregate results from all branches
2. Query Exa for relevant literature
3. Generate summary statistics
4. Create visualizations
5. Export HTML/PDF report

## Data Flow

### Known Sequence Path

```
FASTA → Embed → FAISS Search → [Threshold Check] → High Similarity
                                                   → Report as Known
                                                   → Optional BLAST Verify
```

### Unknown Sequence Path

```
FASTA → Embed → FAISS Search → [Threshold Check] → Low Similarity
                                                   → UMAP Reduce
                                                   → HDBSCAN Cluster
                                                   → Generate Summaries
                                                   → Exa Literature
```

## Scalability Considerations

### Small-Scale (<10K sequences)

- Use `IndexFlatIP` for exact search
- CPU-only inference
- Single-node processing

### Medium-Scale (10K-1M sequences)

- Use `IVF,Flat` or `IVF,PQ` indices
- GPU acceleration for embedding generation
- Consider batch processing

### Large-Scale (>1M sequences)

- Use `IVF,PQ` with large `nlist`
- Cerebras Cloud for embedding generation
- Distributed FAISS (multi-GPU or multi-node)
- Incremental indexing and search

## Configuration Management

All pipeline behavior is controlled through `config.yaml`:

- Model selection and parameters
- FAISS index type and settings
- Similarity thresholds
- Clustering parameters
- API keys (via environment variables)

## Error Handling & Logging

- **Loguru**: Structured logging throughout
- Logs saved to `logs/pipeline.log`
- Error recovery for API failures
- Graceful degradation (e.g., skip BLAST if unavailable)

## Extensibility

The modular design allows easy extension:

1. **New embedding models**: Modify `embed_sequences.py`
2. **Different clustering algorithms**: Extend `cluster_unknowns.py`
3. **Additional verification**: Add new modules to pipeline
4. **Custom reports**: Extend `report_generator.py`

## Performance Optimization

### Embedding Generation

- Batch processing
- GPU/Cerebras acceleration
- Caching embeddings to disk

### FAISS Search

- Index type selection (speed vs. accuracy tradeoff)
- GPU indices for large-scale search
- `nprobe` tuning for IVF indices

### Memory Management

- Streaming large FASTA files
- Chunked processing for massive datasets
- Lazy loading of embeddings

## Security

- API keys stored in `.env` (never committed)
- Input validation for FASTA files
- Sandboxed BLAST execution
- Rate limiting for external API calls (NCBI, Exa)

## Testing Strategy

- **Unit tests**: Individual component testing
- **Integration tests**: Multi-component workflows
- **End-to-end tests**: Full pipeline with sample data
- **Performance tests**: Scalability benchmarks

## Deployment

### Development

```bash
python scripts/analyze_pipeline.py --input data.fasta --output results/
```

### Production

- Containerization (Docker)
- Cloud deployment (AWS, GCP, Azure)
- Cerebras Cloud integration for scale
- Monitoring and alerting
