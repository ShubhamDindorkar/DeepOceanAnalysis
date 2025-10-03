# Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                │
│                                                                   │
│  ┌────────────────┐        ┌─────────────────┐                  │
│  │ User FASTA/    │        │  NCBI Entrez    │                  │
│  │ FASTQ Files    │        │  Downloads      │                  │
│  └────────┬───────┘        └────────┬────────┘                  │
│           │                         │                            │
│           └────────────┬────────────┘                            │
│                        │                                         │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                 │
│                                                                   │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐         │
│  │  Quality   │→ │Deduplication│→ │ Length Filtering │         │
│  │  Control   │  │             │  │                  │         │
│  └────────────┘  └─────────────┘  └──────────────────┘         │
│  (fastp/cutadapt)                                                │
└────────────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                   EMBEDDING GENERATION                            │
│                                                                   │
│  ┌────────────────────────────────────────────┐                 │
│  │         DNA Language Model                 │                 │
│  │         (DNABERT-2 / Nucleotide            │                 │
│  │          Transformer)                      │                 │
│  │                                            │                 │
│  │  ┌────────────┐          ┌──────────────┐ │                 │
│  │  │  Local GPU │   OR     │   Cerebras   │ │                 │
│  │  │  Inference │          │    Cloud     │ │                 │
│  │  └────────────┘          └──────────────┘ │                 │
│  └────────────────────────────────────────────┘                 │
│                                                                   │
│  Output: Dense Vector Embeddings (n_sequences × embedding_dim)   │
└────────────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FAISS INDEXING                                │
│                                                                   │
│  ┌──────────────────────────────────────────────┐               │
│  │  Reference Database Embeddings               │               │
│  │  (GenBank, SILVA, Custom)                    │               │
│  └──────────────────┬───────────────────────────┘               │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────┐               │
│  │          Build FAISS Index                   │               │
│  │  • IndexFlatIP (exact, small scale)          │               │
│  │  • IVF,PQ (approximate, large scale)         │               │
│  └──────────────────────────────────────────────┘               │
└────────────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                   SIMILARITY SEARCH                               │
│                                                                   │
│  Query Embeddings → FAISS k-NN Search → Top-k Matches            │
│                                                                   │
│  For each query:                                                  │
│    similarity_score = cosine_similarity(query, reference)         │
│                                                                   │
│  ┌────────────────────────────────────────────┐                 │
│  │        Threshold Decision                  │                 │
│  │                                            │                 │
│  │  similarity ≥ threshold ────┐             │                 │
│  │                             │             │                 │
│  │  similarity < threshold ────┤             │                 │
│  └─────────────────────────────┼─────────────┘                 │
└─────────────────────────────────┼─────────────────────────────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
    ┌─────────────────────────┐   ┌─────────────────────────┐
    │    KNOWN BRANCH         │   │   UNKNOWN BRANCH        │
    │                         │   │                         │
    │  ┌──────────────────┐   │   │  ┌──────────────────┐  │
    │  │ High Similarity  │   │   │  │ Low Similarity   │  │
    │  │   Matches        │   │   │  │  Sequences       │  │
    │  └────────┬─────────┘   │   │  └────────┬─────────┘  │
    │           │              │   │           │             │
    │           ▼              │   │           ▼             │
    │  ┌──────────────────┐   │   │  ┌──────────────────┐  │
    │  │ BLAST            │   │   │  │ UMAP             │  │
    │  │ Verification     │   │   │  │ Dimensionality   │  │
    │  │ (Optional)       │   │   │  │ Reduction        │  │
    │  └────────┬─────────┘   │   │  └────────┬─────────┘  │
    │           │              │   │           │             │
    │           ▼              │   │           ▼             │
    │  ┌──────────────────┐   │   │  ┌──────────────────┐  │
    │  │ Report Match     │   │   │  │ HDBSCAN          │  │
    │  │ • Taxon ID       │   │   │  │ Clustering       │  │
    │  │ • Confidence     │   │   │  └────────┬─────────┘  │
    │  │ • E-value        │   │   │           │             │
    │  └──────────────────┘   │   │           ▼             │
    │                         │   │  ┌──────────────────┐  │
    └─────────────┬───────────┘   │  │ Cluster          │  │
                  │               │  │ Summaries        │  │
                  │               │  │ • Size           │  │
                  │               │  │ • Representatives│  │
                  │               │  │ • Cohesion       │  │
                  │               │  └────────┬─────────┘  │
                  │               │           │             │
                  │               └───────────┼─────────────┘
                  │                           │
                  └──────────┬────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────────────┐
           │        LITERATURE ENRICHMENT             │
           │              (Exa API)                   │
           │                                          │
           │  For each cluster/match:                 │
           │    • Query recent papers                 │
           │    • Fetch taxonomy updates              │
           │    • Get species descriptions            │
           └──────────────┬───────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────────────┐
           │          REPORT GENERATION               │
           │                                          │
           │  ┌─────────────────────────────────┐    │
           │  │  HTML Report                    │    │
           │  │  • Summary statistics           │    │
           │  │  • Similarity matches           │    │
           │  │  • Cluster visualizations       │    │
           │  │  • Literature references        │    │
           │  └─────────────────────────────────┘    │
           │                                          │
           │  ┌─────────────────────────────────┐    │
           │  │  Exportable Data                │    │
           │  │  • CSV (matches, clusters)      │    │
           │  │  • FASTA (representatives)      │    │
           │  │  • Plots (PNG/PDF)              │    │
           │  └─────────────────────────────────┘    │
           └──────────────────────────────────────────┘
```

## Key Decision Points

### Threshold Calibration
- **Input**: Held-out labeled validation set
- **Process**: ROC curve analysis
- **Output**: Optimal similarity threshold (e.g., 0.82)

### Index Selection
- **Small (<10K)**: `IndexFlatIP` (exact search)
- **Medium (10K-1M)**: `IVF,Flat` (fast approximate)
- **Large (>1M)**: `IVF,PQ` (memory-efficient)

### Clustering Parameters
- **UMAP**: n_neighbors=15, min_dist=0.1
- **HDBSCAN**: min_cluster_size=10, min_samples=5
- Tune based on dataset characteristics

## Data Flow Summary

1. **Raw Sequences** → Quality Control → Clean Sequences
2. **Clean Sequences** → Embedding Model → Dense Vectors
3. **Dense Vectors** → FAISS Index → Similarity Scores
4. **High Similarity** → Known Branch → Verified Matches
5. **Low Similarity** → Unknown Branch → Clusters
6. **All Results** → Exa Enrichment → Final Report
