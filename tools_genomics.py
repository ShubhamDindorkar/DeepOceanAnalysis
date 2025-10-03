"""
AquaGenomeAI - Genomic Analysis Tools
=====================================

Tools for deep-sea genomic sequence analysis using DNABERT-2, FAISS, and ArangoDB.
"""

import os
import sys
import hashlib
import tempfile
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from io import StringIO

from db import db, arango_graph, sequence_collection, taxon_collection, sample_collection

import pandas as pd
import numpy as np

from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModel
import torch

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool

from pydantic import BaseModel, Field

from Bio import SeqIO, Entrez, Align
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

import faiss
import umap
import hdbscan

import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Configuration =================

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
ncbi_api_key = os.getenv("NCBI_API_KEY")
exa_api_key = os.getenv("EXA_API_KEY")

# Set NCBI email (required)
Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")
if ncbi_api_key:
    Entrez.api_key = ncbi_api_key

# ================= Models =================

# Load DNABERT-2 for sequence embeddings
print("Loading DNABERT-2 model...")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model.eval()
print("✅ DNABERT-2 loaded successfully")

# ================= Helper Functions =================

def _GenerateKey(sequence: str) -> str:
    """Generate a unique _key for the sequence using hash."""
    hash_value = hashlib.sha256(sequence.encode()).hexdigest()[:12]
    return f"SEQ:{hash_value}"

def _SanitizeInput(d: Any, list_limit: int = 100) -> Any:
    """Sanitize input by removing large lists and embeddings."""
    if isinstance(d, dict):
        return {
            k: _SanitizeInput(v, list_limit)
            for k, v in d.items()
            if k not in ['embedding', 'sequence'] or len(str(v)) < 200
        }
    elif isinstance(d, list):
        if len(d) > list_limit:
            return f"<list with {len(d)} items>"
        return [_SanitizeInput(item, list_limit) for item in d]
    return d

def _display_sidebar_output(title: str, content: Any, content_type: str = "json") -> None:
    """Display output in Streamlit sidebar."""
    try:
        with st.sidebar:
            with st.expander(f"📊 {title}", expanded=False):
                if content_type == "json":
                    st.json(_SanitizeInput(content))
                elif content_type == "table":
                    st.dataframe(content)
                elif content_type == "text":
                    st.text(content)
                elif content_type == "error":
                    st.error(content)
    except:
        pass  # Fail silently if not in Streamlit context

# ================= Core Genomic Tools =================

def GetDNABERTEmbedding(sequence: str) -> List[float]:
    """
    Generate DNABERT-2 embedding for a DNA sequence.
    
    Args:
        sequence (str): DNA sequence (e.g., "ATCGATCGATCG")
    
    Returns:
        List[float]: 768-dimensional embedding vector
    
    Example:
        embedding = GetDNABERTEmbedding("ATCGATCGATCG")
    """
    try:
        # Validate sequence
        sequence = sequence.upper().strip()
        if not all(base in 'ATCGN' for base in sequence):
            return []
        
        # Tokenize
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output[0]
        else:
            embedding = outputs.last_hidden_state.mean(dim=1)[0]
        
        embedding_list = embedding.cpu().numpy().tolist()
        
        _display_sidebar_output("DNABERT Embedding", {
            "sequence_length": len(sequence),
            "embedding_dim": len(embedding_list),
            "sample_values": embedding_list[:5]
        })
        
        return embedding_list
        
    except Exception as e:
        _display_sidebar_output("Embedding Error", str(e), "error")
        return []


def FindSequence(sequence_id: str) -> Dict[str, Any]:
    """
    Find a sequence in the database by ID or partial match.
    
    Args:
        sequence_id (str): Sequence identifier (e.g., "SEQ_123", "GenBank_AB123456")
    
    Returns:
        Dict: Sequence document with metadata
    
    Example:
        seq = FindSequence("SEQ_123")
    """
    try:
        # Try exact key match
        if sequence_collection.has(sequence_id):
            result = sequence_collection.get(sequence_id)
        else:
            # Search by sequence_id field
            aql = """
            FOR seq IN sequence
                FILTER seq.sequence_id == @seq_id OR CONTAINS(seq.sequence_id, @seq_id)
                LIMIT 1
                RETURN seq
            """
            cursor = db.aql.execute(aql, bind_vars={"seq_id": sequence_id})
            results = list(cursor)
            result = results[0] if results else None
        
        if result:
            _display_sidebar_output("Found Sequence", {
                "_key": result.get("_key"),
                "sequence_id": result.get("sequence_id"),
                "length": result.get("length"),
                "type": result.get("sequence_type"),
                "sample": result.get("sample_id"),
                "gc_content": result.get("gc_content")
            })
            return result
        else:
            _display_sidebar_output("Not Found", f"No sequence found for: {sequence_id}", "error")
            return {}
            
    except Exception as e:
        _display_sidebar_output("Search Error", str(e), "error")
        return {}


def FindTaxaFromSequence(sequence_id: str) -> List[Dict[str, Any]]:
    """
    Get taxonomic classifications linked to a sequence.
    
    Args:
        sequence_id (str): Sequence identifier
    
    Returns:
        List[Dict]: List of taxa with similarity scores
    
    Example:
        taxa = FindTaxaFromSequence("SEQ_123")
    """
    try:
        aql = """
        FOR seq IN sequence
            FILTER seq._key == @seq_key OR seq.sequence_id == @seq_id
            FOR v, e IN 1..1 OUTBOUND seq._id `sequence-taxon`
                SORT e.similarity_score DESC
                RETURN {
                    taxon_key: v._key,
                    scientific_name: v.scientific_name,
                    common_name: v.common_name,
                    rank: v.rank,
                    similarity_score: e.similarity_score,
                    method: e.method,
                    verified: e.verified,
                    lineage: v.lineage
                }
        """
        
        cursor = db.aql.execute(aql, bind_vars={
            "seq_key": sequence_id,
            "seq_id": sequence_id
        })
        results = list(cursor)
        
        if results:
            df = pd.DataFrame(results)
            st.table(df[['scientific_name', 'rank', 'similarity_score', 'verified']])
            
            _display_sidebar_output("Taxa Found", {
                "num_taxa": len(results),
                "top_match": results[0].get("scientific_name") if results else None
            })
        else:
            _display_sidebar_output("No Taxa", f"No taxa found for {sequence_id}", "error")
        
        return results
        
    except Exception as e:
        _display_sidebar_output("Query Error", str(e), "error")
        return []


def FindSimilarSequences(sequence: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Find similar sequences using DNABERT embedding similarity.
    
    Args:
        sequence (str): Query DNA sequence
        top_k (int): Number of similar sequences to return
    
    Returns:
        List[Dict]: Similar sequences with scores
    
    Example:
        similar = FindSimilarSequences("ATCGATCG", top_k=5)
    """
    try:
        # Validate sequence
        sequence = sequence.upper().strip()
        if not all(base in 'ATCGN' for base in sequence):
            _display_sidebar_output("Invalid Sequence", "Sequence contains invalid bases", "error")
            return []
        
        # Generate embedding
        embedding = GetDNABERTEmbedding(sequence)
        if not embedding:
            return []
        
        # Vector similarity search in ArangoDB
        aql = """
        LET query_vector = @query_vector
        FOR seq IN sequence
            FILTER seq.embedding != null
            LET score = COSINE_SIMILARITY(seq.embedding, query_vector)
            FILTER score > 0.5
            SORT score DESC
            LIMIT @top_k
            RETURN {
                sequence_key: seq._key,
                sequence_id: seq.sequence_id,
                similarity_score: score,
                sequence_type: seq.sequence_type,
                length: seq.length,
                sample_id: seq.sample_id,
                source: seq.source
            }
        """
        
        cursor = db.aql.execute(aql, bind_vars={
            "query_vector": embedding,
            "top_k": top_k
        })
        results = list(cursor)
        
        if results:
            df = pd.DataFrame(results)
            df['similarity_score'] = df['similarity_score'].round(4)
            st.table(df)
            
            _display_sidebar_output("Similarity Search", {
                "query_length": len(sequence),
                "results_found": len(results),
                "top_score": results[0].get('similarity_score') if results else 0
            })
        else:
            _display_sidebar_output("No Matches", "No similar sequences found", "error")
        
        return results
        
    except Exception as e:
        _display_sidebar_output("Search Error", str(e), "error")
        return []


def TextToAQL(question: str) -> str:
    """
    Convert natural language question to AQL query using LLM.
    
    Args:
        question (str): Natural language question about the database
    
    Returns:
        str: AQL query results as formatted string
    
    Example:
        result = TextToAQL("How many sequences are from deep-sea vents?")
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=google_api_key
        )
        
        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=arango_graph,
            verbose=True,
            return_direct=True,
            allow_dangerous_requests=True
        )
        
        result = chain.invoke({"query": question})
        
        _display_sidebar_output("AQL Query Result", result)
        
        return str(result)
        
    except Exception as e:
        error_msg = f"Query failed: {str(e)}"
        _display_sidebar_output("Query Error", error_msg, "error")
        return error_msg


def FetchNCBISequences(query: str, database: str = "nucleotide", max_results: int = 100) -> str:
    """
    Download sequences from NCBI databases.
    
    Args:
        query (str): Search query (e.g., "deep sea bacteria 16S")
        database (str): NCBI database (nucleotide, protein, etc.)
        max_results (int): Maximum sequences to download
    
    Returns:
        str: Path to downloaded FASTA file or status message
    
    Example:
        file = FetchNCBISequences("deep sea archaea 16S", max_results=50)
    """
    try:
        # Search NCBI
        handle = Entrez.esearch(
            db=database,
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        
        if not id_list:
            return f"No sequences found for query: {query}"
        
        # Fetch sequences
        handle = Entrez.efetch(
            db=database,
            id=id_list,
            rettype="fasta",
            retmode="text"
        )
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.fasta',
            delete=False,
            dir='data/raw' if os.path.exists('data/raw') else None
        )
        temp_file.write(handle.read())
        temp_file.close()
        handle.close()
        
        _display_sidebar_output("NCBI Download", {
            "query": query,
            "sequences_found": len(id_list),
            "file_path": temp_file.name
        })
        
        return f"Downloaded {len(id_list)} sequences to {temp_file.name}"
        
    except Exception as e:
        error_msg = f"NCBI fetch failed: {str(e)}"
        _display_sidebar_output("NCBI Error", error_msg, "error")
        return error_msg


def PrepareSequenceData(fasta_file: str) -> Dict[str, Any]:
    """
    Process FASTA file and prepare for database insertion.
    
    Args:
        fasta_file (str): Path to FASTA file
    
    Returns:
        Dict: Processing summary with statistics
    
    Example:
        summary = PrepareSequenceData("data/raw/samples.fasta")
    """
    try:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        
        stats = {
            "total_sequences": len(sequences),
            "avg_length": np.mean([len(s.seq) for s in sequences]),
            "min_length": min([len(s.seq) for s in sequences]),
            "max_length": max([len(s.seq) for s in sequences]),
            "avg_gc_content": np.mean([gc_fraction(s.seq) for s in sequences]) * 100
        }
        
        _display_sidebar_output("FASTA Processing", stats)
        
        return {
            "status": "success",
            "file": fasta_file,
            "sequences": len(sequences),
            "statistics": stats
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        _display_sidebar_output("Processing Error", error_msg, "error")
        return {"status": "error", "message": error_msg}


def ClusterUnknowns(min_similarity: float = 0.7, min_cluster_size: int = 5) -> Dict[str, Any]:
    """
    Cluster unknown/unclassified sequences using UMAP + HDBSCAN.
    
    Args:
        min_similarity (float): Sequences below this similarity are "unknown"
        min_cluster_size (int): Minimum cluster size for HDBSCAN
    
    Returns:
        Dict: Clustering results with statistics
    
    Example:
        clusters = ClusterUnknowns(min_similarity=0.7, min_cluster_size=10)
    """
    try:
        # Get unknown sequences (low similarity scores)
        aql = """
        FOR seq IN sequence
            FILTER seq.embedding != null
            LET max_sim = (
                FOR v, e IN 1..1 OUTBOUND seq._id `sequence-taxon`
                    RETURN e.similarity_score
            )
            FILTER LENGTH(max_sim) == 0 OR MAX(max_sim) < @threshold
            RETURN {
                key: seq._key,
                embedding: seq.embedding
            }
        """
        
        cursor = db.aql.execute(aql, bind_vars={"threshold": min_similarity})
        results = list(cursor)
        
        if len(results) < min_cluster_size:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(results)} unknown sequences found, need at least {min_cluster_size}"
            }
        
        # Extract embeddings
        embeddings = np.array([r['embedding'] for r in results])
        keys = [r['key'] for r in results]
        
        # UMAP reduction
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')
        embedding_2d = reducer.fit_transform(embeddings)
        
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3)
        labels = clusterer.fit_predict(embedding_2d)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6
        )
        ax.set_title(f'Unknown Sequence Clusters\n{n_clusters} clusters, {n_noise} noise points')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        
        cluster_summary = {
            "total_unknowns": len(results),
            "clusters_found": n_clusters,
            "noise_points": n_noise,
            "cluster_labels": labels.tolist()
        }
        
        _display_sidebar_output("Clustering Results", cluster_summary)
        
        return cluster_summary
        
    except Exception as e:
        error_msg = f"Clustering failed: {str(e)}"
        _display_sidebar_output("Clustering Error", error_msg, "error")
        return {"status": "error", "message": error_msg}


def PlotSequenceAlignment(sequences: List[str], labels: List[str] = None) -> str:
    """
    Visualize multiple sequence alignment.
    
    Args:
        sequences (List[str]): List of DNA sequences to align
        labels (List[str]): Optional labels for each sequence
    
    Returns:
        str: Status message
    
    Example:
        PlotSequenceAlignment(["ATCG", "ATGG", "ATCG"], ["Seq1", "Seq2", "Seq3"])
    """
    try:
        if len(sequences) > 50:
            return "Too many sequences to visualize. Maximum: 50"
        
        # Simple visualization of sequence lengths and GC content
        if labels is None:
            labels = [f"Seq{i+1}" for i in range(len(sequences))]
        
        data = {
            'Sequence': labels,
            'Length': [len(s) for s in sequences],
            'GC%': [round(gc_fraction(Seq(s)) * 100, 2) for s in sequences]
        }
        
        df = pd.DataFrame(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.barh(df['Sequence'], df['Length'])
        ax1.set_xlabel('Length (bp)')
        ax1.set_title('Sequence Lengths')
        
        ax2.barh(df['Sequence'], df['GC%'])
        ax2.set_xlabel('GC Content (%)')
        ax2.set_title('GC Content')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(df)
        
        return f"Plotted {len(sequences)} sequences"
        
    except Exception as e:
        return f"Plot failed: {str(e)}"


def SearchLiterature(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search scientific literature using Exa API.
    
    Args:
        query (str): Search query (e.g., "novel deep-sea archaea")
        num_results (int): Number of results to return
    
    Returns:
        List[Dict]: Literature results with titles, URLs, snippets
    
    Example:
        papers = SearchLiterature("deep sea microbiome genomics", 5)
    """
    try:
        if not exa_api_key:
            return [{"error": "EXA_API_KEY not set in .env file"}]
        
        from exa_py import Exa
        
        exa = Exa(api_key=exa_api_key)
        results = exa.search_and_contents(
            query=f"{query} genomics taxonomy",
            type="neural",
            num_results=num_results,
            text=True
        )
        
        papers = []
        for result in results.results:
            papers.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.text[:200] if hasattr(result, 'text') else ""
            })
        
        if papers:
            df = pd.DataFrame(papers)
            st.dataframe(df)
            
            _display_sidebar_output("Literature Search", {
                "query": query,
                "results_found": len(papers)
            })
        
        return papers
        
    except Exception as e:
        error_msg = f"Literature search failed: {str(e)}"
        _display_sidebar_output("Search Error", error_msg, "error")
        return [{"error": error_msg}]


# ================= Tool Wrappers for LangChain =================

find_sequence = Tool(
    name="FindSequence",
    func=FindSequence,
    description=FindSequence.__doc__
)

find_taxa_from_sequence = Tool(
    name="FindTaxaFromSequence",
    func=FindTaxaFromSequence,
    description=FindTaxaFromSequence.__doc__
)

text_to_aql = Tool(
    name="TextToAQL",
    func=TextToAQL,
    description=TextToAQL.__doc__
)

plot_sequence_alignment = Tool(
    name="PlotSequenceAlignment",
    func=PlotSequenceAlignment,
    description=PlotSequenceAlignment.__doc__
)

get_dnabert_embedding = Tool(
    name="GetDNABERTEmbedding",
    func=GetDNABERTEmbedding,
    description=GetDNABERTEmbedding.__doc__
)

prepare_sequence_data = Tool(
    name="PrepareSequenceData",
    func=PrepareSequenceData,
    description=PrepareSequenceData.__doc__
)

fetch_ncbi_sequences = Tool(
    name="FetchNCBISequences",
    func=FetchNCBISequences,
    description=FetchNCBISequences.__doc__
)

find_similar_sequences = Tool(
    name="FindSimilarSequences",
    func=FindSimilarSequences,
    description=FindSimilarSequences.__doc__
)

cluster_unknowns = Tool(
    name="ClusterUnknowns",
    func=ClusterUnknowns,
    description=ClusterUnknowns.__doc__
)

search_literature = Tool(
    name="SearchLiterature",
    func=SearchLiterature,
    description=SearchLiterature.__doc__
)

# ================= Tool Collection =================

tools = [
    find_sequence,
    find_taxa_from_sequence,
    text_to_aql,
    plot_sequence_alignment,
    get_dnabert_embedding,
    prepare_sequence_data,
    fetch_ncbi_sequences,
    find_similar_sequences,
    cluster_unknowns,
    search_literature
]

print(f"✅ Loaded {len(tools)} genomic analysis tools")
