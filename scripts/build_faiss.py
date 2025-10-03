#!/usr/bin/env python3
"""
Build and query FAISS index for sequence similarity search.
"""

import argparse
import yaml
import numpy as np
import faiss
from pathlib import Path
from loguru import logger
import pickle


class FAISSIndexBuilder:
    """Build and manage FAISS indices for sequence embeddings."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.faiss_config = self.config['faiss']
        self.index_type = self.faiss_config['index_type']
        self.use_gpu = self.faiss_config.get('use_gpu', False)
    
    def build_index(self, embeddings, index_type=None):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n_sequences, embedding_dim)
            index_type: FAISS index type (overrides config)
        
        Returns:
            FAISS index object
        """
        if index_type is None:
            index_type = self.index_type
        
        n_sequences, embedding_dim = embeddings.shape
        logger.info(f"Building {index_type} index for {n_sequences} sequences")
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Normalize for inner product search (cosine similarity)
        if "IP" in index_type:
            logger.info("Normalizing embeddings for inner product search")
            faiss.normalize_L2(embeddings)
        
        # Create index
        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(embedding_dim)
        elif "IVF" in index_type:
            # Parse composite index type
            nlist = self.faiss_config.get('nlist', 1024)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            index.train(embeddings)
        else:
            # Use index_factory for complex indices
            logger.info(f"Using index_factory for: {index_type}")
            index = faiss.index_factory(embedding_dim, index_type)
            
            if not index.is_trained:
                logger.info("Training index...")
                index.train(embeddings)
        
        # Add embeddings to index
        logger.info("Adding embeddings to index...")
        index.add(embeddings)
        
        logger.info(f"Index built successfully. Total vectors: {index.ntotal}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                index
            )
        
        return index
    
    def save_index(self, index, output_path, metadata=None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index object
            output_path: Path to save index file
            metadata: Optional metadata dict (sequence IDs, etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU if on GPU
        if hasattr(index, 'index'):  # GpuIndex
            index = faiss.index_gpu_to_cpu(index)
        
        # Save index
        logger.info(f"Saving index to {output_path}")
        faiss.write_index(index, str(output_path))
        
        # Save metadata
        if metadata:
            metadata_path = output_path.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved metadata to {metadata_path}")
    
    def load_index(self, index_path):
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to index file
        
        Returns:
            tuple of (index, metadata)
        """
        index_path = Path(index_path)
        logger.info(f"Loading index from {index_path}")
        
        index = faiss.read_index(str(index_path))
        
        # Load metadata if exists
        metadata_path = index_path.with_suffix('.metadata.pkl')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded metadata with {len(metadata.get('ids', []))} entries")
        
        return index, metadata
    
    def search(self, index, query_embeddings, k=10, nprobe=None):
        """
        Search index for nearest neighbors.
        
        Args:
            index: FAISS index
            query_embeddings: Query vectors (n_queries, embedding_dim)
            k: Number of nearest neighbors to return
            nprobe: Number of clusters to search (for IVF indices)
        
        Returns:
            tuple of (distances, indices)
        """
        # Normalize queries if using IP
        if isinstance(index, (faiss.IndexFlatIP, faiss.IndexIVFFlat)):
            faiss.normalize_L2(query_embeddings)
        
        # Set nprobe for IVF indices
        if nprobe and hasattr(index, 'nprobe'):
            index.nprobe = nprobe
        
        logger.info(f"Searching for {k} nearest neighbors")
        distances, indices = index.search(query_embeddings, k)
        
        return distances, indices


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Build or query FAISS index")
    parser.add_argument("--build", action="store_true",
                       help="Build index from embeddings")
    parser.add_argument("--search", action="store_true",
                       help="Search index with query embeddings")
    parser.add_argument("--embeddings", type=str,
                       help="Path to embeddings .npy file")
    parser.add_argument("--query", type=str,
                       help="Path to query embeddings .npy file")
    parser.add_argument("--index", type=str, default="data/faiss_index/index.faiss",
                       help="Path to index file")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of nearest neighbors")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    builder = FAISSIndexBuilder(config_path=args.config)
    
    if args.build:
        # Load embeddings
        logger.info(f"Loading embeddings from {args.embeddings}")
        data = np.load(args.embeddings, allow_pickle=True).item()
        embeddings = data['embeddings']
        metadata = {
            'ids': data.get('ids', []),
            'descriptions': data.get('descriptions', [])
        }
        
        # Build index
        index = builder.build_index(embeddings)
        
        # Save index
        builder.save_index(index, args.index, metadata)
    
    elif args.search:
        # Load index
        index, metadata = builder.load_index(args.index)
        
        # Load query
        logger.info(f"Loading query from {args.query}")
        query_data = np.load(args.query, allow_pickle=True).item()
        query_embeddings = query_data['embeddings']
        
        # Search
        distances, indices = builder.search(index, query_embeddings, k=args.k)
        
        # Print results
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            query_id = query_data.get('ids', [f"query_{i}"])[i]
            print(f"\nTop {args.k} matches for {query_id}:")
            for rank, (dist, idx) in enumerate(zip(dists, idxs), 1):
                match_id = metadata['ids'][idx] if metadata else f"seq_{idx}"
                print(f"  {rank}. {match_id} (similarity: {dist:.4f})")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
