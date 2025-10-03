#!/usr/bin/env python3
"""
End-to-end analysis pipeline orchestrator.

Coordinates all steps: preprocessing, embedding, similarity search, clustering.
"""

import argparse
import yaml
from pathlib import Path
from loguru import logger
import sys

# Import pipeline modules
from fetch_ncbi import NCBIFetcher
from embed_sequences import SequenceEmbedder
from build_faiss import FAISSIndexBuilder
from run_blast import BLASTRunner
from cluster_unknowns import UnknownClusterer
from report_generator import ReportGenerator


class AnalysisPipeline:
    """End-to-end deep-sea sequence analysis pipeline."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.embedder = SequenceEmbedder(config_path)
        self.faiss_builder = FAISSIndexBuilder(config_path)
        self.blast_runner = BLASTRunner(config_path)
        self.clusterer = UnknownClusterer(config_path)
        self.reporter = ReportGenerator(config_path)
        
        logger.info("Pipeline initialized")
    
    def run_full_analysis(self, input_fasta, reference_fasta=None, output_dir="data/results"):
        """
        Run complete analysis pipeline.
        
        Args:
            input_fasta: Path to query sequences (FASTA)
            reference_fasta: Path to reference sequences (optional, for building index)
            output_dir: Directory for all outputs
        
        Returns:
            dict with paths to results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Deep-Sea Sequence Analysis Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Generate query embeddings
        logger.info("\n[Step 1/6] Generating query embeddings...")
        query_emb_file = output_path / "query_embeddings.npy"
        self.embedder.embed_fasta(input_fasta, query_emb_file)
        
        # Step 2: Build/load reference index
        logger.info("\n[Step 2/6] Setting up reference index...")
        index_file = Path(self.config['paths']['faiss_index']) / "reference.faiss"
        
        if reference_fasta and not index_file.exists():
            logger.info("Building new reference index...")
            ref_emb_file = output_path / "reference_embeddings.npy"
            self.embedder.embed_fasta(reference_fasta, ref_emb_file)
            
            import numpy as np
            ref_data = np.load(ref_emb_file, allow_pickle=True).item()
            index = self.faiss_builder.build_index(ref_data['embeddings'])
            self.faiss_builder.save_index(index, index_file, metadata=ref_data)
        
        # Step 3: Similarity search
        logger.info("\n[Step 3/6] Running similarity search...")
        index, ref_metadata = self.faiss_builder.load_index(index_file)
        
        import numpy as np
        query_data = np.load(query_emb_file, allow_pickle=True).item()
        
        k = self.config['similarity']['top_k']
        distances, indices = self.faiss_builder.search(index, query_data['embeddings'], k=k)
        
        # Save similarity results
        import pandas as pd
        similarity_results = []
        threshold = self.config['similarity']['known_threshold']
        
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            query_id = query_data['ids'][i]
            max_sim = dists[0]
            
            for rank, (dist, idx) in enumerate(zip(dists, idxs), 1):
                match_id = ref_metadata['ids'][idx]
                similarity_results.append({
                    'query_id': query_id,
                    'rank': rank,
                    'match_id': match_id,
                    'similarity': float(dist),
                    'is_known': dist >= threshold
                })
        
        similarity_df = pd.DataFrame(similarity_results)
        similarity_file = output_path / "similarity_matches.csv"
        similarity_df.to_csv(similarity_file, index=False)
        logger.info(f"Saved similarity results to {similarity_file}")
        
        # Step 4: BLAST verification (optional)
        if self.config.get('blast', {}).get('enabled', False):
            logger.info("\n[Step 4/6] Running BLAST verification...")
            try:
                blast_file = self.blast_runner.run_blastn(input_fasta)
                logger.info(f"BLAST results saved to {blast_file}")
            except Exception as e:
                logger.warning(f"BLAST verification failed: {e}")
                logger.warning("Continuing without BLAST verification")
        else:
            logger.info("\n[Step 4/6] Skipping BLAST verification (disabled in config)")
        
        # Step 5: Cluster unknowns
        logger.info("\n[Step 5/6] Clustering unknown sequences...")
        
        # Filter unknown sequences
        unknown_mask = similarity_df.groupby('query_id')['similarity'].max() < threshold
        unknown_ids = unknown_mask[unknown_mask].index.tolist()
        
        logger.info(f"Found {len(unknown_ids)} unknown sequences to cluster")
        
        if len(unknown_ids) > 0:
            # Extract unknown embeddings
            unknown_indices = [i for i, qid in enumerate(query_data['ids']) if qid in unknown_ids]
            unknown_emb = query_data['embeddings'][unknown_indices]
            unknown_data = {
                'embeddings': unknown_emb,
                'ids': [query_data['ids'][i] for i in unknown_indices]
            }
            
            unknown_emb_file = output_path / "unknown_embeddings.npy"
            np.save(unknown_emb_file, unknown_data)
            
            # Cluster
            cluster_dir = output_path / "clusters"
            self.clusterer.run_pipeline(unknown_emb_file, cluster_dir, visualize=True)
        else:
            logger.info("No unknown sequences to cluster")
        
        # Step 6: Generate report
        logger.info("\n[Step 6/6] Generating final report...")
        report_file = output_path / "analysis_report.html"
        self.reporter.generate_report(output_path, report_file)
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info(f"\nResults saved to: {output_path}")
        logger.info(f"  - Similarity matches: {similarity_file}")
        if len(unknown_ids) > 0:
            logger.info(f"  - Cluster results: {cluster_dir}")
        logger.info(f"  - Final report: {report_file}")
        
        return {
            'similarity_file': similarity_file,
            'cluster_dir': cluster_dir if len(unknown_ids) > 0 else None,
            'report_file': report_file
        }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Run end-to-end analysis pipeline")
    parser.add_argument("--input", type=str, required=True,
                       help="Input query sequences (FASTA)")
    parser.add_argument("--reference", type=str,
                       help="Reference sequences (FASTA) for building index")
    parser.add_argument("--output", type=str, default="data/results",
                       help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    pipeline = AnalysisPipeline(config_path=args.config)
    pipeline.run_full_analysis(
        input_fasta=args.input,
        reference_fasta=args.reference,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
