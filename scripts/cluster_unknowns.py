#!/usr/bin/env python3
"""
Cluster unknown sequences using UMAP + HDBSCAN.

Groups sequences that don't match known references into coherent clusters.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns


class UnknownClusterer:
    """Cluster unknown sequences using dimensionality reduction and density-based clustering."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.clustering_config = self.config['clustering']
        self.umap_params = self.clustering_config['umap']
        self.hdbscan_params = self.clustering_config['hdbscan']
    
    def reduce_dimensions(self, embeddings, n_components=2):
        """
        Reduce embedding dimensionality using UMAP.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            n_components: Target dimensionality
        
        Returns:
            Low-dimensional representation
        """
        logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}")
        
        reducer = umap.UMAP(
            n_neighbors=self.umap_params['n_neighbors'],
            min_dist=self.umap_params['min_dist'],
            n_components=n_components,
            metric=self.umap_params['metric'],
            random_state=self.umap_params.get('random_state', 42)
        )
        
        embedding_2d = reducer.fit_transform(embeddings)
        logger.info(f"UMAP completed. Output shape: {embedding_2d.shape}")
        
        return embedding_2d, reducer
    
    def cluster(self, embeddings):
        """
        Perform density-based clustering with HDBSCAN.
        
        Args:
            embeddings: Embeddings to cluster (can be original or UMAP-reduced)
        
        Returns:
            Cluster labels
        """
        logger.info("Clustering with HDBSCAN")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_params['min_cluster_size'],
            min_samples=self.hdbscan_params.get('min_samples', 5),
            cluster_selection_epsilon=self.hdbscan_params.get('cluster_selection_epsilon', 0.0),
            metric=self.hdbscan_params.get('metric', 'euclidean')
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters")
        logger.info(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        return labels, clusterer
    
    def compute_cluster_summaries(self, embeddings, labels, metadata=None):
        """
        Compute summary statistics for each cluster.
        
        Args:
            embeddings: Original embeddings
            labels: Cluster labels
            metadata: Optional metadata dict with sequence IDs, etc.
        
        Returns:
            pandas DataFrame with cluster summaries
        """
        logger.info("Computing cluster summaries")
        
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Exclude noise
        
        summaries = []
        
        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            
            # Basic stats
            size = mask.sum()
            centroid = cluster_embeddings.mean(axis=0)
            
            # Intra-cluster distance (cohesion)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            avg_distance = distances.mean()
            
            summary = {
                'cluster_id': cluster_id,
                'size': size,
                'avg_intra_distance': avg_distance,
                'centroid_norm': np.linalg.norm(centroid)
            }
            
            # Add sequence IDs if available
            if metadata and 'ids' in metadata:
                cluster_ids = [metadata['ids'][i] for i, m in enumerate(mask) if m]
                summary['sequence_ids'] = ','.join(cluster_ids[:10])  # First 10
                summary['num_sequences'] = len(cluster_ids)
            
            summaries.append(summary)
        
        df = pd.DataFrame(summaries)
        logger.info(f"Generated summaries for {len(df)} clusters")
        
        return df
    
    def visualize_clusters(self, embedding_2d, labels, output_file=None, metadata=None):
        """
        Visualize clusters in 2D space.
        
        Args:
            embedding_2d: 2D UMAP embedding
            labels: Cluster labels
            output_file: Path to save plot
            metadata: Optional metadata for annotations
        """
        logger.info("Generating cluster visualization")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Color by cluster
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Use a colormap
        colors = sns.color_palette('tab20', n_colors=max(n_clusters, 20))
        
        for label in unique_labels:
            mask = labels == label
            
            if label == -1:
                # Noise points in gray
                ax.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c='gray',
                    s=20,
                    alpha=0.3,
                    label='Noise'
                )
            else:
                ax.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[colors[label % len(colors)]],
                    s=50,
                    alpha=0.7,
                    label=f'Cluster {label}'
                )
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Unknown Sequences Clustering\n{n_clusters} clusters, {list(labels).count(-1)} noise points')
        
        # Legend (limit to first 20 clusters)
        handles, labels_legend = ax.get_legend_handles_labels()
        if len(handles) > 21:
            ax.legend(handles[:21], labels_legend[:21], bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_file}")
        
        plt.close()
    
    def run_pipeline(self, embeddings_file, output_dir="data/clusters", visualize=True):
        """
        Run complete clustering pipeline.
        
        Args:
            embeddings_file: Path to embeddings .npy file
            output_dir: Directory to save results
            visualize: Generate visualization plots
        
        Returns:
            dict with results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_file}")
        data = np.load(embeddings_file, allow_pickle=True).item()
        embeddings = data['embeddings']
        metadata = {
            'ids': data.get('ids', []),
            'descriptions': data.get('descriptions', [])
        }
        
        # Reduce dimensions
        embedding_2d, reducer = self.reduce_dimensions(embeddings)
        
        # Cluster
        labels, clusterer = self.cluster(embedding_2d)
        
        # Compute summaries
        summaries = self.compute_cluster_summaries(embeddings, labels, metadata)
        
        # Save results
        summaries.to_csv(output_path / 'cluster_summary.csv', index=False)
        logger.info(f"Saved cluster summary to {output_path / 'cluster_summary.csv'}")
        
        # Save labels
        labels_df = pd.DataFrame({
            'sequence_id': metadata['ids'],
            'cluster_label': labels
        })
        labels_df.to_csv(output_path / 'cluster_labels.csv', index=False)
        logger.info(f"Saved cluster labels to {output_path / 'cluster_labels.csv'}")
        
        # Visualize
        if visualize:
            self.visualize_clusters(
                embedding_2d,
                labels,
                output_file=output_path / 'clusters_umap.png',
                metadata=metadata
            )
        
        return {
            'labels': labels,
            'embedding_2d': embedding_2d,
            'summaries': summaries,
            'clusterer': clusterer
        }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Cluster unknown sequences")
    parser.add_argument("--embeddings", type=str, required=True,
                       help="Input embeddings .npy file")
    parser.add_argument("--output", type=str, default="data/clusters",
                       help="Output directory")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    clusterer = UnknownClusterer(config_path=args.config)
    clusterer.run_pipeline(
        args.embeddings,
        output_dir=args.output,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
