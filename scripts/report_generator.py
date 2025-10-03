#!/usr/bin/env python3
"""
Generate analysis reports using Exa for literature enrichment.
"""

import argparse
import yaml
import os
from pathlib import Path
from loguru import logger
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize report generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.exa_enabled = self.config.get('exa', {}).get('enabled', False)
        
        if self.exa_enabled:
            try:
                from exa_py import Exa
                api_key = os.getenv('EXA_API_KEY')
                if api_key:
                    self.exa = Exa(api_key=api_key)
                    logger.info("Exa client initialized")
                else:
                    logger.warning("EXA_API_KEY not found, literature lookup disabled")
                    self.exa_enabled = False
            except ImportError:
                logger.warning("exa_py not installed, literature lookup disabled")
                self.exa_enabled = False
    
    def fetch_literature(self, taxon_name, max_results=5):
        """
        Fetch relevant literature using Exa.
        
        Args:
            taxon_name: Taxon or species name
            max_results: Maximum number of results
        
        Returns:
            List of literature results
        """
        if not self.exa_enabled:
            return []
        
        try:
            query = f"{taxon_name} deep sea taxonomy genomics"
            results = self.exa.search_and_contents(
                query=query,
                type=self.config['exa'].get('search_type', 'neural'),
                num_results=max_results,
                text=True
            )
            
            literature = []
            for result in results.results:
                literature.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.text[:200] if hasattr(result, 'text') else ""
                })
            
            logger.info(f"Found {len(literature)} literature references for {taxon_name}")
            return literature
            
        except Exception as e:
            logger.error(f"Failed to fetch literature: {e}")
            return []
    
    def generate_html_report(self, results_dir, output_file):
        """
        Generate HTML report from analysis results.
        
        Args:
            results_dir: Directory containing analysis results
            output_file: Path to output HTML file
        """
        results_path = Path(results_dir)
        
        # Load results
        similarity_file = results_path / "similarity_matches.csv"
        cluster_summary_file = results_path / "clusters" / "cluster_summary.csv"
        
        if not similarity_file.exists():
            logger.error(f"Similarity results not found: {similarity_file}")
            return
        
        similarity_df = pd.read_csv(similarity_file)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep-Sea Sequence Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .known {{ color: #27ae60; font-weight: bold; }}
        .unknown {{ color: #e74c3c; font-weight: bold; }}
        .literature {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <h1>🌊 Deep-Sea Sequence Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p><strong>Total sequences analyzed:</strong> {similarity_df['query_id'].nunique()}</p>
        <p><strong>Known matches (similarity ≥ {self.config['similarity']['known_threshold']}):</strong> 
           <span class="known">{similarity_df.groupby('query_id')['similarity'].max().ge(self.config['similarity']['known_threshold']).sum()}</span></p>
        <p><strong>Unknown sequences:</strong> 
           <span class="unknown">{similarity_df.groupby('query_id')['similarity'].max().lt(self.config['similarity']['known_threshold']).sum()}</span></p>
    </div>
    
    <h2>Top Similarity Matches</h2>
    <table>
        <tr>
            <th>Query ID</th>
            <th>Best Match</th>
            <th>Similarity</th>
            <th>Status</th>
        </tr>
"""
        
        # Add top matches
        top_matches = similarity_df[similarity_df['rank'] == 1].head(20)
        threshold = self.config['similarity']['known_threshold']
        
        for _, row in top_matches.iterrows():
            status = "Known" if row['similarity'] >= threshold else "Unknown"
            status_class = "known" if status == "Known" else "unknown"
            
            html += f"""
        <tr>
            <td>{row['query_id']}</td>
            <td>{row['match_id']}</td>
            <td>{row['similarity']:.4f}</td>
            <td class="{status_class}">{status}</td>
        </tr>
"""
        
        html += """
    </table>
"""
        
        # Add cluster information if available
        if cluster_summary_file.exists():
            cluster_df = pd.read_csv(cluster_summary_file)
            
            html += f"""
    <h2>Unknown Sequence Clusters</h2>
    <p>Found <strong>{len(cluster_df)}</strong> distinct clusters of unknown sequences.</p>
    <table>
        <tr>
            <th>Cluster ID</th>
            <th>Size</th>
            <th>Avg. Intra-distance</th>
            <th>Representative Sequences</th>
        </tr>
"""
            
            for _, row in cluster_df.iterrows():
                seq_ids = row.get('sequence_ids', 'N/A')
                html += f"""
        <tr>
            <td>{row['cluster_id']}</td>
            <td>{row['size']}</td>
            <td>{row.get('avg_intra_distance', 'N/A'):.4f}</td>
            <td>{seq_ids}</td>
        </tr>
"""
            
            html += """
    </table>
"""
        
        # Add visualization if exists
        cluster_plot = results_path / "clusters" / "clusters_umap.png"
        if cluster_plot.exists():
            html += """
    <h2>Cluster Visualization</h2>
    <img src="clusters/clusters_umap.png" alt="UMAP Cluster Visualization" style="max-width: 100%; height: auto;">
"""
        
        html += """
    <h2>Methodology</h2>
    <p>This analysis used:</p>
    <ul>
        <li><strong>Embedding model:</strong> """ + self.config['model']['name'] + """</li>
        <li><strong>Similarity threshold:</strong> """ + str(self.config['similarity']['known_threshold']) + """</li>
        <li><strong>Clustering:</strong> UMAP + HDBSCAN for unknowns</li>
        <li><strong>Tools:</strong> FAISS (similarity search), Cerebras (inference), Exa (literature)</li>
    </ul>
    
    <hr>
    <p><em>Generated by Deep-Sea Sequence Analysis Pipeline</em></p>
</body>
</html>
"""
        
        # Write HTML
        with open(output_file, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report: {output_file}")
    
    def generate_report(self, results_dir, output_file):
        """
        Generate comprehensive report (wrapper for format selection).
        
        Args:
            results_dir: Directory containing results
            output_file: Path to output file
        """
        self.generate_html_report(results_dir, output_file)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Generate analysis reports")
    parser.add_argument("--results", type=str, required=True,
                       help="Results directory")
    parser.add_argument("--output", type=str, default="report.html",
                       help="Output report file")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    generator = ReportGenerator(config_path=args.config)
    generator.generate_report(args.results, args.output)


if __name__ == "__main__":
    main()
