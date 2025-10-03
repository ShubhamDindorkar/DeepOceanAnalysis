#!/usr/bin/env python3
"""
Run BLAST for sequence verification and alignment.

Complements FAISS similarity search with traditional alignment-based verification.
"""

import argparse
import yaml
import subprocess
from pathlib import Path
from Bio import SeqIO
from Bio.Blast import NCBIXML
from loguru import logger
import pandas as pd


class BLASTRunner:
    """Run BLAST searches for sequence verification."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.blast_config = self.config.get('blast', {})
        self.db_path = Path(self.config['paths']['blast_db'])
        
    def run_blastn(self, query_file, database="nt", output_file=None, output_format="xml"):
        """
        Run BLASTN search.
        
        Args:
            query_file: Path to query FASTA file
            database: BLAST database name or path
            output_file: Path to output file (optional)
            output_format: Output format (xml, json, tsv, etc.)
        
        Returns:
            Path to output file
        """
        if not output_file:
            query_path = Path(query_file)
            output_file = f"data/results/{query_path.stem}_blast.xml"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if database exists locally
        db_full_path = self.db_path / database / database
        if not db_full_path.with_suffix('.nin').exists():
            logger.warning(f"Local database not found: {db_full_path}")
            logger.info(f"Using database name: {database}")
            db_arg = database
        else:
            db_arg = str(db_full_path)
        
        # Prepare BLAST command
        cmd = [
            "blastn",
            "-query", query_file,
            "-db", db_arg,
            "-out", str(output_path),
            "-outfmt", self._get_outfmt_code(output_format),
            "-evalue", str(self.blast_config.get('e_value', 1e-5)),
            "-max_target_seqs", str(self.blast_config.get('max_target_seqs', 5)),
            "-num_threads", str(self.blast_config.get('num_threads', 4))
        ]
        
        logger.info(f"Running BLASTN: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"BLAST completed successfully. Output: {output_path}")
            
            if result.stdout:
                logger.debug(f"BLAST stdout: {result.stdout}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"BLAST failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("blastn command not found. Install BLAST+ toolkit.")
            raise
    
    def _get_outfmt_code(self, format_name):
        """Map format names to BLAST outfmt codes."""
        format_map = {
            'pairwise': '0',
            'query-anchored': '1',
            'flat-query-anchored': '2',
            'xml': '5',
            'tsv': '6',
            'tsv-comments': '7',
            'asn': '8',
            'csv': '10',
            'archive': '11',
            'json': '13',
            'xml2': '16'
        }
        return format_map.get(format_name, '5')  # Default to XML
    
    def parse_blast_xml(self, xml_file, min_identity=70, min_coverage=50):
        """
        Parse BLAST XML output and extract results.
        
        Args:
            xml_file: Path to BLAST XML output
            min_identity: Minimum percent identity threshold
            min_coverage: Minimum query coverage threshold
        
        Returns:
            pandas DataFrame with results
        """
        logger.info(f"Parsing BLAST results from {xml_file}")
        
        results = []
        
        with open(xml_file) as f:
            blast_records = NCBIXML.parse(f)
            
            for record in blast_records:
                query_id = record.query
                query_len = record.query_length
                
                if not record.alignments:
                    logger.debug(f"No hits for {query_id}")
                    continue
                
                for alignment in record.alignments:
                    for hsp in alignment.hsps:
                        # Calculate metrics
                        identity_pct = (hsp.identities / hsp.align_length) * 100
                        coverage_pct = (hsp.align_length / query_len) * 100
                        
                        # Filter by thresholds
                        if identity_pct < min_identity or coverage_pct < min_coverage:
                            continue
                        
                        results.append({
                            'query_id': query_id,
                            'hit_id': alignment.hit_id,
                            'hit_def': alignment.hit_def,
                            'e_value': hsp.expect,
                            'bit_score': hsp.bits,
                            'identity_pct': identity_pct,
                            'coverage_pct': coverage_pct,
                            'alignment_length': hsp.align_length,
                            'query_start': hsp.query_start,
                            'query_end': hsp.query_end,
                            'hit_start': hsp.sbjct_start,
                            'hit_end': hsp.sbjct_end
                        })
        
        df = pd.DataFrame(results)
        logger.info(f"Parsed {len(df)} BLAST hits")
        
        return df
    
    def verify_faiss_matches(self, faiss_matches_file, sequences_file, output_file=None):
        """
        Verify FAISS matches using BLAST alignment.
        
        Args:
            faiss_matches_file: CSV file with FAISS similarity results
            sequences_file: FASTA file with query sequences
            output_file: Path to output verification results
        
        Returns:
            pandas DataFrame with combined results
        """
        logger.info("Verifying FAISS matches with BLAST")
        
        # Run BLAST
        blast_output = self.run_blastn(sequences_file)
        
        # Parse BLAST results
        blast_df = self.parse_blast_xml(blast_output)
        
        # Load FAISS matches
        faiss_df = pd.read_csv(faiss_matches_file)
        
        # Merge results
        merged = pd.merge(
            faiss_df,
            blast_df,
            on='query_id',
            how='left',
            suffixes=('_faiss', '_blast')
        )
        
        # Add verification status
        merged['verified'] = merged['hit_id'].notna()
        merged['method'] = merged.apply(
            lambda row: 'both' if row['verified'] else 'faiss_only',
            axis=1
        )
        
        if output_file:
            merged.to_csv(output_file, index=False)
            logger.info(f"Saved verification results to {output_file}")
        
        return merged


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Run BLAST searches")
    parser.add_argument("--query", type=str, required=True,
                       help="Query FASTA file")
    parser.add_argument("--database", type=str, default="nt",
                       help="BLAST database")
    parser.add_argument("--output", type=str,
                       help="Output file path")
    parser.add_argument("--format", type=str, default="xml",
                       choices=['xml', 'tsv', 'json', 'csv'],
                       help="Output format")
    parser.add_argument("--parse", action="store_true",
                       help="Parse XML output to CSV")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    runner = BLASTRunner(config_path=args.config)
    
    # Run BLAST
    output_file = runner.run_blastn(
        args.query,
        database=args.database,
        output_file=args.output,
        output_format=args.format
    )
    
    # Parse if XML and requested
    if args.parse and args.format == 'xml':
        df = runner.parse_blast_xml(output_file)
        csv_file = Path(output_file).with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved parsed results to {csv_file}")


if __name__ == "__main__":
    main()
