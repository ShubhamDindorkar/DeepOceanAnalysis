#!/usr/bin/env python3
"""
Fetch sequences from NCBI databases.

This script downloads sequences from NCBI Entrez or downloads BLAST databases
from NCBI FTP servers.
"""

import os
import argparse
from pathlib import Path
from Bio import Entrez, SeqIO
from dotenv import load_dotenv
import yaml
from loguru import logger

load_dotenv()


class NCBIFetcher:
    """Fetch sequences from NCBI."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize NCBI fetcher with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set NCBI email (required)
        Entrez.email = self.config['ncbi']['email']
        
        # Optional API key for higher rate limits
        api_key = os.getenv('NCBI_API_KEY')
        if api_key:
            Entrez.api_key = api_key
            logger.info("Using NCBI API key for enhanced rate limits")
    
    def search_sequences(self, query, database="nucleotide", max_results=100):
        """
        Search NCBI for sequences matching a query.
        
        Args:
            query: Search query (e.g., "deep sea 16S rRNA")
            database: NCBI database to search
            max_results: Maximum number of results to return
        
        Returns:
            List of sequence IDs
        """
        logger.info(f"Searching NCBI {database} for: {query}")
        
        handle = Entrez.esearch(
            db=database,
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        logger.info(f"Found {len(id_list)} sequences")
        
        return id_list
    
    def fetch_sequences(self, id_list, output_file, database="nucleotide"):
        """
        Fetch sequences by ID and save to FASTA.
        
        Args:
            id_list: List of NCBI sequence IDs
            output_file: Path to output FASTA file
            database: NCBI database
        """
        logger.info(f"Fetching {len(id_list)} sequences from {database}")
        
        # Fetch in batches to avoid timeouts
        batch_size = 100
        all_records = []
        
        for i in range(0, len(id_list), batch_size):
            batch = id_list[i:i + batch_size]
            logger.debug(f"Fetching batch {i//batch_size + 1}")
            
            handle = Entrez.efetch(
                db=database,
                id=batch,
                rettype="fasta",
                retmode="text"
            )
            
            records = list(SeqIO.parse(handle, "fasta"))
            all_records.extend(records)
            handle.close()
        
        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        SeqIO.write(all_records, output_file, "fasta")
        logger.info(f"Saved {len(all_records)} sequences to {output_file}")
        
        return len(all_records)
    
    def download_blast_db(self, database="nt", output_dir="data/blast_db"):
        """
        Download BLAST database from NCBI FTP.
        
        Args:
            database: BLAST database name (nt, nr, 16S_ribosomal_RNA, etc.)
            output_dir: Directory to save database files
        """
        logger.info(f"Downloading BLAST database: {database}")
        logger.warning("This may take a long time for large databases like 'nt'")
        
        import subprocess
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use update_blastdb.pl script (part of BLAST+ installation)
        cmd = [
            "update_blastdb.pl",
            "--decompress",
            database,
            "--num_threads", "4"
        ]
        
        try:
            subprocess.run(cmd, cwd=output_dir, check=True)
            logger.info(f"Successfully downloaded {database} to {output_dir}")
        except FileNotFoundError:
            logger.error("update_blastdb.pl not found. Install BLAST+ toolkit.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download database: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Fetch sequences from NCBI")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--database", type=str, default="nucleotide", 
                       help="NCBI database to search")
    parser.add_argument("--max-results", type=int, default=100,
                       help="Maximum number of results")
    parser.add_argument("--output", type=str, default="data/raw/ncbi_sequences.fasta",
                       help="Output FASTA file")
    parser.add_argument("--download-blast-db", type=str,
                       help="Download BLAST database (e.g., 'nt', '16S_ribosomal_RNA')")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    fetcher = NCBIFetcher(config_path=args.config)
    
    if args.download_blast_db:
        fetcher.download_blast_db(database=args.download_blast_db)
    elif args.query:
        id_list = fetcher.search_sequences(
            query=args.query,
            database=args.database,
            max_results=args.max_results
        )
        fetcher.fetch_sequences(id_list, args.output, database=args.database)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
