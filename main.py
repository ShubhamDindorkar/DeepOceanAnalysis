#!/usr/bin/env python3
"""
Deep-Sea NCBI Analysis Pipeline
Main entry point for the analysis pipeline.

Usage:
    python main.py --help
    python main.py --input data/raw/sequences.fasta --output data/results/
"""

import argparse
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from analyze_pipeline import AnalysisPipeline
from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deep-Sea NCBI Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis
  python main.py --input data/raw/sequences.fasta --output data/results/
  
  # With reference database
  python main.py --input queries.fasta --reference silva.fasta --output results/
  
  # Custom config
  python main.py --input data.fasta --config my_config.yaml
  
For more examples, see: docs/usage_examples.md
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input FASTA file with query sequences'
    )
    
    parser.add_argument(
        '--reference', '-r',
        type=str,
        help='Reference FASTA file for building index (optional)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/results',
        help='Output directory (default: data/results)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Validate config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║      🌊  Deep-Sea NCBI Analysis Pipeline  🧬           ║
    ║                                                          ║
    ║  Powered by FAISS · Cerebras · Exa                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Config: {args.config}")
    
    # Run pipeline
    try:
        pipeline = AnalysisPipeline(config_path=args.config)
        results = pipeline.run_full_analysis(
            input_fasta=args.input,
            reference_fasta=args.reference,
            output_dir=args.output
        )
        
        print("\n✅ Analysis completed successfully!")
        print(f"\nResults saved to: {args.output}")
        if results.get('report_file'):
            print(f"📄 View report: {results['report_file']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
