#!/usr/bin/env python3
"""
Generate embeddings for DNA sequences using DNABERT or other DNA language models.

Supports both local inference and Cerebras Cloud for large-scale processing.
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()


class SequenceEmbedder:
    """Generate embeddings for DNA sequences."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize embedder with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        self.model_name = model_config['name']
        self.max_length = model_config['max_length']
        self.batch_size = model_config['batch_size']
        self.device = model_config['device']
        self.use_cerebras = model_config.get('use_cerebras', False)
        
        if self.use_cerebras:
            logger.info("Using Cerebras for embedding generation")
            self._init_cerebras()
        else:
            logger.info(f"Loading model: {self.model_name}")
            self._init_local_model()
    
    def _init_local_model(self):
        """Initialize local HuggingFace model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("Using GPU for inference")
        else:
            self.device = "cpu"
            logger.info("Using CPU for inference")
        
        self.model.eval()
    
    def _init_cerebras(self):
        """Initialize Cerebras Cloud client."""
        # TODO: Implement Cerebras Cloud SDK integration
        logger.warning("Cerebras integration not yet implemented")
        logger.info("Falling back to local inference")
        self.use_cerebras = False
        self._init_local_model()
    
    def embed_sequence(self, sequence):
        """
        Embed a single DNA sequence.
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            numpy array of embedding
        """
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output[0]
        else:
            # Mean pooling over sequence length
            embedding = outputs.last_hidden_state.mean(dim=1)[0]
        
        return embedding.cpu().numpy()
    
    def embed_fasta(self, fasta_file, output_file=None, include_metadata=True):
        """
        Embed all sequences in a FASTA file.
        
        Args:
            fasta_file: Path to input FASTA file
            output_file: Path to output .npy file (optional)
            include_metadata: Save sequence IDs and descriptions
        
        Returns:
            dict with embeddings, ids, and descriptions
        """
        logger.info(f"Reading sequences from {fasta_file}")
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        
        logger.info(f"Embedding {len(sequences)} sequences")
        
        embeddings = []
        seq_ids = []
        descriptions = []
        
        for record in tqdm(sequences, desc="Generating embeddings"):
            seq_str = str(record.seq)
            embedding = self.embed_sequence(seq_str)
            
            embeddings.append(embedding)
            seq_ids.append(record.id)
            descriptions.append(record.description)
        
        # Stack into matrix
        embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        result = {
            'embeddings': embeddings,
            'ids': seq_ids,
            'descriptions': descriptions
        }
        
        # Save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path, result)
            logger.info(f"Saved embeddings to {output_file}")
        
        return result


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Generate sequence embeddings")
    parser.add_argument("--input", type=str, required=True,
                       help="Input FASTA file")
    parser.add_argument("--output", type=str,
                       help="Output .npy file (default: auto-generated)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = f"data/embeddings/{input_path.stem}_embeddings.npy"
    
    embedder = SequenceEmbedder(config_path=args.config)
    embedder.embed_fasta(args.input, args.output)


if __name__ == "__main__":
    main()
