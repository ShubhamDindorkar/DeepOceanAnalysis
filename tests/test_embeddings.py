"""
Unit tests for sequence embedding generation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.embed_sequences import SequenceEmbedder


class TestSequenceEmbedder:
    """Test sequence embedding generation."""
    
    @pytest.fixture
    def embedder(self, tmp_path):
        """Create SequenceEmbedder with test config."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  name: "zhihan1996/DNABERT-2-117M"
  max_length: 512
  batch_size: 4
  device: "cpu"
  use_cerebras: false
""")
        # Note: This will attempt to download the model
        # In real tests, you might want to mock this
        return SequenceEmbedder(config_path=str(config_file))
    
    def test_embed_sequence_shape(self, embedder):
        """Test that embedding has correct shape."""
        sequence = "ATCGATCGATCG"
        embedding = embedder.embed_sequence(sequence)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0  # Has some dimension
    
    # Additional tests would require model download and can be slow
    # Mark as integration tests or skip in CI
