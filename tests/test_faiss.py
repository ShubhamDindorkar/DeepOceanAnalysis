"""
Unit tests for FAISS indexing and search.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_faiss import FAISSIndexBuilder


class TestFAISSIndexBuilder:
    """Test FAISS index building and search."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(100, 768).astype('float32')
    
    @pytest.fixture
    def builder(self, tmp_path):
        """Create FAISSIndexBuilder with test config."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
faiss:
  index_type: "IndexFlatIP"
  use_gpu: false
""")
        return FAISSIndexBuilder(config_path=str(config_file))
    
    def test_build_flat_index(self, builder, sample_embeddings):
        """Test building flat index."""
        index = builder.build_index(sample_embeddings, index_type="IndexFlatIP")
        
        assert index is not None
        assert index.ntotal == 100
        assert index.d == 768
    
    def test_search(self, builder, sample_embeddings):
        """Test similarity search."""
        index = builder.build_index(sample_embeddings)
        
        # Search for first 5 vectors
        query = sample_embeddings[:5]
        distances, indices = builder.search(index, query, k=10)
        
        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)
        
        # First result should be the query itself (or very close)
        assert indices[0, 0] == 0
    
    def test_save_load_index(self, builder, sample_embeddings, tmp_path):
        """Test saving and loading index."""
        index = builder.build_index(sample_embeddings)
        index_path = tmp_path / "test.faiss"
        
        metadata = {'ids': [f'seq_{i}' for i in range(100)]}
        builder.save_index(index, index_path, metadata)
        
        assert index_path.exists()
        
        loaded_index, loaded_metadata = builder.load_index(index_path)
        assert loaded_index.ntotal == 100
        assert len(loaded_metadata['ids']) == 100
