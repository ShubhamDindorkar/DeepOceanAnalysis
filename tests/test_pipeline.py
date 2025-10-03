"""
Integration tests for the full pipeline.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_pipeline import AnalysisPipeline


class TestPipeline:
    """Test end-to-end pipeline."""
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline with test config."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  name: "zhihan1996/DNABERT-2-117M"
  max_length: 512
  batch_size: 4
  device: "cpu"
  use_cerebras: false

faiss:
  index_type: "IndexFlatIP"
  use_gpu: false

similarity:
  known_threshold: 0.82
  top_k: 5

blast:
  enabled: false

clustering:
  umap:
    n_neighbors: 5
    min_dist: 0.1
    n_components: 2
    metric: "cosine"
  hdbscan:
    min_cluster_size: 3
    min_samples: 2

paths:
  faiss_index: "data/faiss_index"
  blast_db: "data/blast_db"
  
exa:
  enabled: false
""")
        return AnalysisPipeline(config_path=str(config_file))
    
    # Full pipeline tests would require sample data and model downloads
    # These are marked as integration tests
    
    @pytest.mark.integration
    def test_full_pipeline(self, pipeline, tmp_path):
        """Test complete pipeline execution."""
        # This would require sample FASTA files
        # Skip in unit tests
        pytest.skip("Integration test - requires sample data and models")
