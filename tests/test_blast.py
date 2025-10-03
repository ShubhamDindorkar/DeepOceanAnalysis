"""
Unit tests for BLAST functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_blast import BLASTRunner


class TestBLASTRunner:
    """Test BLAST search functionality."""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create BLASTRunner with test config."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
blast:
  e_value: 1e-5
  max_target_seqs: 5
  num_threads: 2
paths:
  blast_db: "data/blast_db"
""")
        return BLASTRunner(config_path=str(config_file))
    
    def test_outfmt_mapping(self, runner):
        """Test BLAST output format mapping."""
        assert runner._get_outfmt_code('xml') == '5'
        assert runner._get_outfmt_code('tsv') == '6'
        assert runner._get_outfmt_code('json') == '13'
        assert runner._get_outfmt_code('unknown') == '5'  # Default
    
    # Note: Actual BLAST tests require BLAST+ installation and databases
    # These would be integration tests rather than unit tests
