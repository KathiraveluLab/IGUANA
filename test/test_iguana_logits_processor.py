import pytest
import torch
import sys
import os

# Ensure the src/python directory is in the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/python')))

import iguana_bridge
from iguana_logits_processor import IguanaLogitsProcessor

@pytest.fixture(autouse=True)
def reset_bridge_state():
    """Reset the global bridge state before each test."""
    iguana_bridge.ACTIVE_BIAS_VECTOR = None
    iguana_bridge.GENERATION_HALTED = False
    yield

def test_processor_initialization():
    """Test that the processor initializes with the correct EOS token ID."""
    processor = IguanaLogitsProcessor(eos_token_id=2)
    assert processor.eos_token_id == 2

def test_hard_veto_override():
    """
    Test that when GENERATION_HALTED is True, the processor forces the entire
    probability space to -infinity except for the EOS token (set to 0.0), 
    immediately terminating the autoregressive loop.
    """
    processor = IguanaLogitsProcessor(eos_token_id=2)
    iguana_bridge.GENERATION_HALTED = True
    
    # Mock input_ids and scores (batch size 1, vocab size 5)
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    
    modified_scores = processor(input_ids, scores)
    
    # EOS token (index 2) should be 0.0, everything else -inf
    assert modified_scores[0, 2].item() == 0.0
    assert modified_scores[0, 0].item() == float('-inf')
    assert modified_scores[0, 1].item() == float('-inf')
    assert modified_scores[0, 3].item() == float('-inf')
    assert modified_scores[0, 4].item() == float('-inf')

def test_soft_correction_bias_injection():
    """
    Test that when ACTIVE_BIAS_VECTOR is populated by the Erlang supervisor,
    the processor performs element-wise tensor addition on the scores, modifying
    the statistical trajectory, and then flushes the bias vector.
    """
    processor = IguanaLogitsProcessor(eos_token_id=2)
    
    # Erlang calculates a SkewPNN bias matrix
    iguana_bridge.ACTIVE_BIAS_VECTOR = [1.0, -1.0, 0.5, 0.0, 2.0]
    
    # Base LLM scores
    input_ids = torch.tensor([[1, 2, 3]])
    base_scores = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    
    # Apply hook
    modified_scores = processor(input_ids, base_scores.clone())
    
    # Verify tensor addition
    assert modified_scores[0, 0].item() == 1.0
    assert modified_scores[0, 1].item() == -1.0
    assert modified_scores[0, 2].item() == 0.5
    assert modified_scores[0, 3].item() == 0.0
    assert modified_scores[0, 4].item() == 2.0
    
    # Verify the bias vector was decayed (flushed) after single use
    assert iguana_bridge.ACTIVE_BIAS_VECTOR is None

def test_soft_correction_size_mismatch():
    """
    Test that the tensor addition handles size padding gracefully if the 
    vocabulary size is larger than the received bias vector.
    """
    processor = IguanaLogitsProcessor(eos_token_id=2)
    
    # Erlang payload is smaller than the vocab space
    iguana_bridge.ACTIVE_BIAS_VECTOR = [1.0, 2.0]
    
    input_ids = torch.tensor([[1, 2, 3]])
    base_scores = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    
    modified_scores = processor(input_ids, base_scores.clone())
    
    # The first two elements get the bias, the rest are unchanged (0.0)
    assert modified_scores[0, 0].item() == 1.0
    assert modified_scores[0, 1].item() == 2.0
    assert modified_scores[0, 2].item() == 0.0
    assert modified_scores[0, 3].item() == 0.0
    assert modified_scores[0, 4].item() == 0.0
