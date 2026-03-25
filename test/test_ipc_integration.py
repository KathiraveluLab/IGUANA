"""
Integration tests for the IGUANA Erlang-Python IPC bridge.

These tests exercise the full message-handling logic in iguana_bridge
(handle_guardrail_message, apply_bias_to_logits, halt_generation) and the
round-trip with IguanaLogitsProcessor — without requiring a live BEAM node.

The erlport module itself is used only inside iguana_hf_runner.register_message_handler()
and inside iguana_bridge.send_activation_state() (which makes a real cast). All tests
below bypass those functions to stay pytest-runnable in CI environments without Erlang.
"""
import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/python")))

import iguana_bridge
from iguana_logits_processor import IguanaLogitsProcessor


@pytest.fixture(autouse=True)
def mock_erlport_api(monkeypatch):
    """Mock erlport.erlang API functions that are normally injected at runtime."""
    import erlport.erlang as erlang
    mock_self = lambda: b"mock_pid"
    mock_cast = lambda *args: True
    
    # Patch erlport.erlang
    monkeypatch.setattr(erlang, "self", mock_self)
    monkeypatch.setattr(erlang, "cast", mock_cast)
    
    # Also patch the names already imported into iguana_bridge
    monkeypatch.setattr(iguana_bridge, "self", mock_self)
    monkeypatch.setattr(iguana_bridge, "cast", mock_cast)
    yield


@pytest.fixture(autouse=True)
def reset_bridge_state():
    """Reset shared mutable bridge state before every test."""
    iguana_bridge.ACTIVE_BIAS_VECTOR = None
    iguana_bridge.GENERATION_HALTED  = False
    yield
    iguana_bridge.ACTIVE_BIAS_VECTOR = None
    iguana_bridge.GENERATION_HALTED  = False


# ---------------------------------------------------------------------------
# TC-IPC-1: inject_bias message populates ACTIVE_BIAS_VECTOR
# ---------------------------------------------------------------------------

def test_inject_bias_sets_active_bias_vector():
    """
    Simulates the Erlang supervisor sending {inject_bias, Weights} back to Python.
    The message handler must deserialise the payload and store it in ACTIVE_BIAS_VECTOR.
    """
    # Erlport decodes Erlang atoms as bytes; simulate that encoding here.
    from erlport.erlterms import Atom
    weights = [0.1, -0.4, 0.5, 0.2]
    iguana_bridge.handle_guardrail_message((Atom(b"inject_bias"), weights))

    assert iguana_bridge.ACTIVE_BIAS_VECTOR == [0.1, -0.4, 0.5, 0.2]
    assert iguana_bridge.GENERATION_HALTED is False


# ---------------------------------------------------------------------------
# TC-IPC-2: veto_token message sets GENERATION_HALTED
# ---------------------------------------------------------------------------

def test_veto_token_sets_generation_halted():
    """
    Simulates the Erlang supervisor sending a hard veto command.
    GENERATION_HALTED must flip to True so the logits processor forces EOS.
    """
    from erlport.erlterms import Atom
    iguana_bridge.handle_guardrail_message((Atom(b"veto_token"), None))

    assert iguana_bridge.GENERATION_HALTED is True
    assert iguana_bridge.ACTIVE_BIAS_VECTOR is None


# ---------------------------------------------------------------------------
# TC-IPC-3: Unknown command is handled gracefully (no exception, no state change)
# ---------------------------------------------------------------------------

def test_unknown_command_is_silently_ignored():
    """
    An unknown Erlang command must not raise an exception or mutate state.
    This guards the bridge against future protocol additions.
    """
    from erlport.erlterms import Atom
    iguana_bridge.handle_guardrail_message((Atom(b"unknown_command"), "payload"))

    assert iguana_bridge.ACTIVE_BIAS_VECTOR is None
    assert iguana_bridge.GENERATION_HALTED is False


# ---------------------------------------------------------------------------
# TC-IPC-4: Malformed messages are silently dropped
# ---------------------------------------------------------------------------

def test_malformed_non_tuple_message_ignored():
    iguana_bridge.handle_guardrail_message("not a tuple")
    assert iguana_bridge.ACTIVE_BIAS_VECTOR is None

def test_malformed_wrong_length_tuple_ignored():
    iguana_bridge.handle_guardrail_message(("cmd",))        # length 1, not 2
    assert iguana_bridge.ACTIVE_BIAS_VECTOR is None


# ---------------------------------------------------------------------------
# TC-IPC-5: Full round-trip — inject_bias → IguanaLogitsProcessor applies it
# ---------------------------------------------------------------------------

def test_round_trip_inject_bias_applied_by_logits_processor():
    """
    Simulates the complete IPC round-trip:
      1. Erlang sends {inject_bias, Weights}  →  handle_guardrail_message stores it
      2. IguanaLogitsProcessor.__call__ reads ACTIVE_BIAS_VECTOR and adds it to scores
      3. After one generation step the bias is flushed (single-use decay)
    """
    from erlport.erlterms import Atom

    weights = [1.0, -1.0, 0.5, 0.0, 2.0]
    iguana_bridge.handle_guardrail_message((Atom(b"inject_bias"), weights))

    processor  = IguanaLogitsProcessor(eos_token_id=2)
    input_ids  = torch.tensor([[1, 2, 3]])
    base_scores = torch.zeros(1, 5)

    modified = processor(input_ids, base_scores.clone())

    assert modified[0, 0].item() == pytest.approx(1.0)
    assert modified[0, 1].item() == pytest.approx(-1.0)
    assert modified[0, 2].item() == pytest.approx(0.5)
    assert modified[0, 3].item() == pytest.approx(0.0)
    assert modified[0, 4].item() == pytest.approx(2.0)

    # Bias must be consumed (decayed) after a single generation step
    assert iguana_bridge.ACTIVE_BIAS_VECTOR is None


# ---------------------------------------------------------------------------
# TC-IPC-6: veto_token → IguanaLogitsProcessor forces EOS on next step
# ---------------------------------------------------------------------------

def test_round_trip_veto_forces_eos():
    """
    Simulates a hard veto:
      1. Erlang sends {veto_token, _}  →  GENERATION_HALTED = True
      2. IguanaLogitsProcessor.__call__ detects the flag and forces EOS
    """
    from erlport.erlterms import Atom

    iguana_bridge.handle_guardrail_message((Atom(b"veto_token"), None))
    assert iguana_bridge.GENERATION_HALTED is True

    processor   = IguanaLogitsProcessor(eos_token_id=2)
    input_ids   = torch.tensor([[1, 2, 3]])
    scores      = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    modified    = processor(input_ids, scores)

    # EOS (index 2) must be 0.0; everything else -inf
    assert modified[0, 2].item() == pytest.approx(0.0)
    for idx in [0, 1, 3, 4]:
        assert modified[0, idx].item() == float("-inf")
