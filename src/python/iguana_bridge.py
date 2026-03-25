"""
IGUANA Python-Erlang Bridge
Real bidirectional IPC between the Python ML inference engine and the
Erlang Supervisor Swarm (iguana_entropy_guard) via ErlPort.

Message flow:
  Python  ->  Erlang: cast({evaluate_entropy, EnginePid, Probs})
  Erlang  ->  Python: {inject_bias, Weights} | {veto_token, _}
"""
from erlport.erlterms import Atom
from erlport.erlang import cast, self, set_message_handler

# ---------------------------------------------------------------------------
# Shared mutable state — written by the Erlang→Python callback thread,
# read by IguanaLogitsProcessor on the PyTorch generation thread.
# ---------------------------------------------------------------------------
ACTIVE_BIAS_VECTOR = None   # List[float] | None
GENERATION_HALTED  = False  # bool


# ---------------------------------------------------------------------------
# Python → Erlang  (called after every forward pass)
# ---------------------------------------------------------------------------

def send_activation_state(probabilities: list) -> bool:
    """
    Called after the PyTorch/TensorFlow forward pass.
    Dispatches the normalised token-probability distribution to the Erlang
    guardrail supervisor as a non-blocking cast so inference is never stalled.
    """
    guardrail_server = Atom(b"iguana_entropy_guard")
    engine_pid = self()
    message = (Atom(b"evaluate_entropy"), engine_pid, probabilities)
    cast(guardrail_server, message)
    # Fire-and-forget: the inference engine continues to the next token
    # without waiting for an Erlang reply — this is the latency win (RQ1).
    return True


# ---------------------------------------------------------------------------
# Erlang → Python  (registered as the erlport message handler)
# ---------------------------------------------------------------------------

def handle_guardrail_message(message) -> None:
    """
    Asynchronous callback invoked by erlport whenever the Erlang supervisor
    sends a message back to this Python process.  Registered via
    set_message_handler() at module load in iguana_hf_runner.
    """
    global ACTIVE_BIAS_VECTOR, GENERATION_HALTED

    if not (isinstance(message, tuple) and len(message) == 2):
        return

    command, payload = message

    if command == Atom(b"inject_bias"):
        # SkewPNN bias weights from the Erlang Adaptivity Supervisor.
        # Convert the erlport-decoded list to plain Python floats.
        bias_weights = [float(w) for w in payload]
        print(f"[PYTHON INFERENCE] Received dynamic bias injection: {bias_weights}")
        ACTIVE_BIAS_VECTOR = bias_weights

    elif command == Atom(b"veto_token"):
        # Hard safety veto — stop autoregressive generation immediately.
        print("[PYTHON INFERENCE] Erlang Guardrail issued a hard veto. Halting generation.")
        GENERATION_HALTED = True


# ---------------------------------------------------------------------------
# Context-trust adaptation  (Python → Erlang cast)
# ---------------------------------------------------------------------------

def update_context_trust(trust_score: float) -> None:
    """
    Translates a user trust score (0.0–1.0) into an entropy threshold cast
    to the Erlang entropy guard, resolving the Context Blindness problem.

      0.0 (Layperson)  → strict  threshold 1.5
      1.0 (Clinician)  → relaxed threshold 3.0
    """
    threshold = 1.5 + (trust_score * 1.5)
    guardrail_server = Atom(b"iguana_entropy_guard")
    message = (Atom(b"set_trust_threshold"), threshold)
    cast(guardrail_server, message)
    print(
        f"[PYTHON INFERENCE] Context trust set to {trust_score:.2f}. "
        f"Adjusting Erlang threshold to {threshold:.2f}"
    )


# ---------------------------------------------------------------------------
# Internal helpers (used by IguanaLogitsProcessor via ACTIVE_BIAS_VECTOR)
# ---------------------------------------------------------------------------

def apply_bias_to_logits(bias_weights: list) -> None:
    """Store the Erlang-supplied bias for consumption by the logits processor."""
    global ACTIVE_BIAS_VECTOR
    ACTIVE_BIAS_VECTOR = [float(w) for w in bias_weights]


def halt_generation() -> None:
    """Signal the logits processor to force EOS on the next call."""
    global GENERATION_HALTED
    GENERATION_HALTED = True
