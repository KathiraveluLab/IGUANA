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
ACTIVE_BIAS_VECTOR  = None   # List[float] | None
ACTIVE_BIAS_INDICES = None   # List[int] | None
GENERATION_HALTED   = False  # bool
GUARDRAIL_PID       = None   # Resolved Erlang PID

def gen_cast(dest, message):
    """
    Sends a gen_server cast by wrapping the message in the standard 
    Erlang {'$gen_cast', Msg} format.
    """
    from erlport.erlang import cast
    cast(dest, (Atom(b"$gen_cast"), message))

def get_guardrail_dest():
    """
    Returns the destination for guardrail messages.
    Resolves the registered name to a PID on the first call to ensure 
    robust communication across the ErlPort bridge.
    """
    global GUARDRAIL_PID
    if GUARDRAIL_PID is None:
        from erlport.erlang import call
        # Remote whereis(iguana_entropy_guard)
        GUARDRAIL_PID = call(Atom(b"erlang"), Atom(b"whereis"), [Atom(b"iguana_entropy_guard")])
    return GUARDRAIL_PID


# ---------------------------------------------------------------------------
# Python → Erlang  (Initialization & Heartbeat)
# ---------------------------------------------------------------------------

def initialize_swarm(vocab_size: int) -> bool:
    """
    Synchronizes the model vocabulary size with the Erlang entropy guards.
    Ensures accurate Shannon entropy approximation for the 'Rest' mass.
    """
    guardrail_server = get_guardrail_dest()
    message = (Atom(b"set_vocab_size"), vocab_size)
    gen_cast(guardrail_server, message)
    print(f"[PYTHON BRIDGE] Swarm initialized with VocabSize={vocab_size}")
    return True


# ---------------------------------------------------------------------------
# Python → Erlang  (Per-token telemetry)
# ---------------------------------------------------------------------------

def send_activation_state(indices: list, probabilities: list) -> bool:
    """
    Called after the PyTorch/TensorFlow forward pass.
    Dispatches the Top-K indices and their probabilities to the Erlang swarm.
    """
    guardrail_server = get_guardrail_dest()
    engine_pid = self()
    # Payload: (Indices, ProbabilitiesPlustRest)
    message = (Atom(b"evaluate_entropy"), engine_pid, indices, probabilities)
    gen_cast(guardrail_server, message)
    return True


# ---------------------------------------------------------------------------
# Erlang → Python  (registered as the erlport message handler)
# ---------------------------------------------------------------------------

def handle_guardrail_message(message) -> None:
    """
    Handles incoming commands from the Erlang supervisor.
    """
    global ACTIVE_BIAS_VECTOR, ACTIVE_BIAS_INDICES, GENERATION_HALTED

    if not isinstance(message, tuple):
        return

    command = message[0]

    if command == Atom(b"inject_bias") and len(message) == 3:
        # Expected: {inject_bias, Weights, Indices}
        ACTIVE_BIAS_VECTOR = [float(w) for w in message[1]]
        ACTIVE_BIAS_INDICES = [int(i) for i in message[2]]
        print(f"[PYTHON BRIDGE] Received dynamic bias injection for {len(ACTIVE_BIAS_INDICES)} tokens.")

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
    """
    threshold = 1.5 + (trust_score * 1.5)
    guardrail_server = Atom(b"iguana_entropy_guard")
    message = (Atom(b"set_trust_threshold"), threshold)
    cast(guardrail_server, message)
    print(f"[PYTHON INFERENCE] Context trust set to {trust_score:.2f}. Threshold: {threshold:.2f}")

def update_domain_context(domain: str) -> None:
    """
    Switches the architectural context by domain (e.g., 'medical', 'creative').
    This triggers the Erlang Meta-Guard to broadcast specific thresholds.
    """
    meta_guard = Atom(b"iguana_meta_guard")
    domain_atom = Atom(domain.encode('utf-8'))
    message = (Atom(b"update_domain"), domain_atom)
    cast(meta_guard, message)
    print(f"[PYTHON INFERENCE] Architectural domain switched to: {domain}")


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
