"""
IGUANA Python-Erlang Bridge
Simulates the Machine Learning Inference Engine sending token probabilities
to the asynchronous Erlang Supervisor Swarm (iguana_entropy_guard) via ErlPort.
"""
from erlport.erlterms import Atom # type: ignore
from erlport.erlang import cast, self # type: ignore

ACTIVE_BIAS_VECTOR = None
GENERATION_HALTED = False

def send_activation_state(probabilities):
    """
    Called after the PyTorch/TensorFlow forward pass.
    Sends the token distribution to the Erlang guardrail supervisor.
    """
    # The registered name of the gen_server in Erlang
    guardrail_server = Atom(b"iguana_entropy_guard")
    
    # EnginePid is the current Erlang process representing this Python instance
    engine_pid = self()
    
    # Cast `{evaluate_entropy, EnginePid, Probabilities}` to the Erlang gen_server
    message = (Atom(b"evaluate_entropy"), engine_pid, probabilities)
    cast(guardrail_server, message)
    
    # The inference engine does NOT block here. It continues to generate the next token.
    # This solves the catastrophic generational latency issue defined in RQ1!
    return True

def handle_guardrail_message(message):
    """
    Asynchronous callback triggered when Erlang sends a message back to Python.
    """
    if isinstance(message, tuple) and len(message) == 2:
        command, payload = message
        
        if command == Atom(b"inject_bias"):
            # A2/SkewPNN Bias Weights received from Erlang Adaptivity Supervisor
            bias_weights = payload
            print(f"[PYTHON INFERENCE] Received dynamic bias injection: {bias_weights}")
            apply_bias_to_logits(bias_weights)
            
        elif command == Atom(b"veto_token"):
            # Strict safety threshold breached (e.g., Toxicity)
            print("[PYTHON INFERENCE] Erlang Guardrail issued a hard veto. Halting generation.")
            halt_generation()

def get_adjusted_logits(original_logits):
    """
    Applies the active dynamic bias vector (SkewPNN constraint) to the output logits.
    """
    global ACTIVE_BIAS_VECTOR
    if ACTIVE_BIAS_VECTOR is not None:
        # In a real PyTorch implementation, this would be a tensor addition
        # For this simulation, we assume lists and add element-wise
        adjusted = [log + bias for log, bias in zip(original_logits, ACTIVE_BIAS_VECTOR)]
        # Consume the bias (decay mechanism)
        ACTIVE_BIAS_VECTOR = None
        return adjusted
    return original_logits

def update_context_trust(trust_score: float):
    """
    Translates a user trust score (0.0 to 1.0) into an entropy threshold.
    0.0 (Layperson) -> Strict Threshold (1.5)
    1.0 (Clinician) -> Relaxed Threshold (3.0)
    Resolves Context Blindness by allowing fluid adjustments per session.
    """
    threshold = 1.5 + (trust_score * 1.5)
    guardrail_server = Atom(b"iguana_entropy_guard")
    message = (Atom(b"set_trust_threshold"), threshold)
    cast(guardrail_server, message)
    print(f"[PYTHON INFERENCE] Context trust set to {trust_score:.2f}. Adjusting Erlang threshold to {threshold:.2f}")

def apply_bias_to_logits(bias_weights):
    global ACTIVE_BIAS_VECTOR
    ACTIVE_BIAS_VECTOR = bias_weights

def halt_generation():
    global GENERATION_HALTED
    GENERATION_HALTED = True
