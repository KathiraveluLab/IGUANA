"""
iguana_hf_runner.py
Executed natively by the Erlang Master Node via ErlPort.

This module boots the foundation model into GPU memory and attaches the
IGUANA LogitsProcessor hook to autoregressive inference.  It also registers
iguana_bridge.handle_guardrail_message as the erlport message handler so that
any Erlang-to-Python cast (inject_bias, veto_token) is processed immediately.

Runtime Requirements:
    - Python >= 3.10
    - Erlang/OTP >= 26  (invoked via ErlPort from iguana_hf_controller.erl)
    - erlport            (pip install erlport)
    - PyTorch >= 2.0 with CUDA support
    - Hugging Face `transformers` >= 4.38
    NOTE: Model loading / generation calls are guarded by MODEL is not None
    checks so this module is importable on CPU-only machines for testing.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # type: ignore
import torch
import iguana_bridge
from iguana_logits_processor import IguanaLogitsProcessor

MODEL     = None
TOKENIZER = None


# ---------------------------------------------------------------------------
# Called synchronously by iguana_hf_controller at startup (python:call/4)
# ---------------------------------------------------------------------------

def register_message_handler():
    """
    Registers iguana_bridge.handle_guardrail_message with erlport so every
    Erlang-to-Python message (inject_bias, veto_token) is dispatched to the
    correct handler without polling.  Must be called once before generation.
    """
    from erlport.erlang import set_message_handler
    set_message_handler(iguana_bridge.handle_guardrail_message)
    print("[PYTHON WORKER] erlport message handler registered.")


# ---------------------------------------------------------------------------
# Model lifecycle  (invoked via python:cast from iguana_hf_controller)
# ---------------------------------------------------------------------------

def load_model(model_id_bytes):
    """
    Invoked by the Erlang Master Controller to load the LLM into GPU memory.
    ErlPort serialises Erlang strings as Python bytes, hence the decode.
    """
    global MODEL, TOKENIZER
    model_id = model_id_bytes.decode("utf-8")
    print(f"[PYTHON WORKER] Received Erlang Deployment Directive. Booting model: {model_id}")

    # Requires a CUDA-capable GPU with >= 14 GB VRAM for a 7B-parameter model.
    MODEL     = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, device_map="auto")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    print(f"[PYTHON WORKER] Neural Network '{model_id}' successfully bootstrapped to VRAM.")


def generate(prompt_bytes):
    """
    Commands the loaded PyTorch pipeline to synthesise a response.
    The IguanaLogitsProcessor fires send_activation_state() on every token
    step, which casts telemetry to iguana_entropy_guard without blocking.
    """
    if MODEL is None or TOKENIZER is None:
        print("[PYTHON WORKER] Model not loaded — cannot generate.")
        return

    prompt = prompt_bytes.decode("utf-8")
    print(f"[PYTHON WORKER] Initiating sequence generation: '{prompt}'")

    eos_token_id    = TOKENIZER.eos_token_id
    iguana_hook     = IguanaLogitsProcessor(eos_token_id)
    processors      = LogitsProcessorList([iguana_hook])

    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    outputs = MODEL.generate(
        **inputs,
        max_new_tokens=100,
        logits_processor=processors
    )
    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    print(f"[PYTHON WORKER] Generation complete: {response}")
    return response
