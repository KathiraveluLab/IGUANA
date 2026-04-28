"""
IGUANA Hugging Face LogitsProcessor Integration
Bridges state-of-the-art transformer architectures (LLaMA, GPT, Mistral, T5)
to the asynchronous Erlang Supervisor Swarm using the Python-side bridge.
"""
import torch
from transformers import LogitsProcessor

import iguana_bridge

class IguanaLogitsProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor that intercepts autoregressive sequence generation
    at the decoding terminus (pre-softmax calculation) to dispatch telemetry to
    the Erlang swarm. It natively applies soft corrections via tensor addition
    and enforces hard terminations via EOS token forcing.
    """
    
    def __init__(self, eos_token_id: int):
        """
        Initializes the processor.
        :param eos_token_id: The specific End-Of-Sequence token ID for the active 
                             Hugging Face model tokenizer (e.g., 2 for LLaMA).
        """
        self.eos_token_id = eos_token_id
        self.vocab_initialized = False
        print(f"[IGUANA HOOK] Initialized LogitsProcessor. Guardrail EOS tied to ID: {self.eos_token_id}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        try:
            # 0. One-time Swarm Initialization
            if not self.vocab_initialized:
                vocab_size = scores.shape[-1]
                iguana_bridge.initialize_swarm(vocab_size)
                self.vocab_initialized = True

            # 1. Hardware Terminus Override Check
            if iguana_bridge.GENERATION_HALTED:
                scores[:, :] = -float('inf')
                scores[:, self.eos_token_id] = 0.0
                return scores
            
            # 2. Extract Token Probabilities (Telemetry)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            k = 100
            topk_probs, topk_indices = torch.topk(probs[0], k)
            
            telemetry_probs = topk_probs.tolist()
            telemetry_indices = topk_indices.tolist()
            
            rest_mass = 1.0 - sum(telemetry_probs)
            telemetry_probs.append(max(0.0, rest_mass))
            
            iguana_bridge.send_activation_state(telemetry_indices, telemetry_probs)
            
            # 3. Targeted Bias Adjustment (SkewPNN implementation)
            if iguana_bridge.ACTIVE_BIAS_VECTOR is not None and iguana_bridge.ACTIVE_BIAS_INDICES is not None:
                weights = torch.tensor(iguana_bridge.ACTIVE_BIAS_VECTOR, device=scores.device)
                target_indices = torch.tensor(iguana_bridge.ACTIVE_BIAS_INDICES, device=scores.device)
                scores[0, target_indices] += weights
                iguana_bridge.ACTIVE_BIAS_VECTOR = None
                iguana_bridge.ACTIVE_BIAS_INDICES = None
                
            return scores
        except Exception as e:
            print(f"[IGUANA ERROR] LogitsProcessor failed: {e}")
            import traceback
            traceback.print_exc()
            return scores
