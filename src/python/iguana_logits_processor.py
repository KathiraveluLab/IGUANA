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
        print(f"[IGUANA HOOK] Initialized LogitsProcessor. Guardrail EOS tied to ID: {self.eos_token_id}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Intercepts the unnormalized probability scores across the vocabulary space
        for the upcoming token generation.
        """
        # 1. Hardware Terminus Override Check
        if iguana_bridge.GENERATION_HALTED:
            # If the Erlang Supervisor Swarm issued a veto_token command, we must
            # immediately algorithmically yield by forcing the entire probability 
            # space to -infinity, except for the explicit EOS token.
            scores[:, :] = -float('inf')
            scores[:, self.eos_token_id] = 0.0
            return scores
        
        # 2. Extract Token Probabilities (Telemetry)
        # Apply softmax across the tensor dimension to calculate normalized probabilities.
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # Top-K Optimization: To resolve the "Severe Serialization Bottleneck",
        # we only transmit the top 100 most probable tokens. 
        k = 100
        topk_probs, topk_indices = torch.topk(probs[0], k)
        
        # Convert to lists for ErlPort serialization
        telemetry_probs = topk_probs.tolist()
        telemetry_indices = topk_indices.tolist()
        
        # Append the sum of the remaining probability mass ("Rest") as the last element.
        rest_mass = 1.0 - sum(telemetry_probs)
        telemetry_probs.append(max(0.0, rest_mass))
        
        # Send both indices and probabilities to Erlang
        iguana_bridge.send_activation_state(telemetry_indices, telemetry_probs)
        
        # 3. Targeted Bias Adjustment (SkewPNN implementation)
        # Verify if an asynchronous bias tensor was transmitted from the Erlang actors
        if iguana_bridge.ACTIVE_BIAS_VECTOR is not None and iguana_bridge.ACTIVE_BIAS_INDICES is not None:
            weights = torch.tensor(iguana_bridge.ACTIVE_BIAS_VECTOR, device=scores.device)
            target_indices = torch.tensor(iguana_bridge.ACTIVE_BIAS_INDICES, device=scores.device)
            
            # Scatter the bias weights only to the specific target indices
            # This ensures mathematical correctness for any vocabulary size.
            scores[0, target_indices] += weights
            
            # Flush state
            iguana_bridge.ACTIVE_BIAS_VECTOR = None
            iguana_bridge.ACTIVE_BIAS_INDICES = None
            
        return scores
