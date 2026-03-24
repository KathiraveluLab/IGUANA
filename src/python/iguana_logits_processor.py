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
        
        # Dispatch the live sequence probabilities across the ErlPort bridge to the BEAM VM.
        # To minimize cross-system IPC, we serialize only the required vector dimensions.
        batch_probs = probs[0].tolist() 
        iguana_bridge.send_activation_state(batch_probs)
        
        # 3. Stateful Bias Adjustment (SkewPNN implementation)
        # Verify if an asynchronous bias tensor was transmitted from the Erlang actors
        # during previous generations.
        if iguana_bridge.ACTIVE_BIAS_VECTOR is not None:
            # Cast the Erlang payload to a corresponding float tensor
            bias_tensor = torch.tensor(iguana_bridge.ACTIVE_BIAS_VECTOR, device=scores.device)
            
            # Execute element-wise tensor addition to gently rebalance the generation logits
            if bias_tensor.size(0) < scores.size(-1):
                padded_bias = torch.zeros(scores.size(-1), device=scores.device)
                padded_bias[:bias_tensor.size(0)] = bias_tensor
                scores += padded_bias
            else:
                scores += bias_tensor[:scores.size(-1)]
            
            # Flush the matrix to decay the bias constraint natively
            iguana_bridge.ACTIVE_BIAS_VECTOR = None
            
        return scores
