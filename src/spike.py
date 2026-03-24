import time
import random

import sys
import os

# Dynamically add the current directory to sys.path to resolve the IDE lint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iguana_bridge import send_activation_state # type: ignore

def simulate_inference():
    print("[SPIKE] Starting Python Inference Engine Loop...")
    for token_idx in range(1, 6):
        # Generate some mock probability distributions
        probs = [random.uniform(0.01, 0.4) for _ in range(5)]
        
        print(f"[SPIKE] Token {token_idx} generated. Forwarding probs to Erlang: {probs}")
        send_activation_state(probs)
        
        # Simulate local LLM generation latency
        time.sleep(0.1) 
        print(f"[SPIKE] Token {token_idx} forward pass complete (Non-blocking!)")

if __name__ == "__main__":
    simulate_inference()
