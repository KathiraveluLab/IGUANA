# IGUANA Swarm API Reference

This document provides a high-level technical reference for the IGUANA guardrail swarm. 

## Meta Guard (iguana_meta_guard)
The central orchestrator for domain-specific safety thresholds.

### API Core
- **start_link/0**: Initializes the global Meta Guard server.
- **set_domain(Domain)**: Shifts the swarm context (e.g., medical, finance) and broadcasts new thresholds.
- **get_current_domain/0**: Returns the active safety context.

## Entropy Guard (iguana_entropy_guard)
Distributed workers responsible for real-time token monitoring and bias injection.

### API Core
- **monitor_token(EnginePid, Probabilities)**: Asynchronously evaluates token entropy and triggers SkewPNN bias if the threshold is breached.
- **set_threshold(Float)**: Broadcasts a new entropy limit to the entire swarm.
- **get_stats(Pid)**: Retrieves runtime metrics (entropy spikes, injections) for a specific worker.

## Accelerator (iguana_accelerator)
Hardware-accelerated entropy calculation with native fallback.

### API Core
- **accelerated_entropy(List)**: Calculates Shannon entropy with C-NIF optimization.

---

## Swarm Topology
- **Process Group**: iguana_swarm (managed via pg).
- **Communication**: Hybrid gen_server:call (synchronous threshold updates) and gen_server:cast (asynchronous token monitoring).
