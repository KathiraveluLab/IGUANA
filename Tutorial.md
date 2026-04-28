# IGUANA Tutorial: Using the AI Safety Swarm 🦎

This tutorial provides a practical guide on how to use IGUANA to run safe, non-blocking AI inference with real-time guardrails.

## 1. Starting the IGUANA Environment

Before running any inference, you need to start the Erlang node and the IGUANA application.

```bash
# Start the rebar3 shell
rebar3 shell
```

Inside the shell, start the application:

```erlang
ok = application:start(iguana).
```

## 2. Basic Inference with Guardrails

IGUANA uses `iguana_hf_controller` to manage the lifecycle of the Python GPU worker and the Hugging Face model.

### Loading a Model
Start the inference engine with a specific model ID (e.g., Llama-2-7b).

```erlang
{ok, P} = iguana_hf_controller:start_inference_engine("meta-llama/Llama-2-7b-hf").
```

### Generating Text
Send a prompt to the engine. The generation happens asynchronously in the Python worker while the Erlang swarm monitors the token distributions.

```erlang
iguana_hf_controller:generate_sequence(P, <<"Explain the concept of quantum entanglement.">>).
```

### Stopping the Engine
When finished, stop the engine to free up VRAM and terminate the Python process.

```erlang
iguana_hf_controller:stop(P).
```

## 3. Dynamic Thresholds with Meta-Guard

The `iguana_meta_guard` allows you to change safety thresholds based on the application domain.

### Switching Domains
You can switch between different domains to apply more or less strict safety checks:
- `medical`: Strict (Threshold 1.8)
- `financial`: Moderate (Threshold 2.2)
- `general`: Balanced (Threshold 2.8)
- `creative`: Relaxed (Threshold 3.5)

```erlang
% Set to strict medical mode
iguana_meta_guard:update_context(medical).

% Set to relaxed creative mode
iguana_meta_guard:update_context(creative).
```

### Manual Threshold & Augmentation
You can also manually set the augmentation factor (A2) which controls the strength of the soft-bias corrections.

```erlang
% Increase bias strength
iguana_meta_guard:update_augmentation(0.15).
```

## 4. Real-Time Monitoring (Swarm Dashboard)

IGUANA provides a built-in dashboard to monitor the state of the supervisor swarm and the guardrail activity.

```erlang
iguana_swarm_dashboard:display().
```

This will output a table showing each worker PID, its current threshold, and the number of active safety injections it has performed.

## 5. Performance & Statistical Analysis

### Running Benchmarks
To compare the performance of the native Erlang implementation against the SIMD-accelerated C-NIF:

```erlang
benchmark:run().
```

### Analyzing Results
You can use `iguana_stat_analyzer` to get a detailed statistical profile of any latency data.

```erlang
% Example: Analyzing a list of latencies (in milliseconds)
Latencies = [2.1, 2.3, 1.9, 4.5, 2.2, 2.0].
iguana_stat_analyzer:report(Latencies).
```

## 6. Command Line Interface (CLI)

IGUANA also provides a unified CLI for common tasks.

```bash
# Run the benchmark
./scripts/iguana_cli bench

# Open the Swarm Dashboard
./scripts/iguana_cli dash

# Synchronize LaTeX results (for researchers)
./scripts/iguana_cli sync
```

## 7. Python-Side Integration

If you are developing custom Python logic, you can interact with the Erlang guardrails directly via the `iguana_bridge`.

```python
import iguana_bridge

# Manually switch domain from Python
iguana_bridge.update_domain_context("medical")

# Set a custom trust score (0.0 to 1.0)
iguana_bridge.update_context_trust(0.8)
```

---

> [!TIP]
> **GPU Requirements**: Running full inference requires a CUDA-capable GPU with at least 14GB of VRAM. If you don't have a GPU, you can still run the Erlang components and benchmarks to verify the swarm logic.
