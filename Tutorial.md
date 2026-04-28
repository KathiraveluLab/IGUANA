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
{ok, _} = application:ensure_all_started(iguana).
```

## 2. Basic Inference with Guardrails

IGUANA uses `iguana_hf_controller` to manage the lifecycle of the Python GPU worker and the Hugging Face model.

### Loading a Model
Start the inference engine with a specific model ID (e.g., gpt2).

```erlang
{ok, P} = iguana_hf_controller:start_inference_engine("gpt2").
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

## 8. Verifying the Guardrail Activity

To confirm that IGUANA is actively steering the model, monitor your shell for the following signals:

### Console Logs
When the swarm detects a high-uncertainty state, it will log:
`[IGUANA_GUARD] Entropy spike detected! Injecting Soft SkewPNN bias vector...`

This indicates that the Erlang swarm has successfully calculated a corrective bias and sent it back to the Python GPU worker before the next token was chosen.

### Swarm Statistics
You can inspect the workload and intervention count of the entire swarm at any time:
```erlang
iguana_swarm_dashboard:display().
```

### Contextual Sensitivity
Change the domain to see the guardrail adapt its sensitivity:
```erlang
% Very strict: guards will trigger on minor uncertainty
iguana_meta_guard:update_context(medical).

% Relaxed: guards will only trigger on extreme uncertainty
iguana_meta_guard:update_context(creative).
```

### Manual Stress Test
If you want to force the guardrail to trigger on almost every token (for testing purposes), set a manually extreme threshold:
```erlang
% Set threshold to 0.5 (GPT-2 average entropy is usually > 1.0)
iguana_entropy_guard:set_threshold(0.5).

% Run a generic prompt
iguana_hf_controller:generate_sequence(P, "The").
```
You should now see a constant stream of `[IGUANA_GUARD] Entropy spike detected!` messages in your terminal.

---

> [!TIP]
> **GPU Requirements**: Running full inference requires a CUDA-capable GPU with at least 14GB of VRAM. If you don't have a GPU, you can still run the Erlang components and benchmarks to verify the swarm logic.
