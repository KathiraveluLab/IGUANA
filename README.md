# IGUANA 🦎

**Integrated Guardrails for Unbiased and Adaptive Neural Network Architectures**

IGUANA is a polyglot AI safety framework that decouples guardrail evaluation from the sequential forward pass of generative neural networks. An Erlang/OTP Supervisor Swarm evaluates the Shannon entropy of each token distribution concurrently with PyTorch inference, applying SkewPNN soft-bias corrections or hard-veto interrupts without blocking the generation pipeline.

## Requirements

| Component | Version |
|-----------|---------|
| Erlang/OTP | ≥ 26 |
| rebar3 | ≥ 3.22 |
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 (CUDA-capable GPU required for live inference) |
| Hugging Face `transformers` | ≥ 4.38 |

> **Note:** The model loading and generation calls in `src/python/iguana_hf_runner.py` are commented out by default. Uncomment them only when running on hardware with ≥ 14 GB VRAM (e.g., NVIDIA A100 or equivalent).

## Project Structure

```
IGUANA/
├── src/
│   ├── erlang/                  # Erlang/OTP sources
│   │   ├── iguana_app.erl       # Application callback
│   │   ├── iguana_sup.erl       # 10-actor swarm supervisor
│   │   ├── iguana_meta_guard.erl # Context Broker (Dynamic Thresholds)
│   │   ├── iguana_entropy_guard.erl # Parallel safety actors
│   │   ├── iguana_accelerator.erl # NIF harness for hardware acceleration
│   │   ├── iguana_cli.erl        # Unified command-line interface
│   │   ├── iguana_stat_analyzer.erl # Statistical profiling
│   │   ├── iguana_swarm_dashboard.erl # Swarm monitoring
│   │   └── iguana_hf_controller.erl # RLHF/Inference relay
│   ├── c/                       # Native C sources (SIMD Accelerated)
│   │   ├── iguana_nif_accelerator.c # Primary hardware kernel
│   │   └── iguana_nif.c          # Alternative entropy logic
│   ├── python/                  # Python GPU worker sources
│   │   ├── iguana_bridge.py     # Python-to-Erlang bridge
│   │   ├── iguana_hf_runner.py  # Hugging Face model runner
│   │   └── iguana_logits_processor.py # LogitsProcessor hook
│   └── eval/                    # Benchmark suite
│       └── benchmark.erl        # Cross-platform latency benchmark
├── test/
│   ├── iguana_entropy_guard_test.erl # EUnit suites
│   └── iguana_SUITE.erl         # Common Test integration suite
├── include/
│   └── iguana.hrl               # Shared record definitions
├── Makefile                     # Native NIF build system (Linux/macOS)
├── rebar.config                 # rebar3 orchestration config
└── priv/
    └── iguana_nif_accelerator.* # Hardware-accelerated binaries
```

## Setup

### 1. Compile Native Components
IGUANA utilizes hardware acceleration via a C-NIF. On Linux/macOS, use the included Makefile:

```bash
make
```

On Windows, or using `rebar3` directly (requires the `pc` plugin and a C compiler like MSVC or GCC in the path):

```bash
rebar3 compile
```

> [!TIP]
> **Robustness**: IGUANA features an automatic fallback mechanism. If the native C-NIF cannot be loaded, the system seamlessly transitions to a pure Erlang functional implementation to ensure safety continuity.

### 2. Fetch Erlang dependencies and compile
IGUANA is optimized for Erlang/OTP 25+ and uses `rebar3` for lifecycle management:

```bash
rebar3 get-deps
rebar3 compile
```

### 3. Run the Verification Suites
Execute the native correctness and integration tests:

```bash
# Unit Tests
rebar3 eunit

# Integration (Common Test)
rebar3 ct
```

### 4. Run the Performance Benchmark
Compare native Erlang performance against the hardware-accelerated NIF:

```bash
# Run unified benchmark
rebar3 shell --eval "benchmark:run(), init:stop()."
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                   Erlang/OTP BEAM                        │
│ ┌────────────────┐      ┌──────────────────────────────┐ │
│ │  Meta-Guard    │      │    Supervisor Swarm (10x)    │ │
│ │(Context Broker)├──┬──►│ [Guard] [Guard] ... [Guard]  │ │
│ └────────────────┘  │   └──────────────┬───────────────┘ │
│                     │                  │                 │
│                     │   ┌──────────────▼──────────────┐  │
│                     └──►│   C-NIF Hardware Accelerator│  │
│                         └──────────────┬──────────────┘  │
└────────────────────────────────────────┼─────────────────┘
                                         │ ErlPort (Local IPC)
┌────────────────────────────────────────▼─────────────────┐
│                   Python / PyTorch                       │
│ ┌────────────────┐      ┌──────────────────────────────┐ │
│ │ Hugging Face   │      │   IguanaLogitsProcessor      │ │
│ │ Model Runner   │◄────►│   (Soft Bias / Hard Veto)    │ │
│ └────────────────┘      └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

- **Meta-Guard**: Intelligent context broker that broadcasts domain-specific thresholds.
- **Swarm**: Decentralized pool of ten actors providing non-blocking safety telemetry.
- **Hardware Accelerator**: SIMD-optimized NIF kernel that scales to high-frequency token bursts.
- **Performance**: achieving a 1.40x speedup over standard functional implementations.
```erlang
application:start(iguana).
{ok, P} = iguana_hf_controller:start_inference_engine("meta-llama/Llama-2-7b-hf").
iguana_hf_controller:generate_sequence(P, <<"Tell me about climate change.">>).
iguana_hf_controller:stop(P).
```


- **Erlang master** owns the process lifecycle and safety telemetry.
- **Python worker** owns the GPU matrix multiplications.
- **ErlPort** bridges them with sub-2ms IPC overhead.


