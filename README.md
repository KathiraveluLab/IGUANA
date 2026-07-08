# IGUANA 🦎

**Integrated Guardrails for Unbiased and Adaptive Neural Network Architectures**

IGUANA is a polyglot AI safety framework that decouples guardrail evaluation from the sequential forward pass of generative neural networks. An Erlang/OTP Supervisor Swarm evaluates the Shannon entropy of each token distribution concurrently with PyTorch inference, applying SkewPNN soft-bias corrections or hard-veto interrupts.

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
│   │   ├── usecases/            # Domain-specific safety sentinels (New)
│   │   │   ├── README.md        # Use cases documentation
│   │   │   ├── iguana_clinical_usecase.erl
│   │   │   ├── iguana_water_usecase.erl
│   │   │   ├── iguana_financial_usecase.erl
│   │   │   └── iguana_legal_usecase.erl
│   │   ├── iguana_app.erl       # Application callback
│   │   ├── iguana_sup.erl       # 10-actor swarm supervisor
│   │   ├── iguana_meta_guard.erl # Context Broker (Dynamic Thresholds)
│   │   ├── iguana_entropy_guard.erl # Parallel safety actors
│   │   ├── iguana_accelerator.erl # NIF harness for CPU SIMD software acceleration
│   │   ├── iguana_cli.erl        # Unified command-line interface
│   │   ├── iguana_stat_analyzer.erl # Statistical profiling
│   │   ├── iguana_swarm_dashboard.erl # Swarm monitoring
│   │   └── iguana_hf_controller.erl # RLHF/Inference relay
│   ├── c/                       # Native C sources (SIMD Accelerated)
│   │   ├── iguana_nif_accelerator.c # Primary native C SIMD vector kernel
│   │   └── iguana_nif.c          # Alternative entropy logic
│   ├── python/                  # Python GPU worker sources
│   │   ├── iguana_bridge.py     # Python-to-Erlang bridge
│   │   ├── iguana_hf_runner.py  # Hugging Face model runner
│   │   └── iguana_logits_processor.py # LogitsProcessor hook
│   └── eval/                    # Benchmark suite
│       └── benchmark.erl        # Cross-platform latency benchmark
├── test/
│   ├── iguana_entropy_guard_test.erl # EUnit suites
│   ├── iguana_SUITE.erl         # Common Test integration suite
│   └── iguana_usecases_SUITE.erl # Use case integration suite (New)
├── include/
│   └── iguana.hrl               # Shared record definitions
├── config/
│   └── ssl_dist.conf            # SSL/TLS distribution configuration
├── scripts/
│   └── generate_certs.sh        # Self-signed certificate generation script
├── Makefile                     # Native NIF build system (Linux/macOS)
└── rebar.config                 # rebar3 orchestration config
```

## Setup

### 1. Compile Native Components
IGUANA utilizes native software acceleration via a C-NIF targeting CPU-level SIMD vector instructions (AVX2 on x86_64, NEON on ARMv8). On Linux/macOS, use the included Makefile:

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
IGUANA is optimized for Erlang/OTP 26+ and uses `rebar3` for lifecycle management:

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
Compare native Erlang performance against the SIMD-accelerated NIF:

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
│                     └──►│  C-NIF Software Accelerator │  │
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
- **C-NIF Software Accelerator**: SIMD-optimized native NIF kernel (targeting CPU AVX2/NEON vector instructions) that scales to high-frequency token bursts.
- **Distributed Cluster**: `iguana_cluster_manager` handles automated node discovery and scale-out safety.
- **Performance**: Verified **1.27x speedup** and **300x IPC reduction** (via Top-K telemetry).

## Key Telemetry & Enforcement Modes

### 1. Dual-Threshold Safety Enforcement
- **Hard Veto**: If token distribution entropy falls below `0.5` (extremely low entropy, representing a deterministic policy violation), the guard issues a hard `veto_token` interrupt, forcing the generator to yield an EOS token and halt.
- **Soft Correction**: If entropy exceeds the context-specific threshold (indicating an entropy spike/uncertainty), the guard injects a SkewPNN bias vector to softly adjust logits pre-softmax.
- **Accept**: Otherwise, generation is allowed without modification.

### 2. Telemetry Modes: Asynchronous vs. Preventative Block-Mode
- **Asynchronous (Default)**: Telemetry is cast out-of-band to the swarm, and the GPU inference thread continues to the next forward pass in parallel. To prevent visual leakage on streaming interfaces, a client-side sliding window buffer of size $N=1$ or $2$ tokens can be deployed to intercept safety overrides before they render.
- **Preventative Block-Mode**: By setting the environment variable `IGUANA_BLOCK_MODE=true`, the PyTorch logits processor blocks synchronously on Erlang evaluation, enforcing absolute zero-leak compliance before returning scores.

### 3. Deployment & Cluster Security
- **Cloud-Native & Virtualization**: The SIMD vector functions run entirely as standard user-space instructions, making IGUANA native-friendly for containerized environments (Docker/Kubernetes) without requiring special host privileges or GPU pass-through.
- **Swarm Security**: Node-to-node telemetry and configuration updates are secured via TLS encryption, cookie-based authentication, and private network overlays.

## Sample Usage

```erlang
application:start(iguana).
{ok, P} = iguana_hf_controller:start_inference_engine("meta-llama/Llama-2-7b-hf").
iguana_hf_controller:generate_sequence(P, <<"Tell me about climate change.">>).
iguana_hf_controller:stop(P).
```

- **Erlang primary node** owns the process lifecycle and safety telemetry.
- **Python worker** owns the GPU matrix multiplications.
- **ErlPort** bridges them with sub-2ms IPC overhead.

## Domain-Specific Use Cases

IGUANA includes executable implementations for four high-stakes domains:
1. **Clinical Healthcare & Mental Health**: Mitigates selective refusal bias using strict thresholds and soft logit corrections.
2. **Water Reuse & Circular Economy**: Leverages soft-biases for salinity and contaminant constraints in agricultural decision-making instead of binary blocking.
3. **Financial Services Risk Assessment**: Balances loan recommendation probabilities across underrepresented rural regions to counter demographic dataset bias.
4. **Legal & Governmental Decision Support**: Prevents racial and socioeconomic bias skews in parole and sentencing draft summaries without rigid blocklists.

For detailed design and modules, see the [Domain-Specific Use Cases README](src/erlang/usecases/README.md).

## Citation

If you use this work in your research, please cite the following publication:

* Kathiravelu, P. and Galinac Grbac, T. **Integrated Guardrails for Unbiased and Adaptive Neural Network Architectures.** In _the IEEE International Symposium on Systems Engineering (ISSE)._ Accepted. 8 pages. September 2026.
