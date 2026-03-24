# IGUANA

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
│   │   ├── iguana.app.src       # OTP application descriptor
│   │   ├── iguana_app.erl       # Application callback (boots supervision tree)
│   │   ├── iguana_sup.erl       # Top-level one_for_one supervisor
│   │   ├── iguana_entropy_guard.erl  # Shannon entropy guardrail gen_server
│   │   └── iguana_hf_controller.erl  # Erlang master controller (ErlPort)
│   ├── python/                  # Python GPU worker sources
│   │   ├── iguana_bridge.py     # Python-to-Erlang bridge (activation state)
│   │   ├── iguana_hf_runner.py  # Hugging Face model runner (ErlPort entry)
│   │   ├── iguana_logits_processor.py  # LogitsProcessor hook for transformers
│   │   └── spike.py             # Entropy spike simulation utility
│   └── eval/                    # Benchmark suite
│       ├── benchmark.py         # CPython latency benchmark
│       ├── benchmark.erl        # BEAM latency benchmark
│       └── Eval.md              # Benchmark documentation
├── test/
│   └── iguana_entropy_guard_test.erl  # EUnit correctness tests (6 test cases)
├── _paper/                      # LaTeX manuscript
│   ├── main.tex
│   ├── architecture.tex
│   ├── discussion.tex
│   ├── results.tex
│   ├── conclusion.tex
│   └── references.bib
├── rebar.config                 # rebar3 build config (declares erlport dependency)
└── .github/workflows/erlang.yml # CI: rebar3 compile + eunit + benchmark
```

## Setup

### 1. Install rebar3

```bash
curl -fsSL https://s3.amazonaws.com/rebar3/rebar3 -o ~/.local/bin/rebar3
chmod +x ~/.local/bin/rebar3
```

### 2. Fetch Erlang dependencies and compile

```bash
rebar3 get-deps
rebar3 compile
```

This fetches [ErlPort](https://github.com/hdima/erlport) (`0.10.1`) from hex.pm and compiles all four Erlang modules.

### 3. Install Python dependencies

```bash
pip install torch transformers
```

### 4. Run the EUnit test suite

```bash
rebar3 eunit
```

Expected output: **6 passed, 0 failed.**

### 5. Run the benchmark suite

```bash
# Python benchmark
python src/eval/benchmark.py

# Erlang BEAM benchmark
cd src/eval && erlc benchmark.erl && erl -noshell -s benchmark run -s init stop
```

## Running Live Inference (GPU Required)

To run live inference with a real Hugging Face model, uncomment the model loading and generation calls in `src/python/iguana_hf_runner.py`, then start the Erlang node:

```erlang
application:start(iguana).
{ok, P} = iguana_hf_controller:start_inference_engine("meta-llama/Llama-2-7b-hf").
iguana_hf_controller:generate_sequence(P, <<"Tell me about climate change.">>).
iguana_hf_controller:stop(P).
```

The Erlang process `iguana_entropy_guard` will evaluate entropy concurrently and inject SkewPNN bias corrections without blocking generation.

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              Erlang/OTP BEAM                │
│  iguana_app → iguana_sup → iguana_entropy_guard  │
│         (one_for_one restart strategy)      │
└──────────────┬──────────────────────────────┘
               │ ErlPort (bidirectional IPC)
┌──────────────▼──────────────────────────────┐
│            Python / PyTorch                 │
│  iguana_hf_runner → IguanaLogitsProcessor   │
│         → iguana_bridge (send_activation_state) │
└─────────────────────────────────────────────┘
```

- **Erlang master** owns the process lifecycle and safety telemetry.
- **Python worker** owns the GPU matrix multiplications.
- **ErlPort** bridges them with sub-2ms IPC overhead.


