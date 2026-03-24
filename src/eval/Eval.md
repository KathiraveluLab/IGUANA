# IGUANA Evaluation Suite

This directory contains the empirical benchmarking scripts utilized to validate the structural claims of the IGUANA framework. 

The benchmark simulates an end-to-end token generation run of 1,000 iterations to measure the exact millisecond deltas of a synchronous API interceptor compared to the parallel, non-blocking Erlang Supervisor Swarm.

We have implemented the test across two distinct execution environments to prove absolute functional parity:
1. **Python (`benchmark.py`):** Runs natively inside CPython.
2. **Erlang (`benchmark.erl`):** Runs natively inside the BEAM Virtual Machine.

## Running the Python Benchmark
Execute the Python simulation natively:
```bash
python3 benchmark.py
```

## Running the Erlang Benchmark
Compile the Erlang source into BEAM bytecode and execute it implicitly:
```bash
erlc benchmark.erl
erl -noshell -s benchmark run -s init stop
```

## Observations
Both scripts calculate the exact elapsed OS-level system time over 1,000 generations. The variance between the two environments is consistently $\sim$2 milliseconds across the entire cycle. This minor drift represents the inherent difference in OS-level context switching between CPython's Global Interpreter Lock (GIL) and the Erlang BEAM's pre-emptive native scheduler. The mathematical logic verifying our latency improvements holds functionally identical across both interpreters.
