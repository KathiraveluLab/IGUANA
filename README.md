# IGUANA
Integrated Guardrails for Unbiased and Adaptive Neural Network Architectures

## Prerequisites

Before running IGUANA, ensure that both Python (3.10+) and Erlang/OTP are installed on your system.

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip erlang
```

**For macOS (via Homebrew):**
```bash
brew install python erlang
```

## Setup

To run the Python-Erlang bridge (`src/iguana_bridge.py`), you must install the required Python dependencies:

```bash
pip install -r requirements.txt --break-system-packages
```
*(Note: If using a Python virtual environment, omit the `--break-system-packages` flag).*
