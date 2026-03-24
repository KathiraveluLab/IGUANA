-module(iguana_hf_controller).
-export([start_inference_engine/1, generate_sequence/2, stop/1]).

%% Starts the Python inference GPU process via ErlPort, enforcing Erlang as the orchestrator
start_inference_engine(ModelName) ->
    %% Resolve the Python worker directory relative to the current working directory.
    %% After restructuring, all Python modules reside in src/python/.
    PythonPath = filename:join([code:priv_dir_opt(iguana, "src/python"), "src/python"]),
    EffectivePath = case filelib:is_dir(PythonPath) of
        true  -> PythonPath;
        false -> "src/python"   %% fallback for CI / dev environments without a release
    end,
    {ok, P} = python:start([{python, "python3"}, {python_path, EffectivePath}]),

    %% Instruct Python to asynchronously boot the Heavy LLM into GPU memory (VRAM)
    python:cast(P, iguana_hf_runner, load_model, [ModelName]),

    %% Return the PID of the active Python pipeline worker to the Erlang Swarm
    {ok, P}.

%% Commands the active Python GPU worker to autoregressively synthesize a prompt sequence
generate_sequence(P, PromptText) ->
    %% This cast implicitly triggers the generation loop in PyTorch, which will subsequently
    %% automatically fire send_activation_state() callbacks back to iguana_entropy_guard.erl
    %% on every single autoregressive step.
    python:cast(P, iguana_hf_runner, generate, [PromptText]),
    ok.

%% Securely terminates the Python GPU worker process and flushes VRAM
stop(P) ->
    python:stop(P).
