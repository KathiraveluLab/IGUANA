-module(iguana_hf_controller).
-export([start_inference_engine/1, generate_sequence/2, stop/1]).

%% Starts the Python inference GPU process via ErlPort, enforcing Erlang as the orchestrator
start_inference_engine(ModelName) ->
    %% Start the Python node, dynamically setting the pythonpath to the integration directory
    {ok, P} = python:start([{python, "python3"}, {python_path, "src"}]),
    
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
