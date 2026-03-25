-module(iguana_hf_controller).
-export([start_inference_engine/1, generate_sequence/2, stop/1, receive_loop/1]).

%% ---------------------------------------------------------------------------
%% Public API
%% ---------------------------------------------------------------------------

%% @doc Starts the Python inference GPU process via ErlPort.
%%
%% ErlPort's python:start/1 spawns a Python interpreter and returns its PID.
%% The Python path is resolved relative to the project root so the call works
%% whether the node is started via `rebar3 shell` or a compiled release.
start_inference_engine(ModelName) ->
    %% Resolve src/python relative to the application's code path.
    %% code:which/1 gives us the full path of the compiled .beam for this
    %% module; we walk two levels up (ebin -> project root) then into src/python.
    BeamPath    = code:which(?MODULE),
    ProjRoot    = filename:dirname(filename:dirname(BeamPath)),
    PythonPath  = filename:join([ProjRoot, "src", "python"]),

    %% Fallback: if we are running from the source tree without a _build dir
    %% (e.g., directly via `erl`) use the relative path instead.
    EffectivePath = case filelib:is_dir(PythonPath) of
        true  -> PythonPath;
        false -> "src/python"
    end,

    {ok, P} = python:start([{python, "python3"}, {python_path, EffectivePath}]),

    %% Register the message handler on the Python side BEFORE any cast
    %% arrives so no Erlang message is lost.  python:call/4 is synchronous
    %% and blocks until Python returns — safe here at startup.
    ok = python:call(P, iguana_hf_runner, register_message_handler, []),

    %% Boot the model asynchronously (cast = non-blocking).
    python:cast(P, iguana_hf_runner, load_model, [ModelName]),

    %% Spawn the Erlang-side receive loop that relays {inject_bias, Weights}
    %% and {veto_token, _} messages from iguana_entropy_guard back to Python.
    _LoopPid = spawn(?MODULE, receive_loop, [P]),

    {ok, P}.

%% @doc Commands the Python GPU worker to autoregressively generate a response.
%%
%% This cast fires and returns immediately.  Inside Python, the IguanaLogits-
%% Processor calls iguana_bridge.send_activation_state() on every token step,
%% which in turn casts {evaluate_entropy, EnginePid, Probs} to iguana_entropy_guard.
generate_sequence(P, PromptText) ->
    python:cast(P, iguana_hf_runner, generate, [PromptText]),
    ok.

%% @doc Terminates the Python GPU worker and flushes VRAM.
stop(P) ->
    python:stop(P).

%% ---------------------------------------------------------------------------
%% Internal: Erlang → Python feedback relay
%% ---------------------------------------------------------------------------

%% @doc Receive loop that sits between iguana_entropy_guard and the Python process.
%%
%% iguana_entropy_guard sends  `EnginePid ! {inject_bias, Weights}`
%% or                          `EnginePid ! {veto_token, Reason}`
%% where EnginePid is the erlport P returned by python:start/1.
%% erlport processes accept messages; this loop catches them and issues a
%% synchronous python:call so the result is confirmed before continuing.
receive_loop(P) ->
    receive
        {inject_bias, BiasWeights} ->
            %% Forward the SkewPNN bias vector to the Python global state.
            python:call(P, iguana_bridge, apply_bias_to_logits, [BiasWeights]),
            receive_loop(P);

        {veto_token, _Reason} ->
            %% Hard safety halt — set GENERATION_HALTED = True in Python.
            python:call(P, iguana_bridge, halt_generation, []),
            receive_loop(P);

        stop ->
            ok
    end.
