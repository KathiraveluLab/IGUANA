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
    BeamPath = code:which(?MODULE),
    %% We might be in _build/default/lib/iguana/ebin/
    %% We need to find the real project root where .venv and src/ are.

    %% Strategy: Look for the 'src' directory by walking up from the beam path.
    ProjRoot = find_project_root(filename:dirname(BeamPath)),
    PythonPath = filename:join([ProjRoot, "src", "python"]),

    EffectivePath = case filelib:is_dir(PythonPath) of
        true  -> PythonPath;
        false -> "src/python"
    end,

    %% Detection of local virtual environment (.venv)
    VenvPython = filename:join([ProjRoot, ".venv", "bin", "python3"]),
    PythonExe = case filelib:is_file(VenvPython) of
        true  -> VenvPython;
        false -> "python3"
    end,

    {ok, P} = python:start([{python, PythonExe}, {python_path, EffectivePath}]),

    %% Register the message handler on the Python side BEFORE any cast
    %% arrives so no Erlang message is lost.  python:call/4 is synchronous
    %% and blocks until Python returns — safe here at startup.
    true = python:call(P, iguana_hf_runner, register_message_handler, []),

    %% Boot the model asynchronously (cast = non-blocking).
    BinaryModel = ensure_binary(ModelName),
    python:call(P, iguana_hf_runner, load_model, [BinaryModel], [async]),

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
    BinaryPrompt = ensure_binary(PromptText),
    python:call(P, iguana_hf_runner, generate, [BinaryPrompt], [async]),
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

%% @doc Ensures the input is a binary (converts lists/strings).
ensure_binary(B) when is_binary(B) -> B;
ensure_binary(L) when is_list(L)   -> list_to_binary(L).

%% @doc Recursively walks up the directory tree to find the project root (containing rebar.config).
find_project_root("/") -> ".";
find_project_root(Dir) ->
    case filelib:is_file(filename:join(Dir, "rebar.config")) of
        true  -> Dir;
        false ->
            Parent = filename:dirname(Dir),
            if Parent == Dir -> "."; %% Reached root
               true -> find_project_root(Parent)
            end
    end.
