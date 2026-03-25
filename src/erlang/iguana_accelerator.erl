-module(iguana_accelerator).
-export([init/0, accelerated_entropy/1]).
-on_load(init/0).

init() ->
    %% Load the NIF shared library
    Path = filename:join([code:priv_dir(iguana), "iguana_nif_accelerator"]),
    ok = erlang:load_nif(Path, 0).

%% @doc Wrapper for the C-NIF entropy calculation.
%% If the NIF is not loaded, this will fail.
accelerated_entropy(Probabilities) ->
    calculate_entropy_nif(Probabilities).

%% @doc Placeholder for fallback (will be replaced by NIF on load)
calculate_entropy_nif(_Probabilities) ->
    erlang:nif_error("NIF not loaded").
