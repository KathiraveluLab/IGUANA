-module(iguana_accelerator).
-export([init/0, accelerated_entropy/1]).
-on_load(init/0).

init() ->
    %% Try standard priv, fallback to local src relative paths if in development/test
    PrivDir = case code:priv_dir(iguana) of
        {error, bad_name} ->
            %% Fallback for cases where the app is not fully started or we are in a sub-build
            Ebin = filename:dirname(code:which(?MODULE)),
            filename:join([filename:dirname(Ebin), "priv"]);
        Dir -> Dir
    end,
    Path = filename:join([PrivDir, "iguana_nif_accelerator"]),
    erlang:load_nif(Path, 0).

%% @doc Wrapper for the C-NIF entropy calculation.
%% If the NIF is not loaded, this will fail.
accelerated_entropy(Probabilities) ->
    calculate_entropy_nif(Probabilities).

%% @doc Placeholder for fallback (will be replaced by NIF on load)
calculate_entropy_nif(_Probabilities) ->
    erlang:nif_error("NIF not loaded").
