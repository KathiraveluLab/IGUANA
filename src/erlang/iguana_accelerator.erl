-module(iguana_accelerator).
-export([init/0, accelerated_entropy/1]).
-on_load(init/0).

init() ->
    %% Try standard priv, fallback to local src relative paths if in development/test
    PrivDir = case code:priv_dir(iguana) of
        {error, bad_name} ->
            %% Fallback for cases where the app is not fully started or we are in a sub-build
            Ebin = filename:dirname(code:which(?MODULE)),
            AppRoot = filename:dirname(Ebin),
            %% Check both AppRoot/priv and project_root/priv
            case filelib:is_dir(filename:join(AppRoot, "priv")) of
                true -> filename:join(AppRoot, "priv");
                false -> "priv" %% current dir priv
            end;
        Dir -> Dir
    end,
    Path = filename:join([PrivDir, "iguana_nif_accelerator"]),
    case erlang:load_nif(Path, 0) of
        ok -> ok;
        {error, Reason} ->
            error_logger:warning_msg("NIF not loaded (~p), using Erlang fallback.~n", [Reason]),
            ok
    end.

%% @doc Wrapper for the C-NIF entropy calculation with Erlang fallback.
%% Calculates the Shannon entropy of a probability distribution.
%% Attempts to use C-NIF acceleration if available, otherwise falls back to Erlang.
-spec accelerated_entropy([float()]) -> float().
accelerated_entropy(Probabilities) ->
    %% Pack floats into a native double binary (64-bit) for efficient NIF processing
    Bin = << <<P:64/float-native>> || P <- Probabilities >>,
    try
        calculate_entropy_nif(Bin)
    catch
        _:_ -> erl_entropy(Probabilities)
    end.

erl_entropy(Probabilities) ->
    lists:foldl(fun(P, Acc) ->
        if P > 0.0 -> Acc - (P * math:log2(P));
           true    -> Acc
        end
    end, 0.0, Probabilities).

%% @doc Placeholder for fallback (will be replaced by NIF on load)
calculate_entropy_nif(_Probabilities) ->
    erlang:nif_error("NIF not loaded").
