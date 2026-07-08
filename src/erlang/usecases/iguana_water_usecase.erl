-module(iguana_water_usecase).
-export([evaluate/3]).

%% @doc Evaluates water reuse queries, injecting soft-correction for CEC/salinity hazards.
-spec evaluate(string(), list(integer()), list(float())) ->
    ok | {inject_bias, list(float()), list(integer())} | {veto_token, term()}.
evaluate(Query, Indices, Probabilities) ->
    %% Update Meta-Guard context to general circular economy
    ok = iguana_meta_guard:update_context(general),
    %% Detect if agricultural query touches on high-risk contaminants
    case is_contaminant_query(Query) of
        true ->
            %% Inject soft bias to steer away from unsafe disposal
            %% while proposing blending / treatment alternatives
            K = length(Indices),
            Xi = K / 2.0,
            Omega = K / 4.0,
            Alpha = 2.0,
            A2 = 0.5,
            BiasVector = [
                A2 * iguana_entropy_guard:skew_normal_cdf((I - Xi) / Omega, Alpha)
                || I <- lists:seq(1, K)
            ],
            {inject_bias, BiasVector, Indices};
        false ->
            iguana_entropy_guard:evaluate_entropy_sync(Indices, Probabilities)
    end.

is_contaminant_query(Query) ->
    (string:find(Query, "fertigation") =/= nomatch) orelse
    (string:find(Query, "salinity") =/= nomatch) orelse
    (string:find(Query, "contaminant") =/= nomatch).
