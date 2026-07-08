-module(iguana_financial_usecase).
-export([evaluate/3]).

%% @doc Evaluates financial advising queries, rebalancing demographic/geographic loan skew.
-spec evaluate(string(), list(integer()), list(float())) ->
    ok | {inject_bias, list(float()), list(integer())} | {veto_token, term()}.
evaluate(Region, Indices, Probabilities) ->
    %% Update Meta-Guard context to financial
    ok = iguana_meta_guard:update_context(financial),
    %% Underrepresented regions trigger bias rebalancing vector
    case is_underrepresented(Region) of
        true ->
            %% Inject soft corrective bias to rebalance demographic/geographic logits
            K = length(Indices),
            Xi = K / 2.0,
            Omega = K / 4.0,
            Alpha = 2.0,
            A2 = 0.35,
            BiasVector = [
                A2 * iguana_entropy_guard:skew_normal_cdf((I - Xi) / Omega, Alpha)
                || I <- lists:seq(1, K)
            ],
            {inject_bias, BiasVector, Indices};
        false ->
            %% Evaluate standard categorical entropy
            iguana_entropy_guard:evaluate_entropy_sync(Indices, Probabilities)
    end.

is_underrepresented(Region) ->
    (string:find(Region, "rural") =/= nomatch) orelse
    (string:find(Region, "underrepresented") =/= nomatch).
