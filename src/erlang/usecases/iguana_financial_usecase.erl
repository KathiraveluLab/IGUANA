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
            {inject_bias, [0.35 || _ <- lists:seq(1, K)], Indices};
        false ->
            iguana_entropy_guard:evaluate_entropy_sync(Indices, Probabilities)
    end.

is_underrepresented(Region) ->
    string:find(Region, "rural") =/= nomatch or
    string:find(Region, "underrepresented") =/= nomatch.
