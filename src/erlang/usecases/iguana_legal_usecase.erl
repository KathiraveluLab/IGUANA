-module(iguana_legal_usecase).
-export([evaluate/3]).

%% @doc Evaluates legal drafts/sentences, mitigating racial and socioeconomic bias skews.
-spec evaluate(string(), list(integer()), list(float())) ->
    ok | {inject_bias, list(float()), list(integer())} | {veto_token, term()}.
evaluate(DraftText, Indices, Probabilities) ->
    %% Update Meta-Guard context to balanced general
    ok = iguana_meta_guard:update_context(general),
    %% Detect potential sentencing or parole bias indicators
    case detects_sentencing_bias(DraftText) of
        true ->
            %% Adjust probability distribution to rebalance options mathematically
            K = length(Indices),
            {inject_bias, [0.4 || _ <- lists:seq(1, K)], Indices};
        false ->
            iguana_entropy_guard:evaluate_entropy_sync(Indices, Probabilities)
    end.

detects_sentencing_bias(Text) ->
    string:find(Text, "sentencing rationale") =/= nomatch or
    string:find(Text, "parole recommendation") =/= nomatch.
