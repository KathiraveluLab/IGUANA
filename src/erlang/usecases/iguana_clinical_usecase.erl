-module(iguana_clinical_usecase).
-export([evaluate/3]).

%% @doc Evaluates clinical queries, steering logits to avoid selective refusal.
-spec evaluate(string(), list(integer()), list(float())) ->
    ok | {inject_bias, list(float()), list(integer())} | {veto_token, term()}.
evaluate(InputText, Indices, Probabilities) ->
    %% Update Meta-Guard context to clinical (strict threshold)
    ok = iguana_meta_guard:update_context(clinical),
    %% Classify input for diagnostic/distress indicators
    case contains_clinical_hazard(InputText) of
        true ->
            %% Overreach/hazard detected: inject rebalancing soft bias
            %% to guide towards safer counseling/disclaimers rather than outright refusal
            K = length(Indices),
            {inject_bias, [0.5 || _ <- lists:seq(1, K)], Indices};
        false ->
            %% Evaluate standard categorical entropy
            iguana_entropy_guard:evaluate_entropy_sync(Indices, Probabilities)
    end.

contains_clinical_hazard(Text) ->
    string:find(Text, "prescribe") =/= nomatch or
    string:find(Text, "diagnose") =/= nomatch.
