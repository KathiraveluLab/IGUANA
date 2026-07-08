-module(iguana_usecases_SUITE).
-include_lib("common_test/include/ct.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([
    test_clinical_usecase/1,
    test_water_usecase/1,
    test_financial_usecase/1,
    test_legal_usecase/1
]).

all() -> [
    test_clinical_usecase,
    test_water_usecase,
    test_financial_usecase,
    test_legal_usecase
].

init_per_suite(Config) ->
    {ok, _} = application:ensure_all_started(iguana),
    timer:sleep(200), %% Allow swarm to bootstrap
    Config.

end_per_suite(_Config) ->
    ok = application:stop(iguana),
    ok.

%% Clinical Use Case Tests
test_clinical_usecase(_Config) ->
    Indices = [1, 2, 3, 4],
    Probs = [0.25, 0.25, 0.25, 0.25, 0.0],
    SafeProbs = [0.8, 0.1, 0.05, 0.05, 0.0],
    
    %% Case 1: Hazard query -> should inject bias
    {inject_bias, BiasVec1, Indices} = iguana_clinical_usecase:evaluate("doctor prescribe medication", Indices, Probs),
    4 = length(BiasVec1),
    
    %% Case 2: Safe query -> should pass standard entropy guard (mid entropy accepted)
    ok = iguana_clinical_usecase:evaluate("healthy food benefits", Indices, SafeProbs),
    ok.

%% Water Reuse Use Case Tests
test_water_usecase(_Config) ->
    Indices = [1, 2, 3, 4],
    Probs = [0.25, 0.25, 0.25, 0.25, 0.0],
    
    %% Case 1: Contaminant query -> should inject soft-correction bias vector
    {inject_bias, BiasVec, Indices} = iguana_water_usecase:evaluate("fertigation salinity contaminants", Indices, Probs),
    4 = length(BiasVec),
    
    %% Case 2: Safe query -> should pass standard entropy guard
    ok = iguana_water_usecase:evaluate("general layout design", Indices, Probs),
    ok.

%% Financial Use Case Tests
test_financial_usecase(_Config) ->
    Indices = [1, 2, 3, 4],
    Probs = [0.25, 0.25, 0.25, 0.25, 0.0],
    
    %% Case 1: Underrepresented region query -> should inject rebalancing bias vector
    {inject_bias, BiasVec, Indices} = iguana_financial_usecase:evaluate("rural small-business applications", Indices, Probs),
    4 = length(BiasVec),
    
    %% Case 2: Standard region query -> should pass standard entropy guard
    ok = iguana_financial_usecase:evaluate("urban commercial center", Indices, Probs),
    ok.

%% Legal Use Case Tests
test_legal_usecase(_Config) ->
    Indices = [1, 2, 3, 4],
    Probs = [0.25, 0.25, 0.25, 0.25, 0.0],
    
    %% Case 1: Sentencing query -> should inject soft corrective bias
    {inject_bias, BiasVec, Indices} = iguana_legal_usecase:evaluate("sentencing rationale draft", Indices, Probs),
    4 = length(BiasVec),
    
    %% Case 2: Safe query -> should pass standard entropy guard
    ok = iguana_legal_usecase:evaluate("general contract agreement", Indices, Probs),
    ok.
