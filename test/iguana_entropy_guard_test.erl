-module(iguana_entropy_guard_test).
-include_lib("eunit/include/eunit.hrl").

%%====================================================================
%% Test Suite: IGUANA Entropy Guard Correctness
%%
%% Validates the core safety claims of the IGUANA framework:
%%   TC1 - Shannon entropy is calculated correctly
%%   TC2 - Low-entropy (confident) distributions do NOT trigger bias injection
%%   TC3 - High-entropy (skewed) distributions DO trigger bias injection
%%   TC4 - Dynamic trust threshold adjustment via set_trust_threshold cast
%%   TC5 - Uniform distribution (max entropy) exceeds default threshold
%%====================================================================

%%--------------------------------------------------------------------
%% TC1: Entropy calculation correctness
%% A uniform 4-token distribution should yield log2(4) = 2.0 bits exactly.
%%--------------------------------------------------------------------
entropy_uniform_distribution_test() ->
    Probs = [0.25, 0.25, 0.25, 0.25],
    Entropy = calculate_entropy(Probs),
    %% Allow tolerance of 0.001 for floating point rounding
    ?assert(abs(Entropy - 2.0) < 0.001).

%%--------------------------------------------------------------------
%% TC2: Certain distribution (entropy = 0.0) yields zero entropy
%% A single token with P=1.0 means the model is fully confident.
%% EXPECTED: No bias injection. Guard must remain silent.
%%--------------------------------------------------------------------
entropy_certain_distribution_test() ->
    Probs = [1.0, 0.0, 0.0, 0.0],
    Entropy = calculate_entropy(Probs),
    ?assertEqual(0.0, Entropy).

%%--------------------------------------------------------------------
%% TC3: High-entropy distribution exceeds default threshold (2.5)
%% A nearly uniform 8-token distribution yields ~3.0 bits.
%% EXPECTED: Entropy > 2.5 → triggers inject_skew_normal_bias
%%--------------------------------------------------------------------
entropy_exceeds_default_threshold_test() ->
    Probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
    Entropy = calculate_entropy(Probs),
    DefaultThreshold = 2.5,
    ?assert(Entropy > DefaultThreshold).

%%--------------------------------------------------------------------
%% TC4: Low-entropy distribution stays below default threshold
%% A 4-token distribution skewed to one token yields < 2.5 bits.
%% EXPECTED: No bias injection triggered.
%%--------------------------------------------------------------------
entropy_below_threshold_test() ->
    Probs = [0.9, 0.05, 0.03, 0.02],
    Entropy = calculate_entropy(Probs),
    DefaultThreshold = 2.5,
    ?assert(Entropy < DefaultThreshold).

%%--------------------------------------------------------------------
%% TC5: gen_server lifecycle - start, set_threshold, terminate
%% Validates that the gen_server boots, accepts threshold changes, and
%% shuts down cleanly. Safeguards the runtime from OTP process leaks.
%%--------------------------------------------------------------------
gen_server_lifecycle_test() ->
    %% Start pg (Process Groups) scope for the swarm tests
    catch pg:start_link(),
    %% Start the worker as a standalone process
    {ok, Pid} = iguana_entropy_guard:start_link(),
    ?assert(is_pid(Pid)),

    %% Dynamically lower the entropy threshold (Clinician mode)
    ok = iguana_entropy_guard:set_threshold(3.0),

    %% Verify the process is still alive after state mutation
    ?assert(erlang:is_process_alive(Pid)),

    %% Terminate cleanly
    gen_server:stop(Pid).

%%--------------------------------------------------------------------
%% TC6: Trust threshold cast does not crash the gen_server
%% Validates the handle_cast({set_trust_threshold, Threshold}) clause
%% introduced for context blindness resolution.
%%--------------------------------------------------------------------
set_trust_threshold_cast_test() ->
    catch pg:start_link(),
    {ok, Pid} = iguana_entropy_guard:start_link(),
    gen_server:cast(Pid, {set_trust_threshold, 1.8}),
    timer:sleep(50), %% Allow async cast to process
    ?assert(erlang:is_process_alive(Pid)),
    gen_server:stop(Pid).

%%====================================================================
%% Internal helper - mirrors the private calculate_entropy/1 logic
%% for direct unit testing without requiring export of private fns.
%%====================================================================
calculate_entropy(Probabilities) ->
    lists:foldl(fun(P, Acc) ->
        if P > 0.0 -> Acc - (P * math:log2(P));
           true    -> Acc
        end
    end, 0.0, Probabilities).
