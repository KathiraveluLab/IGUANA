-module(iguana_SUITE).
-include_lib("common_test/include/ct.hrl").
-include("iguana.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([
    tc1_uniform_entropy/1,
    tc2_deterministic_entropy/1,
    tc3_veto_violation/1,
    tc4_accepted_generation/1,
    tc5_process_lifecycle/1,
    tc6_state_mutation/1,
    tc7_mathematical_purity/1,
    tc8_distributed_handshake/1,
    tc9_adaptive_augmentation/1
]).

all() -> [
    tc1_uniform_entropy,
    tc2_deterministic_entropy,
    tc3_veto_violation,
    tc4_accepted_generation,
    tc5_process_lifecycle,
    tc6_state_mutation,
    tc7_mathematical_purity,
    tc8_distributed_handshake,
    tc9_adaptive_augmentation
].

init_per_suite(Config) ->
    application:ensure_all_started(iguana),
    wait_for_swarm(10, 50), % Wait up to 5s for all 10 workers
    Config.

wait_for_swarm(Count, 0) ->
    ct:fail({swarm_bootstrap_failed, expected, Count, got, length(pg:get_members(iguana_swarm))});
wait_for_swarm(Count, N) ->
    case length(pg:get_members(iguana_swarm)) of
        Count -> ok;
        _ ->
            timer:sleep(100),
            wait_for_swarm(Count, N-1)
    end.

end_per_suite(_Config) ->
    application:stop(iguana),
    ok.

%% TC1: Asserts Shannon entropy of a statistically uniform categorical distribution.
tc1_uniform_entropy(_Config) ->
    %% Uniform over 4 tokens: -4 * (0.25 * log2(0.25)) = 2.0
    %% Our Top-K payload adds a 0.0 'Rest' mass
    Probs = [0.25, 0.25, 0.25, 0.25, 0.0],
    %% Use the guard's internal calculation (32000 vocab)
    Entropy = iguana_entropy_guard:calculate_entropy(Probs, 32000),
    true = (Entropy > 1.99) and (Entropy < 2.01),
    ok.

%% TC2: Asserts H(P) = 0.0 for a completely deterministic logit distribution.
tc2_deterministic_entropy(_Config) ->
    Probs = [1.0, 0.0, 0.0, 0.0, 0.0],
    Entropy = iguana_entropy_guard:calculate_entropy(Probs, 32000),
    0.0 = Entropy,
    ok.

%% TC3: Asserts veto boundary crossed when H(P) >= tau_v.
tc3_veto_violation(_Config) ->
    %% Set a very strict threshold
    iguana_entropy_guard:set_threshold(0.5),
    Self = self(),
    Indices = [1, 2, 3, 4],
    Probs = [0.25, 0.25, 0.25, 0.25, 0.0],
    iguana_entropy_guard:monitor_token(Self, Indices, Probs),
    receive
        {inject_bias, _Weights, _Indices} -> ok;
        {veto_token, _} -> ok
    after 1000 ->
        ct:fail(no_guardrail_action_triggered)
    end.

%% TC4: Asserts generation accepted when H(P) < tau_v.
tc4_accepted_generation(_Config) ->
    %% Relaxed threshold
    iguana_entropy_guard:set_threshold(5.0),
    Self = self(),
    Probs = [0.9, 0.05, 0.02, 0.03, 0.0], %% Low entropy
    iguana_entropy_guard:monitor_token(Self, [1,2,3,4], Probs),
    receive
        Any -> ct:fail({unexpected_message, Any})
    after 500 ->
        ok
    end.

%% TC5: Asserts gen_server initialization and synchronous state queries.
tc5_process_lifecycle(_Config) ->
    [Worker | _] = pg:get_members(iguana_swarm),
    {ok, State} = iguana_entropy_guard:get_stats(Worker),
    true = is_record(State, state),
    ok.

%% TC6: Asserts trust_score/domain correctly updates the boundary.
tc6_state_mutation(_Config) ->
    %% Through Meta-Guard
    iguana_meta_guard:update_context(medical),
    timer:sleep(100),
    [Worker | _] = pg:get_members(iguana_swarm),
    {ok, State} = iguana_entropy_guard:get_stats(Worker),
    1.8 = State#state.entropy_threshold,

    %% Through Domain switch
    iguana_meta_guard:update_context(creative),
    timer:sleep(100),
    {ok, State2} = iguana_entropy_guard:get_stats(Worker),
    3.5 = State2#state.entropy_threshold,
    ok.

%% TC7: Asserts Owen's T-function and Skew-Normal CDF mathematical purity.
tc7_mathematical_purity(_Config) ->
    %% F(1.0, 2.0) should be ~0.684 based on our Simpson's Rule implementation
    V = iguana_entropy_guard:skew_normal_cdf(1.0, 2.0),
    true = (V > 0.68) and (V < 0.69),

    %% T(0, 1) = 0.125 (Integration of 1/(2*pi*(1+x^2)) from 0 to 1)
    T01 = iguana_entropy_guard:owens_t(0.0, 1.0),
    true = (T01 > 0.12) and (T01 < 0.13),
    ok.

%% TC8: Asserts Distributed Swarm Handshake and Threshold Propagation
tc8_distributed_handshake(_Config) ->
    %% 1. Ensure the current node is named for distribution
    case node() of
        nonode@nohost ->
            os:cmd("epmd -daemon"),
            net_kernel:start([master, shortnames]);
        _ -> ok
    end,

    %% Set a deterministic cookie for the test
    Cookie = iguana_test_cookie,
    erlang:set_cookie(node(), Cookie),

    %% 2. Spawn a Peer Node using standard_io for maximum robustness
    {ok, Peer, SlaveNode} = peer:start_link(#{name => slave,
                                            connection => standard_io,
                                            args => ["-setcookie", atom_to_list(Cookie)]}),

    %% 3. Sync code path and start IGUANA on Peer
    %% Filter paths to ensure only existing absolute directories are sent
    Path = [P || P <- code:get_path(), filelib:is_dir(P)],
    true = peer:call(Peer, code, set_path, [Path]),

    %% Ensure the slave node can see the master node
    pong = peer:call(Peer, net_adm, ping, [node()]),
    timer:sleep(100),

    {ok, _} = peer:call(Peer, application, ensure_all_started, [iguana]),

    %% 4. Verify Swarm Membership (10 Local + 10 Remote = 20 total)
    wait_for_swarm(20, 100),

    %% 5. Verify Threshold Propagation from Master -> Slave
    iguana_meta_guard:update_context(medical),
    timer:sleep(500),

    %% Check a worker on the slave node
    Members = pg:get_members(iguana_swarm),
    SlaveWorkers = [P || P <- Members, node(P) == SlaveNode],
    case SlaveWorkers of
        [SlaveWorker | _] ->
            {ok, State} = rpc:call(SlaveNode, iguana_entropy_guard, get_stats, [SlaveWorker]),
            %% Medical threshold should be 1.80
            1.8 = State#state.entropy_threshold;
        [] ->
            ct:fail("No workers found on slave node")
    end,

    %% Clean up
    peer:stop(Peer),
    ok.

tc9_adaptive_augmentation(_Config) ->
    %% 1. Initial State Check (Default A2 = 0.3)
    Members = pg:get_members(iguana_swarm),
    [Worker | _] = Members,
    {ok, State1} = iguana_entropy_guard:get_stats(Worker),
    0.3 = State1#state.augmentation_factor,

    %% 2. Update to High Augmentation (Context Shift)
    iguana_meta_guard:update_augmentation(0.85),
    timer:sleep(200),

    {ok, State2} = iguana_entropy_guard:get_stats(Worker),
    0.85 = State2#state.augmentation_factor,
    ok.
