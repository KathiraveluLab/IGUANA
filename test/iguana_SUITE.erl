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
    tc9_adaptive_augmentation/1,
    tc10_sync_evaluation/1
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
    tc9_adaptive_augmentation,
    tc10_sync_evaluation
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

wait_for_swarm_peer(Peer, Count, 0) ->
    ct:fail({swarm_bootstrap_failed, expected, Count, got, length(peer:call(Peer, pg, get_members, [iguana_swarm]))});
wait_for_swarm_peer(Peer, Count, N) ->
    case length(peer:call(Peer, pg, get_members, [iguana_swarm])) of
        Count -> ok;
        _ ->
            timer:sleep(100),
            wait_for_swarm_peer(Peer, Count, N-1)
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
    %% 0. Dynamically generate ssl_dist_temp.conf with absolute paths
    PrivDir = code:priv_dir(iguana),
    CertFile = filename:join([PrivDir, "ssl", "cert.pem"]),
    KeyFile = filename:join([PrivDir, "ssl", "key.pem"]),

    ConfigContent = io_lib:format(
        "[{server,\n"
        "  [{certfile, \"~ts\"},\n"
        "   {keyfile, \"~ts\"},\n"
        "   {secure_renegotiate, true},\n"
        "   {depth, 0},\n"
        "   {versions, ['tlsv1.2']},\n"
        "   {verify, verify_none}]},\n"
        " {client,\n"
        "  [{secure_renegotiate, true},\n"
        "   {depth, 0},\n"
        "   {verify, verify_none},\n"
        "   {versions, ['tlsv1.2']},\n"
        "   {server_name_indication, disable}]}].\n",
        [CertFile, KeyFile]
    ),

    SSLDistOptFile = filename:join([PrivDir, "ssl_dist_temp.conf"]),
    ok = file:write_file(SSLDistOptFile, ConfigContent),

    %% Start epmd if not running
    os:cmd("epmd -daemon"),

    Cookie = iguana_test_cookie,

    %% Temporarily clear ERL_FLAGS so the peer nodes don't inherit conflicting arguments
    OldErlFlags = os:getenv("ERL_FLAGS"),
    os:putenv("ERL_FLAGS", ""),

    %% Spawn PeerPrimary
    {ok, PeerPrimary, PrimaryNode} = peer:start_link(#{name => test_primary_peer,
                                                      host => "127.0.0.1",
                                                      longnames => true,
                                                      connection => standard_io,
                                                      args => [
                                                          "-proto_dist", "inet_tls",
                                                          "-ssl_dist_optfile", SSLDistOptFile,
                                                          "-setcookie", atom_to_list(Cookie)
                                                      ]}),

    %% Spawn PeerSecondary
    {ok, PeerSecondary, SecondaryNode} = peer:start_link(#{name => test_secondary_peer,
                                                         host => "127.0.0.1",
                                                         longnames => true,
                                                         connection => standard_io,
                                                         args => [
                                                             "-proto_dist", "inet_tls",
                                                             "-ssl_dist_optfile", SSLDistOptFile,
                                                             "-setcookie", atom_to_list(Cookie)
                                                         ]}),

    case OldErlFlags of
        false -> ok;
        _ -> os:putenv("ERL_FLAGS", OldErlFlags)
    end,

    %% Sync code path on both peers
    Path = [P || P <- code:get_path(), filelib:is_dir(P)],
    true = peer:call(PeerPrimary, code, set_path, [Path]),
    true = peer:call(PeerSecondary, code, set_path, [Path]),

    %% Connect PeerSecondary to PeerPrimary
    case peer:call(PeerSecondary, net_adm, ping, [PrimaryNode]) of
        pong ->
            ok;
        pang ->
            EpmdNames = os:cmd("epmd -names"),
            CertExists = filelib:is_file(CertFile),
            KeyExists = filelib:is_file(KeyFile),
            peer:call(PeerPrimary, application, ensure_all_started, [ssl]),
            peer:call(PeerSecondary, application, ensure_all_started, [ssl]),
            ServerOpts = [
                {certfile, CertFile},
                {keyfile, KeyFile},
                {secure_renegotiate, true},
                {depth, 0},
                {versions, ['tlsv1.2']},
                {verify, verify_none}
            ],
            ListenResult = peer:call(PeerPrimary, ssl, listen, [0, ServerOpts]),
            ManualTLSResult = case ListenResult of
                {ok, LSocket} ->
                    {ok, {_, LPort}} = peer:call(PeerPrimary, ssl, sockname, [LSocket]),
                    Self = self(),
                    spawn(fun() ->
                        AcceptRes = peer:call(PeerPrimary, ssl, transport_accept, [LSocket]),
                        case AcceptRes of
                            {ok, ASocket} ->
                                HandshakeRes = peer:call(PeerPrimary, ssl, handshake, [ASocket]),
                                Self ! {server_handshake, HandshakeRes};
                            AcceptErr ->
                                Self ! {server_accept, AcceptErr}
                        end
                    end),
                    timer:sleep(200),
                    ClientOpts = [
                        {secure_renegotiate, true},
                        {depth, 0},
                        {verify, verify_none},
                        {versions, ['tlsv1.2']},
                        {server_name_indication, disable}
                    ],
                    CRes = peer:call(PeerSecondary, ssl, connect, ["127.0.0.1", LPort, ClientOpts, 2000]),
                    SRes = receive
                        {server_handshake, ShRes} -> {server_handshake, ShRes};
                        {server_accept, SaRes} -> {server_accept, SaRes}
                    after 1000 ->
                        timeout
                    end,
                    {ok, LPort, CRes, SRes};
                ListenErr ->
                    {listen_err, ListenErr}
            end,
            ct:fail("Ping failed. Epmd: ~s, Cert: ~p, Key: ~p, ManualTLS: ~p",
                    [EpmdNames, CertExists, KeyExists, ManualTLSResult])
    end,
    timer:sleep(100),

    %% Start IGUANA on both peers
    {ok, _} = peer:call(PeerPrimary, application, ensure_all_started, [iguana]),
    {ok, _} = peer:call(PeerSecondary, application, ensure_all_started, [iguana]),

    %% Verify Swarm Membership (10 on Primary + 10 on Secondary = 20 total)
    wait_for_swarm_peer(PeerPrimary, 20, 100),

    %% Verify Threshold Propagation from Primary -> Secondary
    peer:call(PeerPrimary, iguana_meta_guard, update_context, [medical]),
    timer:sleep(500),

    %% Check a worker on the secondary node
    Members = peer:call(PeerPrimary, pg, get_members, [iguana_swarm]),
    SecondaryWorkers = [P || P <- Members, node(P) == SecondaryNode],
    case SecondaryWorkers of
        [SecondaryWorker | _] ->
            {ok, State} = peer:call(PeerSecondary, iguana_entropy_guard, get_stats, [SecondaryWorker]),
            %% Medical threshold should be 1.80
            1.8 = State#state.entropy_threshold;
        [] ->
            ct:fail("No workers found on secondary node")
    end,

    %% Clean up
    peer:stop(PeerPrimary),
    peer:stop(PeerSecondary),
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

%% TC10: Asserts synchronous block-mode token evaluation (veto, accept, inject bias).
tc10_sync_evaluation(_Config) ->
    %% Set threshold to 2.0 (veto threshold = 1.0)
    iguana_entropy_guard:set_threshold(2.0),
    timer:sleep(100),

    %% Case 1: Low entropy (H(P) = 0.286 < 1.0) -> Expect Veto
    ProbsVeto = [0.95, 0.05, 0.0, 0.0, 0.0],
    {veto_token, low_entropy} = iguana_entropy_guard:evaluate_entropy_sync([1,2,3,4], ProbsVeto),

    %% Case 2: Mid entropy (1.0 <= H(P) = 1.58 <= 2.0) -> Expect Accept (ok)
    ProbsAccept = [0.33, 0.33, 0.34, 0.0, 0.0],
    ok = iguana_entropy_guard:evaluate_entropy_sync([1,2,3,4], ProbsAccept),

    %% Case 3: High entropy (H(P) = 3.0 > 2.0) -> Expect Bias Injection
    ProbsBias = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
    {inject_bias, BiasVector, [1,2,3,4,5,6,7,8]} =
        iguana_entropy_guard:evaluate_entropy_sync([1,2,3,4,5,6,7,8], ProbsBias),
    8 = length(BiasVector),
    ok.
