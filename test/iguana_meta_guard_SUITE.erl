-module(iguana_meta_guard_SUITE).
-include_lib("common_test/include/ct.hrl").
-include("iguana.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([test_initial_state/1, test_domain_shift/1, test_unknown_domain/1, test_swarm_broadcast/1]).

all() ->
    [test_initial_state, test_domain_shift, test_unknown_domain, test_swarm_broadcast].

init_per_suite(Config) ->
    {ok, _} = application:ensure_all_started(iguana),
    Config.

end_per_suite(_Config) ->
    ok.

end_per_testcase(_Case, _Config) ->
    ok.

init_per_testcase(_Case, Config) ->
    Config.

test_initial_state(_Config) ->
    creative = iguana_meta_guard:get_current_domain().

test_domain_shift(_Config) ->
    {ok, GuardPid} = iguana_entropy_guard:start_link(),
    %% Explicitly join swarm so update_context updates it!
    pg:join(iguana_swarm, GuardPid),
    timer:sleep(50), %% Sync delay

    iguana_meta_guard:update_context(medical),
    %% Cast is asynchronous, wait a bit for processing
    timer:sleep(200),
    medical = iguana_meta_guard:get_current_domain(),

    %% Verify the guard received the threshold (medical => 1.8)
    {ok, State} = iguana_entropy_guard:get_stats(GuardPid),
    1.8 = State#state.entropy_threshold,

    gen_server:stop(GuardPid).

test_unknown_domain(_Config) ->
    %% Let's ensure domain is set to a known one first
    iguana_meta_guard:update_context(general),
    timer:sleep(200),

    iguana_meta_guard:update_context(non_existent_domain),
    timer:sleep(200),
    %% Should keep the old domain
    general = iguana_meta_guard:get_current_domain().

test_swarm_broadcast(_Config) ->
    {ok, G1} = iguana_entropy_guard:start_link(),
    {ok, G2} = iguana_entropy_guard:start_link(),
    pg:join(iguana_swarm, G1),
    pg:join(iguana_swarm, G2),
    timer:sleep(50), %% Sync delay

    iguana_meta_guard:update_context(finance),
    timer:sleep(200),

    {ok, S1} = iguana_entropy_guard:get_stats(G1),
    {ok, S2} = iguana_entropy_guard:get_stats(G2),
    case S1 of
        %% If state is a tuple, check element 2 (threshold)
        _ when is_tuple(S1) ->
            2.0 = element(2, S1),
            2.0 = element(2, S2)
    end,

    iguana_meta_guard:update_context(financial),
    timer:sleep(200),

    {ok, S3} = iguana_entropy_guard:get_stats(G1),
    {ok, S4} = iguana_entropy_guard:get_stats(G2),

    case S3 of
        _ when is_tuple(S3) ->
            2.2 = element(2, S3),
            2.2 = element(2, S4)
    end,

    gen_server:stop(G1),
    gen_server:stop(G2).
