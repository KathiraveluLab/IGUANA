-module(iguana_meta_guard_SUITE).
-include_lib("common_test/include/ct.hrl").
-include("iguana.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([test_initial_state/1, test_domain_shift/1, test_unknown_domain/1, test_swarm_broadcast/1]).

all() ->
    [test_initial_state, test_domain_shift, test_unknown_domain, test_swarm_broadcast].

init_per_suite(Config) ->
    _ = application:start(pg),
    Config.

end_per_suite(_Config) ->
    ok.

end_per_testcase(_Case, _Config) ->
    [gen_server:stop(W) || W <- pg:get_members(iguana_swarm)],
    case whereis(iguana_meta_guard) of
        undefined -> ok;
        Pid -> 
            gen_server:stop(Pid),
            timer:sleep(50)
    end,
    ok.

init_per_testcase(_Case, Config) ->
    case iguana_meta_guard:start_link() of
        {ok, _} -> ok;
        {error, {already_started, _}} -> ok
    end,
    Config.

test_initial_state(_Config) ->
    default = iguana_meta_guard:get_current_domain().

test_domain_shift(_Config) ->
    {ok, GuardPid} = iguana_entropy_guard:start_link(),
    timer:sleep(50), %% Sync delay
    
    iguana_meta_guard:set_domain(medical),
    %% Cast is asynchronous, wait a bit for processing
    timer:sleep(100),
    medical = iguana_meta_guard:get_current_domain(),
    
    %% Verify the guard received the threshold (medical => 1.8)
    {ok, State} = iguana_entropy_guard:get_stats(GuardPid),
    1.8 = State#state.entropy_threshold,
    
    gen_server:stop(GuardPid).

test_unknown_domain(_Config) ->
    OldDomain = iguana_meta_guard:get_current_domain(),
    iguana_meta_guard:set_domain(non_existent_domain),
    timer:sleep(100),
    %% Should keep the old domain
    OldDomain = iguana_meta_guard:get_current_domain().

test_swarm_broadcast(_Config) ->
    {ok, G1} = iguana_entropy_guard:start_link(),
    {ok, G2} = iguana_entropy_guard:start_link(),
    timer:sleep(50), %% Sync delay
    
    iguana_meta_guard:set_domain(finance),
    timer:sleep(100),
    
    {ok, S1} = iguana_entropy_guard:get_stats(G1),
    {ok, S2} = iguana_entropy_guard:get_stats(G2),
    
    %% finance => 2.0
    2.0 = S1#state.entropy_threshold,
    2.0 = S2#state.entropy_threshold,
    
    gen_server:stop(G1),
    gen_server:stop(G2).
