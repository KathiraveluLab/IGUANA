-module(iguana_SUITE).
-include_lib("common_test/include/ct.hrl").
-include("iguana.hrl").
-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([test_swarm_scaling/1, test_bias_injection/1, test_entropy_distribution/1]).

all() -> [test_swarm_scaling, test_bias_injection, test_entropy_distribution].

init_per_suite(Config) ->
    pg:start_link(),
    application:ensure_all_started(iguana),
    timer:sleep(500), % Wait for 10 workers to join the pg swarm
    Config.

end_per_suite(_Config) ->
    application:stop(iguana),
    ok.

test_swarm_scaling(_Config) ->
    Members = pg:get_members(iguana_swarm),
    10 = length(Members),
    ok.

test_bias_injection(_Config) ->
    %% Simulate high entropy tokens and monitor self (acting as engine)
    Self = self(),
    iguana_entropy_guard:monitor_token(Self, [0.25, 0.25, 0.25, 0.25]),
    receive
        {inject_bias, Weights} ->
            true = is_list(Weights),
            ok
    after 1000 ->
        ct:fail(timeout_waiting_for_bias)
    end.

test_entropy_distribution(_Config) ->
    %% Validates that different workers in the swarm can process tokens independently
    Members = pg:get_members(iguana_swarm),
    [iguana_entropy_guard:monitor_token(self(), [0.1, 0.9]) || _ <- lists:seq(1, 100)],
    %% Check that no injections were triggered (low entropy)
    lists:foreach(fun(Pid) ->
        {ok, State} = iguana_entropy_guard:get_stats(Pid),
        0 = State#state.active_injections
    end, Members),
    ok.
