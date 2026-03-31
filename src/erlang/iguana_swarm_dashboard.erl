-module(iguana_swarm_dashboard).
-export([display/0]).
-include("iguana.hrl").

display() ->
    Members = pg:get_members(iguana_swarm),
    CurrentDomain = case whereis(iguana_meta_guard) of
                        undefined -> unknown;
                        _ -> iguana_meta_guard:get_current_domain()
                    end,
    io:format("~n=====================================================~n"),
    io:format("        IGUANA SWARM REAL-TIME DASHBOARD             ~n"),
    io:format("=====================================================~n"),
    io:format(" Current Domain : ~p~n", [CurrentDomain]),
    io:format(" ~-15s | ~-10s | ~-15s ~n", ["Worker PID", "Threshold", "Active Injections"]),
    io:format("-----------------------------------------------------~n"),
    [print_worker_stats(M) || M <- Members],
    io:format("=====================================================~n"),
    io:format(" Total Workers: ~p~n", [length(Members)]),
    io:format(" System State : HEALTHY~n~n").

print_worker_stats(Pid) ->
    try iguana_entropy_guard:get_stats(Pid) of
        {ok, State} ->
            io:format(" ~p | ~p | ~p ~n",
                [Pid, State#state.entropy_threshold, State#state.active_injections])
    catch
        _:_ ->
            io:format(" ~-15p | ERROR      | ERROR           ~n", [Pid])
    end.
