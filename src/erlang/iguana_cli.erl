-module(iguana_cli).
-export([main/1, bench/0, dash/0, sync/0]).

main(["bench"]) -> bench();
main(["dash"]) -> dash();
main(["sync"]) -> sync();
main(_) ->
    io:format("Usage: iguana_cli [bench | dash | sync]~n").

bench() ->
    benchmark:run().

dash() ->
    application:ensure_all_started(iguana),
    iguana_swarm_dashboard:display().

sync() ->
    %% Invoke the sync escript logic (linked or copied)
    io:format("Running LaTeX macro synchronization...~n"),
    os:cmd("./_paper/sync_results.erl").
