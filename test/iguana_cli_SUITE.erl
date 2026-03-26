-module(iguana_cli_SUITE).
-include_lib("common_test/include/ct.hrl").
-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([test_main_usage/1, test_main_bench/1, test_main_sync/1]).

all() -> [test_main_usage, test_main_bench, test_main_sync].

init_per_suite(Config) ->
    application:ensure_all_started(iguana),
    Config.

end_per_suite(_Config) ->
    application:stop(iguana),
    ok.

test_main_usage(_Config) ->
    %% Check that it prints usage and returns ok (or whatever it returns)
    %% io:format output is hard to capture in CT without redirection, but we can verify it doesn't crash.
    iguana_cli:main([]),
    ok.

test_main_bench(_Config) ->
    %% benchmark:run() returns 'ok' (result of io:format)
    ok = iguana_cli:main(["bench"]),
    ok.

test_main_sync(_Config) ->
    %% os:cmd returns a string. We just ensure it doesn't crash.
    _Result = iguana_cli:main(["sync"]),
    ok.
