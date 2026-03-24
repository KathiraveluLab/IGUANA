-module(iguana_sup).
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

%% @doc Starts the top-level IGUANA supervisor under the OTP supervision tree.
start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

%% @doc Supervisor initialization. Defines the child specifications and restart strategy.
%%
%% Strategy: one_for_one — if iguana_entropy_guard crashes, only it is restarted,
%% leaving any future sibling guardrail actors unaffected. This maps directly to
%% Erlang's "let it crash" fault-isolation philosophy described in the paper.
%%
%% Intensity/Period: allow up to 5 restarts within 10 seconds before escalating
%% the failure to the parent application supervisor.
init([]) ->
    SupFlags = #{
        strategy  => one_for_one,
        intensity => 5,
        period    => 10
    },
    ChildSpecs = [
        #{
            id       => iguana_entropy_guard,
            start    => {iguana_entropy_guard, start_link, []},
            restart  => permanent,
            shutdown => 5000,
            type     => worker,
            modules  => [iguana_entropy_guard]
        }
    ],
    {ok, {SupFlags, ChildSpecs}}.
