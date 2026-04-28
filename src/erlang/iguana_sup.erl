-module(iguana_sup).
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

%% @doc Starts the top-level IGUANA supervisor under the OTP supervision tree.
start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

%% @doc Supervisor initialization. Defines the child specifications and restart strategy.
init([]) ->
    SupFlags = #{
        strategy  => one_for_one,
        intensity => 5,
        period    => 10
    },
    %% Create a Swarm of 10 guardrail workers
    ChildSpecs = [
        #{
            id       => {iguana_entropy_guard, N},
            start    => {iguana_entropy_guard, start_link, [if N == 1 -> iguana_entropy_guard; true -> undefined end]},
            restart  => permanent,
            shutdown => 5000,
            type     => worker,
            modules  => [iguana_entropy_guard]
        }
        || N <- lists:seq(1, 10)
    ],
    MetaGuard = #{
        id       => iguana_meta_guard,
        start    => {iguana_meta_guard, start_link, []},
        restart  => permanent,
        shutdown => 5000,
        type     => worker,
        modules  => [iguana_meta_guard]
    },
    ClusterManager = #{
        id       => iguana_cluster_manager,
        start    => {iguana_cluster_manager, start_link, []},
        restart  => permanent,
        shutdown => 5000,
        type     => worker,
        modules  => [iguana_cluster_manager]
    },
    {ok, {SupFlags, [MetaGuard, ClusterManager | ChildSpecs]}}.
