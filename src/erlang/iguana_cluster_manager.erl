-module(iguana_cluster_manager).
-behaviour(gen_server).

%% API
-export([start_link/0, get_nodes/0]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-define(PING_INTERVAL, 5000). % 5 seconds

-record(state, {
    seed_nodes = [] :: [node()],
    active_nodes = [] :: [node()]
}).

%%%===================================================================
%%% API
%%%===================================================================

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

get_nodes() ->
    nodes().

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    %% Get seed nodes from application environment
    Seeds = application:get_env(iguana, nodes, []),
    
    %% Start monitoring nodes
    ok = net_kernel:monitor_nodes(true),
    
    %% Trigger first discovery
    self() ! discover,
    
    io:format("[IGUANA_CLUSTER] Cluster Manager initialized with seeds: ~p~n", [Seeds]),
    {ok, #state{seed_nodes = Seeds, active_nodes = nodes()}}.

handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(discover, State = #state{seed_nodes = Seeds}) ->
    %% Attempt to ping all seed nodes
    [net_adm:ping(Node) || Node <- Seeds, Node =/= node()],
    
    %% Schedule next discovery
    erlang:send_after(?PING_INTERVAL, self(), discover),
    {noreply, State#state{active_nodes = nodes()}};

handle_info({nodeup, Node}, State) ->
    io:format("[IGUANA_CLUSTER] Node joined: ~p Swarm size potentially increased.~n", [Node]),
    {noreply, State#state{active_nodes = nodes()}};

handle_info({nodedown, Node}, State) ->
    io:format("[IGUANA_CLUSTER] Node left: ~p Swarm size potentially decreased.~n", [Node]),
    {noreply, State#state{active_nodes = nodes()}};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.
