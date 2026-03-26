-module(iguana_meta_guard).
-behaviour(gen_server).

%% API
-export([start_link/0, set_domain/1, get_current_domain/0]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-type domain() :: default | medical | finance | creative | coding.
-type threshold() :: float().

-record(state, {
    current_domain = default :: domain(),
    domain_map = #{} :: #{domain() => threshold()}
}).

%%%===================================================================
%%% API
%%%===================================================================

%% @doc Starts the Meta Guard gen_server.
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% @doc Sets the conversation domain and broadcasts the new threshold to the swarm.
-spec set_domain(domain()) -> ok.
set_domain(Domain) when is_atom(Domain) ->
    gen_server:cast(?MODULE, {set_domain, Domain}).

%% @doc Retrieves the current active domain from the guard state.
-spec get_current_domain() -> domain().
get_current_domain() ->
    gen_server:call(?MODULE, get_current_domain).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    %% Define domain sensitivity map
    Map = #{
        default  => 2.5,
        medical  => 1.8,  % Ultra-sensitive
        finance  => 2.0,  % High sensitivity
        creative => 3.5,  % Flexible
        coding   => 3.0   % Moderate
    },
    {ok, #state{current_domain = default, domain_map = Map}}.

handle_call(get_current_domain, _From, State) ->
    {reply, State#state.current_domain, State};
handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_cast({set_domain, Domain}, State) ->
    case maps:find(Domain, State#state.domain_map) of
        {ok, NewThreshold} ->
            io:format("[IGUANA_META] Context shift detected: ~p. Broadcasting threshold ~p to swarm.~n", 
                      [Domain, NewThreshold]),
            broadcast_threshold(NewThreshold),
            {noreply, State#state{current_domain = Domain}};
        error ->
            io:format("[IGUANA_META] Unknown domain: ~p. Maintaining ~p.~n", 
                      [Domain, State#state.current_domain]),
            {noreply, State}
    end;
handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%%===================================================================
%%% Internal functions
%%%===================================================================

broadcast_threshold(Threshold) ->
    %% Leverage iguana_entropy_guard's broadcast API
    iguana_entropy_guard:set_threshold(Threshold).
