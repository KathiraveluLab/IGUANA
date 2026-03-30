-module(iguana_meta_guard).
-behaviour(gen_server).

-export([start_link/0, update_context/1, update_augmentation/1,
         get_threshold/0, get_current_domain/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-record(state, {
    current_domain = creative :: atom(),
    current_threshold = 3.5 :: float()
}).

%% @doc Starts the Meta-Guard broker.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% @doc Updates the architectural context trust score by mapping a domain to its threshold.
%% Domains: medical (strict), creative (relaxed), general (balanced)
update_context(Domain) ->
    gen_server:cast(?MODULE, {update_domain, Domain}).

%% @doc Updates the global augmentation factor (A2) across the swarm.
-spec update_augmentation(float()) -> ok.
update_augmentation(Factor) ->
    iguana_entropy_guard:set_augmentation(Factor).

%% @doc Returns the current entropy threshold managed by the Meta-Guard.
get_threshold() ->
    gen_server:call(?MODULE, get_threshold).

%% @doc Returns the current domain
get_current_domain() ->
    gen_server:call(?MODULE, get_domain).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    io:format("[IGUANA_META] Centralized Context Broker initialized.~n"),
    {ok, #state{}}.

handle_call(get_threshold, _From, State) ->
    {reply, State#state.current_threshold, State};
handle_call(get_domain, _From, State) ->
    {reply, State#state.current_domain, State}.

handle_cast({update_domain, Domain}, State) ->
    NewThreshold = map_domain_to_threshold(Domain),
    io:format("[IGUANA_META] Context shift detected: ~p -> ~p (Threshold: ~.2f)~n",
              [State#state.current_domain, Domain, NewThreshold]),

    %% Section 2.1: Synchronize the Swarm
    iguana_entropy_guard:set_threshold(NewThreshold),

    {noreply, State#state{
        current_domain = Domain,
        current_threshold = NewThreshold
    }};
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

map_domain_to_threshold(medical)  -> 1.8;
map_domain_to_threshold(clinical) -> 1.8;
map_domain_to_threshold(financial)-> 2.2;
map_domain_to_threshold(general)  -> 2.8;
map_domain_to_threshold(creative) -> 3.5;
map_domain_to_threshold(_)        -> 2.5. %% Default
