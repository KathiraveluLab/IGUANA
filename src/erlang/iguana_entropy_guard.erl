-module(iguana_entropy_guard).
-behaviour(gen_server).

%% API
-export([start_link/0, monitor_token/2, set_threshold/1]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {
    entropy_threshold = 2.5 :: float(),
    active_injections = 0 :: integer()
}).

%%%===================================================================
%%% API
%%%===================================================================

start_link() ->
    gen_server:start_link(?MODULE, [], []).

%% @doc Asynchronously send the probability distribution of the next token to the guardrail
%% Probabilities should be a list of floats summing to 1.0.
monitor_token(EnginePid, Probabilities) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            %% Pick a random worker from the Swarm (Load Balancing)
            Worker = lists:nth(rand:uniform(length(Members)), Members),
            gen_server:cast(Worker, {evaluate_entropy, EnginePid, Probabilities})
    end.

set_threshold(Threshold) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            %% Broadcast update to all workers in the Swarm
            [gen_server:call(W, {set_threshold, Threshold}) || W <- Members],
            ok
    end.

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    %% Join the swarm process group
    pg:join(iguana_swarm, self()),
    {ok, #state{entropy_threshold = 2.5, active_injections = 0}}.

handle_call({set_threshold, Threshold}, _From, State) ->
    {reply, ok, State#state{entropy_threshold = Threshold}};
handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_cast({evaluate_entropy, EnginePid, Probabilities}, State) ->
    Entropy = calculate_entropy(Probabilities),
    if 
        Entropy > State#state.entropy_threshold ->
            %% The model is statistically confused! (Entropy Spike)
            %% Action: Inject Skew-Normal Bias to debias the generation
            inject_skew_normal_bias(EnginePid),
            {noreply, State#state{active_injections = State#state.active_injections + 1}};
        true ->
            %% Model is confident, do nothing (fire and forget)
            {noreply, State}
    end;
handle_cast({set_trust_threshold, Threshold}, State) ->
    io:format("[IGUANA_GUARD] Context shifted. Adapting entropy threshold to ~p~n", [Threshold]),
    {noreply, State#state{entropy_threshold = Threshold}};
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

%% Calculate Shannon entropy: -Sum(p * log2(p))
calculate_entropy(Probabilities) ->
    lists:foldl(fun(P, Acc) -> 
        if P > 0.0 -> Acc - (P * math:log2(P));
           true    -> Acc
        end
    end, 0.0, Probabilities).

%% Asynchronously send the debiasing weights back to the inference engine
inject_skew_normal_bias(EnginePid) ->
    %% Simulate the calculation of the A2 / SkewPNN bias matrix
    io:format("[IGUANA_GUARD] Entropy spike detected! Injecting Skew-Normal bias matrix to ~p~n", [EnginePid]),
    
    %% In a real integration, this would be a pointer to shared memory (NIF)
    %% or a highly compressed concept vector. For now, we simulate the payload.
    BiasWeights = [0.1, -0.4, 0.5, 0.2], 
    
    %% Send via Erlang message passing !
    EnginePid ! {inject_bias, BiasWeights}.
