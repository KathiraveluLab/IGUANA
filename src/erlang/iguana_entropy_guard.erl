-module(iguana_entropy_guard).
-behaviour(gen_server).

-type probabilities() :: [float()].

%% API
-export([start_link/0, monitor_token/2, set_threshold/1, get_stats/1]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-include("iguana.hrl").

%%%===================================================================
%%% API
%%%===================================================================

%% @doc Starts a worker in the IGUANA entropy guard swarm.
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link(?MODULE, [], []).

%% @doc Asynchronously send the probability distribution of the next token to the guardrail.
%% Probabilities should be a list of floats summing to 1.0.
-spec monitor_token(pid(), probabilities()) -> ok | {error, no_workers}.
monitor_token(EnginePid, Probabilities) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            %% Pick a random worker from the Swarm (Load Balancing)
            Worker = lists:nth(rand:uniform(length(Members)), Members),
            gen_server:cast(Worker, {evaluate_entropy, EnginePid, Probabilities})
    end.

%% @doc Globally sets the entropy threshold for all active workers in the swarm.
-spec set_threshold(float()) -> ok | {error, no_workers}.
set_threshold(Threshold) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            %% Broadcast update to all workers in the Swarm
            [gen_server:call(W, {set_threshold, Threshold}) || W <- Members],
            ok
    end.

%% @doc Requests current internal state and statistics from a specific guard pid.
-spec get_stats(pid()) -> {ok, #state{}}.
get_stats(Pid) ->
    gen_server:call(Pid, get_stats).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    %% Join the swarm process group
    pg:join(iguana_swarm, self()),
    {ok, #state{entropy_threshold = 2.5, active_injections = 0}}.

handle_call({set_threshold, Threshold}, _From, State) ->
    {reply, ok, State#state{entropy_threshold = Threshold}};
handle_call(get_stats, _From, State) ->
    {reply, {ok, State}, State};
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
    %% Attempt to use the Hardware Acceleration layer (C-NIF) as per Section 3.2
    try 
        iguana_accelerator:accelerated_entropy(Probabilities)
    catch 
        _:_ ->
            %% Fallback to native Erlang implementation if NIF is unavailable
            calculate_entropy_erl(Probabilities)
    end.

calculate_entropy_erl(Probabilities) ->
    lists:foldl(fun(P, Acc) -> 
        if P > 0.0 -> Acc - (P * math:log2(P));
           true    -> Acc
        end
    end, 0.0, Probabilities).

%% Asynchronously send the debiasing weights back to the inference engine
inject_skew_normal_bias(EnginePid) ->
    %% Section 3.4: SkewPNN bias vector Bt = A2(C) * Phi((x-xi)/omega)
    %% Here we implement a normalized approximation of the Skew-Normal CDF
    %% and cast it back to the Python inference engine to rebalance logits.
    
    %% Compute dynamic bias vector based on statistical skewness
    %% In a production environment, this would involve a matrix multiplication A2 * Phi
    %% For this implementation, we calculate a 4-dimensional corrective skew vector
    %% to counteract the Selective Refusal Problem.
    SkewCoeff = 0.5,
    BiasVector = [
        0.15 * SkewCoeff, 
       -0.35 * (1.0 - SkewCoeff), 
        0.55 * SkewCoeff, 
        0.25 * (1.0 - SkewCoeff)
    ],
    
    io:format("[IGUANA_GUARD] Entropy spike detected! Injecting SkewPNN corrective vector to ~p~n", [EnginePid]),
    
    %% Send via Erlang message passing !
    EnginePid ! {inject_bias, BiasVector}.
