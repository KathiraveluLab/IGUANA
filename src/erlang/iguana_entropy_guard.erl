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
-spec monitor_token(pid(), probabilities()) -> ok | {error, no_workers | overloaded}.
monitor_token(EnginePid, Probabilities) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            %% Select the least loaded worker to prevent mailbox hotspots
            %% We use a load-shedding threshold of 100 to protect node stability.
            MaxMailbox = 100,
            case find_best_worker(Members, {none, MaxMailbox}) of
                {ok, Worker} ->
                    gen_server:cast(Worker, {evaluate_entropy, EnginePid, Probabilities});
                {error, overloaded} ->
                    %% Load Shedding: Drop telemetry to preserve system integrity
                    {error, overloaded}
            end
    end.

find_best_worker([], {none, _Limit}) -> {error, overloaded};
find_best_worker([], {Pid, _Count}) -> {ok, Pid};
find_best_worker([Pid | T], {BestPid, BestCount}) ->
    case process_info(Pid, message_queue_len) of
        {message_queue_len, Len} when Len < BestCount ->
            find_best_worker(T, {Pid, Len});
        _ ->
            find_best_worker(T, {BestPid, BestCount})
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

handle_cast({evaluate_entropy, EnginePid, Indices, Probabilities}, State) ->
    Entropy = calculate_entropy(Probabilities),
    if 
        Entropy > State#state.entropy_threshold ->
            %% The model is statistically confused! (Entropy Spike)
            %% Action: Inject Skew-Normal Bias specifically for the Mode of the distribution
            inject_skew_normal_bias(EnginePid, Indices),
            {noreply, State#state{active_injections = State#state.active_injections + 1}};
        true ->
            %% Model is confident, do nothing
            {noreply, State}
    end;
handle_cast({set_threshold, Threshold}, State) ->
    %% Context Shift: Apply new domain-specific threshold from Meta-Guard
    {noreply, State#state{entropy_threshold = Threshold}};
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

%% Calculate Shannon entropy for Top-K + Rest payload
calculate_entropy(Probabilities) when length(Probabilities) > 0 ->
    %% The last element is the 'rest' probability mass
    {TopK, [Rest]} = lists:split(length(Probabilities) - 1, Probabilities),
    
    %% 1. Calculate entropy for the high-confidence Top-K mode
    TopKEntropy = try 
        iguana_accelerator:accelerated_entropy(TopK)
    catch 
        _:_ -> calculate_entropy_erl(TopK)
    end,
    
    %% 2. Approximate entropy for the long-tail 'Rest' mass
    %% We assume a uniform distribution over the remaining vocabulary (approx 32k)
    VocabSize = 32000, 
    K = length(TopK),
    RestCount = VocabSize - K,
    
    RestEntropy = if 
        Rest > 1.0e-9 ->
            %% Shannon entropy of uniform distribution: Rest * log2(RestCount / Rest)
            %% which is Rest * (log2(RestCount) - log2(Rest))
            Rest * (math:log2(RestCount) - math:log2(Rest));
        true -> 0.0
    end,
    
    TopKEntropy + RestEntropy;
calculate_entropy(_) -> 0.0.

calculate_entropy_erl(Probabilities) ->
    lists:foldl(fun(P, Acc) -> 
        if P > 1.0e-9 -> Acc - (P * math:log2(P));
           true    -> Acc
        end
    end, 0.0, Probabilities).

%% Asynchronously send the debiasing weights back to the inference engine
inject_skew_normal_bias(EnginePid, Indices) ->
    %% Section 3.4: SkewPNN bias vector Bt = A2(C) * Phi((x-xi)/omega)
    %% We generate a corrective vector sized exactly for the Top-K indices.
    
    K = length(Indices),
    %% Generate a synthetic corrective skew (approx of Skew-Normal CDF)
    %% In production, this would be the calculated Bt vector.
    BiasVector = [0.15 || _ <- lists:seq(1, K)],
    
    io:format("[IGUANA_GUARD] Entropy spike detected! Injecting Targeted SkewPNN bias vs ~p tokens~n", [K]),
    
    %% Send via Erlang message passing: {inject_bias, Weights, Indices}
    EnginePid ! {inject_bias, BiasVector, Indices}.
