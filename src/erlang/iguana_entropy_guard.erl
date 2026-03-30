-module(iguana_entropy_guard).
-behaviour(gen_server).

-type probabilities() :: [float()].

%% API
-export([start_link/0, monitor_token/3, set_threshold/1,
         set_augmentation/1, get_stats/1, set_vocab_size/1]).
-export([calculate_entropy/2, skew_normal_cdf/2, owens_t/2]).

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
-spec monitor_token(pid(), [integer()], probabilities()) -> ok | {error, no_workers | overloaded}.
monitor_token(EnginePid, Indices, Probabilities) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            %% Select the least loaded worker to prevent mailbox hotspots
            %% We use a load-shedding threshold of 100 to protect node stability.
            MaxMailbox = 100,
            case find_best_worker(Members, {none, MaxMailbox}) of
                {ok, Worker} ->
                    gen_server:cast(Worker, {evaluate_entropy, EnginePid, Indices, Probabilities});
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

%% @doc Globally sets the augmentation factor (A2) for all active workers.
-spec set_augmentation(float()) -> ok | {error, no_workers}.
set_augmentation(Factor) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            [gen_server:call(W, {set_augmentation, Factor}) || W <- Members],
            ok
    end.

%% @doc Requests current internal state and statistics from a specific guard pid.
-spec get_stats(pid()) -> {ok, #state{}}.
get_stats(Pid) ->
    gen_server:call(Pid, get_stats).

%% @doc Globally sets the vocabulary size for the entropy approximation.
-spec set_vocab_size(integer()) -> ok | {error, no_workers}.
set_vocab_size(Size) ->
    case pg:get_members(iguana_swarm) of
        [] -> {error, no_workers};
        Members ->
            [gen_server:cast(M, {set_vocab_size, Size}) || M <- Members],
            ok
    end.

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    %% Join the swarm process group (Retry if needed)
    join_swarm(5),
    {ok, #state{}}.

join_swarm(N) when N > 0 ->
    try pg:join(iguana_swarm, self()) of
        ok -> ok
    catch
        _:_ ->
            timer:sleep(100),
            join_swarm(N-1)
    end;
join_swarm(0) ->
    io:format("[IGUANA_GUARD] CRITICAL: Failed to join swarm after retries.~n").

handle_call({set_threshold, Threshold}, _From, State) ->
    {reply, ok, State#state{entropy_threshold = Threshold}};
handle_call({set_augmentation, Factor}, _From, State) ->
    {reply, ok, State#state{augmentation_factor = Factor}};
handle_call(get_stats, _From, State) ->
    {reply, {ok, State}, State};
handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_cast({evaluate_entropy, EnginePid, Indices, Probabilities}, State) ->
    Entropy = calculate_entropy(Probabilities, State#state.vocab_size),
    if
        Entropy > State#state.entropy_threshold ->
            %% The model is statistically confused! (Entropy Spike)
            %% Action: Inject Skew-Normal Bias specifically for the Mode of the distribution
            inject_skew_normal_bias(EnginePid, Indices, State),
            {noreply, State#state{active_injections = State#state.active_injections + 1}};
        true ->
            %% Model is confident, do nothing
            {noreply, State}
    end;
handle_cast({set_threshold, Threshold}, State) ->
    %% Context Shift: Apply new domain-specific threshold from Meta-Guard
    {noreply, State#state{entropy_threshold = Threshold}};
handle_cast({set_augmentation, Factor}, State) ->
    {noreply, State#state{augmentation_factor = Factor}};
handle_cast({set_vocab_size, Size}, State) ->
    %% Architectural Initialization: Set model vocabulary size
    {noreply, State#state{vocab_size = Size}};
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
calculate_entropy(Probabilities, VocabSize) when length(Probabilities) > 0 ->
    %% The last element is the 'rest' probability mass
    {TopK, [Rest]} = lists:split(length(Probabilities) - 1, Probabilities),

    %% 1. Calculate entropy for the high-confidence Top-K mode
    TopKEntropy = try
        iguana_accelerator:accelerated_entropy(TopK)
    catch
        _:_ -> calculate_entropy_erl(TopK)
    end,

    %% 2. Approximate entropy for the long-tail 'Rest' mass
    %% We assume a uniform distribution over the remaining vocabulary
    K = length(TopK),
    RestCount = case VocabSize - K of
        Count when Count > 0 -> Count;
        _ -> 1 %% Safeguard for tiny vocabs
    end,

    RestEntropy = if
        Rest > 1.0e-9 ->
            %% Shannon entropy of uniform distribution: Rest * log2(RestCount / Rest)
            - Rest * (math:log2(Rest / RestCount));
        true -> 0.0
    end,

    TopKEntropy + RestEntropy;
calculate_entropy(_, _) -> 0.0.

calculate_entropy_erl(Probabilities) ->
    lists:foldl(fun(P, Acc) ->
        if P > 1.0e-9 -> Acc - (P * math:log2(P));
           true    -> Acc
        end
    end, 0.0, Probabilities).

%% Asynchronously send the debiasing weights back to the inference engine
inject_skew_normal_bias(EnginePid, Indices, State) ->
    %% Section 3.4/RQ2: SkewPNN bias vector Bt = A2(C) * Phi((x-xi)/omega)
    %% We generate a soft corrective vector sized exactly for the Top-K indices.

    K = length(Indices),
    Xi = K / 2.0,
    %% Scale factor (spread of the correction)
    Omega = K / 4.0,

    %% Generate the Bias Vector following the true Skew-Normal CDF curve
    %% Section 3.4: Bt = A2(C) * [Phi(z) - 2*T(z, alpha)]
    Alpha = 2.0, %% Shape parameter (skewness)
    A2 = State#state.augmentation_factor,
    BiasVector = [
        A2 * skew_normal_cdf((I - Xi) / Omega, Alpha)
        || I <- lists:seq(1, K)
    ],

    io:format("[IGUANA_GUARD] Entropy spike detected! "
              "Injecting Soft SkewPNN bias vector vs ~p tokens~n", [K]),

    %% Send via Erlang message passing: {inject_bias, BiasVector, Indices}
    EnginePid ! {inject_bias, BiasVector, Indices}.

%% @doc Skew-Normal Cumulative Distribution Function (CDF).
%% F(x; alpha) = Phi(x) - 2 * T(x, alpha)
-spec skew_normal_cdf(float(), float()) -> float().
skew_normal_cdf(X, Alpha) ->
    standard_normal_cdf(X) - 2.0 * owens_t(X, Alpha).

%% @doc Standard Normal Cumulative Distribution Function (CDF).
-spec standard_normal_cdf(float()) -> float().
standard_normal_cdf(Z) ->
    0.5 * (1.0 + math:erf(Z / math:sqrt(2.0))).

%% @doc Owen's T-function implementation using Simpson's Rule numerical integration.
%% T(h, a) = (1/2pi) * integral_0^a [exp(-0.5*h^2*(1+x^2)) / (1+x^2)] dx
-spec owens_t(float(), float()) -> float().
owens_t(H, A) ->
    Steps = 40,
    Delta = A / Steps,
    Sum = lists:foldl(fun(I, Acc) ->
        X = I * Delta,
        Weight = if (I == 0) or (I == Steps) -> 1;
                    (I rem 2 == 1) -> 4;
                    true -> 2
                 end,
        Acc + Weight * owens_integrand(X, H)
    end, 0.0, lists:seq(0, Steps)),
    (Delta / 3.0) * Sum / (2.0 * math:pi()).

owens_integrand(X, H) ->
    math:exp(-0.5 * H * H * (1.0 + X * X)) / (1.0 + X * X).
