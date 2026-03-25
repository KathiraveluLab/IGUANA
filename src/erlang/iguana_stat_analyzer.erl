-module(iguana_stat_analyzer).
-export([analyze/1, confidence_interval/2, report/1]).

%% @doc Analyzes a list of latencies and returns {Mean, StdDev, Variance}
analyze(Latencies) when is_list(Latencies), length(Latencies) > 0 ->
    N = length(Latencies),
    Mean = lists:sum(Latencies) / N,
    Variance = lists:sum([(L - Mean) * (L - Mean) || L <- Latencies]) / N,
    StdDev = math:sqrt(Variance),
    {Mean, StdDev, Variance}.

%% @doc Calculates the 95% confidence interval for the mean
confidence_interval(Mean, StdDev) ->
    %% Using Z=1.96 for 95% confidence
    Z = 1.96,
    MarginOfError = Z * StdDev,
    {Mean - MarginOfError, Mean + MarginOfError}.

%% @doc Generates a distribution report for benchmark validation
report(Latencies) ->
    {Mean, StdDev, _} = analyze(Latencies),
    {Low, High} = confidence_interval(Mean, StdDev),
    io:format("--- Statistical Profile ---~n"),
    io:format("Sample Size : ~p~n", [length(Latencies)]),
    io:format("Mean Latency: ~.2f ms~n", [Mean]),
    io:format("Std Dev     : ~.2f ms~n", [StdDev]),
    io:format("95% CI      : [~.2f, ~.2f]~n", [Low, High]).
