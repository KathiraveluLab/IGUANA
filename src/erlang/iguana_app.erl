-module(iguana_app).
-behaviour(application).

-export([start/2, stop/1]).

%% @doc OTP Application callback. Called by the BEAM when the iguana application
%% is started (e.g., via `application:start(iguana)` or automatically at boot).
%% Boots the top-level supervisor which in turn starts iguana_entropy_guard.
start(_StartType, _StartArgs) ->
    %% Start pg (Process Groups) scope for the swarm
    pg:start_link(),
    iguana_sup:start_link().

%% @doc OTP Application stop callback. Called when the application is terminated.
stop(_State) ->
    ok.
