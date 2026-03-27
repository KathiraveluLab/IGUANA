-record(state, {
    entropy_threshold = 2.5 :: float(),
    vocab_size = 32000 :: integer(),
    active_injections = 0 :: integer(),
    augmentation_factor = 0.3 :: float() %% A2 (Adaptive Augmentation)
}).
