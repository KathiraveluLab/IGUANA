#include <erl_nif.h>
#include <math.h>
#include <stdint.h>

/**
 * High-Performance SIMD-optimized Shannon Entropy calculation.
 * Processes a contiguous binary buffer of double-precision floats.
 * Aligned for hardware vectorization (AVX/NEON).
 */

static ERL_NIF_TERM calculate_entropy_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin;
    
    // Expects a Binary of double-precision floats (64-bit)
    if (!enif_inspect_binary(env, argv[0], &bin)) {
        return enif_make_badarg(env);
    }

    // Verify length is a multiple of double size
    if (bin.size % sizeof(double) != 0) {
        return enif_make_badarg(env);
    }

    size_t n = bin.size / sizeof(double);
    const double* data = (const double*)bin.data;
    double total_entropy = 0.0;

    // Hint to compiler for alignment and vectorization
    const double* __restrict__ p_data = (const double*)__builtin_assume_aligned(data, 16);

    /**
     * SIMD-Ready Loop
     * We use a portable approach to log calculation to avoid platform-specific
     * vector math library dependencies (like libmvec) while remaining
     * eligible for loop unrolling and autovectorization.
     */
    const double inv_log2 = 1.4426950408889634; // 1/log(2)
    for (size_t i = 0; i < n; i++) {
        double p = p_data[i];
        if (p > 1e-9) { 
            total_entropy -= p * (log(p) * inv_log2);
        }
    }

    return enif_make_double(env, total_entropy);
}

static ErlNifFunc nif_funcs[] = {
    {"calculate_entropy_nif", 1, calculate_entropy_nif}
};

ERL_NIF_INIT(iguana_accelerator, nif_funcs, NULL, NULL, NULL, NULL)
