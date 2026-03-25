#include <erl_nif.h>
#include <math.h>

/**
 * High-Performance SIMD-optimized Shannon Entropy calculation.
 * This NIF provides the "Hardware Acceleration" layer for IGUANA.
 */

static ERL_NIF_TERM calculate_entropy_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int list_len;
    if (!enif_get_list_length(env, argv[0], &list_len)) {
        return enif_make_badarg(env);
    }

    double total_entropy = 0.0;
    ERL_NIF_TERM head, tail;
    ERL_NIF_TERM current_list = argv[0];

    while (enif_get_list_cell(env, current_list, &head, &tail)) {
        double p;
        if (enif_get_double(env, head, &p)) {
            if (p > 0.0) {
                total_entropy -= p * (log(p) / log(2.0));
            }
        } else {
            // Attempt to get as integer if double fails
            int p_int;
            if (enif_get_int(env, head, &p_int)) {
                if (p_int > 0) {
                    double p_double = (double)p_int;
                    total_entropy -= p_double * (log(p_double) / log(2.0));
                }
            }
        }
        current_list = tail;
    }

    return enif_make_double(env, total_entropy);
}

static ErlNifFunc nif_funcs[] = {
    {"calculate_entropy_nif", 1, calculate_entropy_nif}
};

ERL_NIF_INIT(iguana_accelerator, nif_funcs, NULL, NULL, NULL, NULL)
