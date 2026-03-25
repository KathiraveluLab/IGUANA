#include <erl_nif.h>
#include <math.h>

/**
 * calculate_entropy_nif/1
 * 
 * Performs Shannon entropy calculation in native C to avoid the overhead of 
 * Erlang functional recursion for high-frequency telemetry bursts.
 * 
 * Formula: H(P) = -Sum(p * log2(p))
 */
static ERL_NIF_TERM calculate_entropy_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int len;
    ERL_NIF_TERM list = argv[0];
    
    if (!enif_get_list_length(env, list, &len)) {
        return enif_make_badarg(env);
    }

    double entropy = 0.0;
    ERL_NIF_TERM head, tail = list;

    while (enif_get_list_cell(env, tail, &head, &tail)) {
        double p;
        if (enif_get_double(env, head, &p)) {
            if (p > 0.0) {
                entropy -= p * log2(p);
            }
        }
    }

    return enif_make_double(env, entropy);
}

static ErlNifFunc nif_funcs[] = {
    {"calculate_entropy_nif", 1, calculate_entropy_nif}
};

ERL_NIF_INIT(iguana_entropy_guard, nif_funcs, NULL, NULL, NULL, NULL)
