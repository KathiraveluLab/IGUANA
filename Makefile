# IGUANA NIF Build System
ERL_INCLUDE = $(shell erl -noshell -eval 'io:format("~s", [filename:join([code:root_dir(), "usr", "include"])]), halt().')
PRIV_DIR = priv

all: $(PRIV_DIR)/iguana_nif_accelerator.so

$(PRIV_DIR)/iguana_nif_accelerator.so: src/erlang/iguana_nif_accelerator.c
	mkdir -p $(PRIV_DIR)
	gcc -fPIC -shared -o $@ $^ -I$(ERL_INCLUDE) -lm

clean:
	rm -f $(PRIV_DIR)/*.so
