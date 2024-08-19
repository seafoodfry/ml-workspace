.PHONY: all
all: fmt
	cargo bench

.PHONY: fmt
fmt:
	rustfmt -l src/*.rs
	rustfmt -l benches/*.rs