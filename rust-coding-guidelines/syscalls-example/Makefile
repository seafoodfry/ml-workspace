IMG := container


all: docker

.PHONY: c-example
c-example:
	gcc -o syscall_example main.c
	./syscall_example

.PHONY: rust-example
rust-example:
	cd syscalls-example && cargo run

.PHONY: fmt
fmt:
	rustfmt -l src/*.rs

.PHONY: docker-build
docker-build:
	docker build -t $(IMG) .

.PHONY: docker-run
docker-run: docker-build
	docker run -it -w /src $(IMG) make c-example
	docker run -it -w /src $(IMG) make rust-example

.PHONY: docker-bash
docker-bash: docker-build
	docker run -it $(IMG) bash

.PHONY: clean
clean:
	rm -f ./syscall_example