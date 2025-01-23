IMG := pybindings
CONTAINER_OPTS := --rm \
	-e OTEL_SERVICE_NAME=$(IMG) \
	-e OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io" \
	-e OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
	-e OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=$$(op read 'op://eng-vault/honeycomb-api-key/password')" \
	-v $(PWD):/home/jovyan/work


.DEFAULT_GOAL := help

.PHONY: help
help:  ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Local Development.
.PHONY: debug
debug: build  ## Run the container
	docker run -it $(CONTAINER_OPTS) $(IMG) bash

.PHONY: build
build:  ## Build the container image
	docker build -t $(IMG) .

.PHONY: compile
compile:  ## Compile the C code
	gcc main.c certinfo.c -lcurl -o certinfo

.PHONY: compile-explicit
compile-explicit:  ## Compile the C code step-by-step
	# First compile the C code into object files.
	gcc -c -I. -I/usr/include main.c -o main.o
	gcc -c -I. -I/usr/include certinfo.c -o certinfo.o
	# Then link the object files into an executable.
	gcc main.o certinfo.o -L/usr/local/lib -lcurl -o cert_check

.PHONY: compile-bindings
compile-bindings:  ## Compile the C code with Python bindings
	gcc -shared -fPIC -I/usr/include/python3.8 -I. -I/usr/include main.c -lcurl -o libcertinfo.so certinfo.c
	python build_certinfo.py
	@echo 'To use the shared library, run the following commands:'
	@echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.'
	@echo 'cp libcertinfo.so /usr/local/lib'

##@Source Code Management.
.PHONY: fmt
fmt:  ## Format C code
	clang-format -style=google -i *.c

.PHONY: clean
clean:  ## Clean up the build artifacts
	rm -f certinfo
	rm -f *.o
	rm -f *.so
	rm -f -r __pycache__/
	rm -f cert_check
