IMG := pybindings
CONTAINER_OPTS := --rm \
	-e OTEL_SERVICE_NAME=$(IMG) \
	-e OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io" \
	-e OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
	-e OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=$$(op read 'op://eng-vault/honeycomb-api-key/password')" \
	-v $(PWD):/src


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
	gcc certinfo.c -lcurl -o certinfo


##@Source Code Management.
.PHONY: fmt
fmt:  ## Format C code
	clang-format -style=google -i *.c

.PHONY: clean
clean:  ## Clean up the build artifacts
	rm -f certinfo
