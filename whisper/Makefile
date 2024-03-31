IMG := whisper-lab

OP_RUN_FLAGS := #--network none

.PHONY: debug
debug: build
	docker run -it \
		--rm \
		--cap-drop all \
		--security-opt=no-new-privileges \
		--memory="3000m" \
		--cpus="3" \
		--ulimit nofile=100 \
		--ulimit nproc=30 \
		-v $(CURDIR):/tmp \
		$(OP_RUN_FLAGS) \
		--name $(IMG) \
		$(IMG) bash

.PHONY: exec
exec:
	docker exec -it $(IMG) bash


.PHONY: build
build:
	docker build -t $(IMG) .
