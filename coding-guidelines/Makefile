IMG := clab

.PHONY: all
all: build
	docker run -it -v $(CURDIR):/src $(IMG) bash

.PHONY: build
build:
	docker build -t $(IMG) .

.PHONY: clean
clean:
	rm -f *.o
	rm -f shadowing