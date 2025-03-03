.PHONY: compile
compile:
	mpicc -o mpi_hello mpi_hello.c

.PHONY:
run: compile
	mpirun -np 4 ./mpi_hello

.PHONY: mpi-run
mpi-run: compile
	mpirun -np 8 --hostfile mpi_hostfile ./mpi_hello
