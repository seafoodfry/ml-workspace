.PHONY: all
all:
	nvcc 01-main.cu -o 01-main.out
	./01-main.out
	nvcc 02-hello.cu -o 02-hello.out
	./02-hello.out
	nvcc 03-cudaDeviceProp.cu -o 03-cudaDeviceProp.out
	./03-cudaDeviceProp.out
	nvcc 04-choose-gpu.cu -o 04-choose-gpu.out
	./04-choose-gpu.out
	nvcc 05-vector-sum.cu -o 05-vector-sum.out
	time ./05-vector-sum.out
	# Julia CPU.
	g++ -I/usr/include/GL -c 06-julia-sets.c -o 06-julia-sets.o
	g++ -o 06-julia-sets.out 06-julia-sets.o -lGL -lGLU -lglut
	time ./06-julia-sets.out
	# Julia GPU.
	nvcc -I/usr/include/GL -c 06-julia-sets-gpu.cu -o 06-julia-sets-gpu.o
	nvcc -o 06-julia-sets-gpu.out 06-julia-sets-gpu.o -lGL -lGLU -lglut
	time ./06-julia-sets-gpu.out


.PHONY: clean
clean:
	rm -f *.o *.out