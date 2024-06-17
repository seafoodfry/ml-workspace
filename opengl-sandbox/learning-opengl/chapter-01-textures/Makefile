CC := g++
# Compiler flags:
# -g    adds debugging information to the executable file
# -Wall turns on most compiler warnings
# -Wextra https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wextra
# -Werror Make all warnings into errors. 
CFLAGS := -g -O2 -Wall -Wextra -Wshadow -Werror -std=c++17 -I./../../glad/include #-std=gnu++latest # std=c++20 -std=c++17 -std=c++14 -std=c++11
LFLAGS := -lGL -lglfw3 -lX11 -lpthread -lXrandr -lXi -ldl

TARGET = main.out
SOURCES = $(wildcard *.c)
GLAD_SOURCE = ./../../glad/src/glad.c
OBJECTS = $(SOURCES:.c=.o) $(GLAD_SOURCE:.c=.o)


# Rule to link the program.
# $@ is the name of the target.
# $^ stands for all the items in the dependencies list.
$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LFLAGS)

# Rule to compile every .cpp to an .o
# The -c flag says to generate the object file,
# the -o $@ says to put the output of the compilation in the file named on the left side of the :,
# the $< is the first item in the dependencies list, and CXXFLAGS are the flags passed to the compiler.
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: $(GLAD_SOURCE)
	$(CC) $(CFLAGS) -c $< -o $(notdir $@)

clean:
	rm -f *.o $(TARGET)