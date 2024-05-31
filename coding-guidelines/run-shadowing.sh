#!/bin/bash

echo "Compiling and running shadowing example..."
set -x
g++ -c shadowing.cpp -o shadowing.o
g++ -o shadowing shadowing.o
./shadowing
set +x

echo "Prevent shadowing with compiler flags..."
set -x
g++ -Werror -Wshadow -c shadowing.cpp -o shadowing.o
set +x