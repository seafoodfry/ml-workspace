#!/bin/bash

# List of possible C++ standards
standards=("c++98" "c++03" "c++11" "c++14" "c++17" "c++20" "c++23" "c++26")

# Simple C++ program to compile
program="#include <iostream>\nint main() { std::cout << \"Hello, World!\" << std::endl; return 0; }"

# Create a temporary file for the program
tempfile=$(mktemp /tmp/testprog.XXXXXX.cpp)
echo -e $program > $tempfile

# Check each standard
for std in "${standards[@]}"; do
    echo -n "Checking for $std: "
    if g++ -std=$std $tempfile -o /dev/null 2>/dev/null; then
        echo "Supported"
    else
        echo "Not supported"
    fi
done

# Clean up
rm $tempfile