FROM rust:1.80-bookworm

RUN apt-get update -y && \
    apt-get install -y build-essential

WORKDIR /src

# Start by fetching and building the rust dependencies.
RUN cargo new syscalls-example
COPY Cargo.toml /src/syscalls-example
WORKDIR /src/syscalls-example
RUN cargo build

# Noew copy the source for the C program.
COPY Makefile main.c /src/

# And finally copy the source for the Rust program.
COPY ./src /src/syscalls-example/src/