FROM ubuntu:22.04

WORKDIR /src

RUN apt-get update -y && \
    apt-get install -y build-essential make

COPY *.cpp .