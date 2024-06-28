# Reconstructing SIMD Best practices

```
rsync -rvzP our-cuda-by-example ec2-user@${EC2}:/home/ec2-user/src
```


A 128-bit register is 16 bytes long.
A 256-bit register is 32 bytes, and a 512-bit on is 64 bytes.

Integers are commonly 4 bytes or 32 bits long.


CPUID stands for "CPU Identification". It's a processor supplementary instruction for x86 architecture CPUs.
The CPUID instruction allows software to discover details of the processor, including which instruction sets it supports.

```
grep -E 'sse2|avx|avx2|avx512' /proc/cpuinfo | sort -u
```


The -march=native flag tells the compiler to use the highest SIMD instruction set supported by your CPU.

