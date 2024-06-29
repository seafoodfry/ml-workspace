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


A "leaf" in CPUID terminology is essentially a category of information.
When you call CPUID, you specify which leaf (category) of information you want by setting a value in the EAX register before calling CPUID.

Location in the Manual:
In Volume 2A, look for the section titled "CPUID—CPU Identification".
This is typically in Chapter 3, "Instruction Set Reference, A-L".

Leaf 0: Maximum Basic Information and Vendor ID
Leaf 1: Processor Info and Feature Bits
Leaf 2: Cache and TLB Descriptor Information
Leaf 3: Processor Serial Number (obsolete)
Leaf 4: Deterministic Cache Parameters
Leaf 5: MONITOR/MWAIT Features
Leaf 6: Thermal and Power Management Features
Leaf 7: Structured Extended Feature Flags