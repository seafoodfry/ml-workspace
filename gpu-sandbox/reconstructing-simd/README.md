# Reconstructing SIMD Best practices

```
rsync -rvzP our-cuda-by-example ec2-user@${EC2}:/home/ec2-user/src
```

## Context

### Endianess

This is covered in section 1.3.1 of the manual.

If we had the value `0x12345678`.
A little endian system would store it as so

```
Memory Address   Value
    0x00          0x78
    0x01          0x56
    0x02          0x34
    0x03          0x12
```

And as you read you'd read `78 56 34 12.

A big engdian system (network protocols mostly, nowadays)

```
Memory Address   Value
    0x00          0x12
    0x01          0x34
    0x02          0x56
    0x03          0x78
```

```c
#include <stdio.h>

void printBytes(const void *obj, size_t size) {
    const unsigned char *byte = (const unsigned char *)obj;
    for (size_t i = 0; i < size; i++) {
        printf("%02x ", byte[i]);
    }
    printf("\n");
}

int main() {
    int value = 0x12345678;
    printf("Value: 0x12345678\n");
    printf("Stored as:\n");
    printBytes(&value, sizeof(value));
    return 0;
}
```

### Syntax

The manual uses the source-destination convention in assembly where, for example,

```asm
mov ax, 1234h ; copies the value 1234hex (4660d) into register AX.
```

The above is a snipet of Intel x86 Syntax.
[Wikipedia: intel x86 assembly](https://en.wikipedia.org/wiki/X86_assembly_language)
and
[CS virginia: x86 assembly guide](https://www.cs.virginia.edu/~evans/cs216/guides/x86.html)
are good resources.

For At&T syntax, which is source-destination based, check out
[CS yale: x86 assembly guide](https://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html).


### Registers

In 16-bit arch, the `IP` register contains the address of the current instruction.
In 32-bit arch, it is the extended instruction pointer register `EIP`.
In 64-bit arch, its just the register instruction pointer `RIP`.


### CPUID

In section 1.3.5, we get our first glimpse at CPUID:

> Obtain feature flags, status, and system information by using the CPUID instruction,
> by checking control register bits, and by reading model-specific registers. 

Here we get the example that if we feed `0x1` into the EAX register, then if SSE is avaialble, there will be
at bit 25 on the EDX register a `1`.

---

## Intel's Mini Developer's Manual


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