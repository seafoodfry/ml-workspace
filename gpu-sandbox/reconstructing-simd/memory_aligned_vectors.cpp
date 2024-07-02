#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <new>
#include <limits>

template <typename T, std::size_t Alignment>
class aligned_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;
    template <class U> aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }
};

template <class T, class U, size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return true;
}

template <class T, class U, size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return false;
}

// A struct that will be 12 bytes due to padding
struct TestStruct {
    char a;   // 1 byte
    int b;    // 4 bytes
    char c;   // 1 byte
    // 6 bytes of padding will be added here
};

// A struct that naturally aligns to 16 bytes
struct AlignedStruct {
    double a; // 8 bytes
    double b; // 8 bytes
};

template <typename Vector>
void print_addresses(const Vector& vec, const std::string& name) {
    std::cout << name << " addresses:\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "Element " << i << ": " 
                  << std::hex << std::setw(16) << std::setfill('0')
                  << reinterpret_cast<uintptr_t>(&vec[i]) << std::dec
                  << " (offset from 16-byte boundary: " << reinterpret_cast<uintptr_t>(&vec[i]) % 16 << ")\n";
    }
    std::cout << "Size of each element: " << sizeof(typename Vector::value_type) << " bytes\n\n";
}

int main() {
    const size_t size = 10;

    std::cout << "Size of TestStruct: " << sizeof(TestStruct) << " bytes\n";
    std::cout << "Size of AlignedStruct: " << sizeof(AlignedStruct) << " bytes\n\n";

    std::vector<TestStruct> standard_vec1(size);
    std::vector<TestStruct, aligned_allocator<TestStruct, 16>> aligned_vec1(size);

    std::vector<AlignedStruct> standard_vec2(size);
    std::vector<AlignedStruct, aligned_allocator<AlignedStruct, 16>> aligned_vec2(size);

    print_addresses(standard_vec1, "Standard vector (TestStruct)");
    print_addresses(aligned_vec1, "Aligned vector (TestStruct, 16-byte alignment)");

    print_addresses(standard_vec2, "Standard vector (AlignedStruct)");
    print_addresses(aligned_vec2, "Aligned vector (AlignedStruct, 16-byte alignment)");

    return 0;
}