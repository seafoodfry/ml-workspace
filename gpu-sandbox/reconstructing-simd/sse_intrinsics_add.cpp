/*
    g++ -std=c++17 -msse sse_intrinsics_add.cpp -o sse_intrinsics_add.out
*/
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <memory>
#include <xmmintrin.h>
#include <emmintrin.h>  // For double-precision, 64 bit/8 byte, precision ops.


void print_m128(__m128 value);
void print_m128d(__m128d value);
void simd_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result);
void simd_add_doubles(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result);
void simd_add_aligned(const float* a, const float* b, float* result, size_t size);

template<typename T>
void print_array(const std::vector<T>& arr, const std::string& name);
template<typename SimdType, typename BaseType>
void print_m128_intrinsic(SimdType value);

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

    // Old-style rebind deprecated in C++17.
    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
    // New-style rebind (C++20 and later).
    template <typename U>
    using rebind_alloc = aligned_allocator<U, Alignment>;

    aligned_allocator() noexcept = default;
    template <class U> aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        if (auto p = static_cast<T*>(std::aligned_alloc(Alignment, n * sizeof(T))))
            return p;
        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new((void *)p) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) {
        p->~U();
    }

    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
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


void test_simd_add_aligned() {
    const size_t size = 8; // Must be a multiple of 4 for 128-bit SIMD

    // Method #1: using std::aligned_alloc (C++17).
    // Allignment must be a power of 2 and a multiple of sizeof(void*).
    // Remember that 128 bits are 16 bytes. And that a 128-bit register can fit 4 4-byte float values.
    float* a1 = static_cast<float*>(std::aligned_alloc(16, size * sizeof(float)));
    float* b1 = static_cast<float*>(std::aligned_alloc(16, size * sizeof(float)));
    float* result1 = static_cast<float*>(std::aligned_alloc(16, size * sizeof(float)));

    // Method 2: posix_memalign
    float *a2, *b2, *result2;
    if (posix_memalign(reinterpret_cast<void**>(&a2), 16, size * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    if (posix_memalign(reinterpret_cast<void**>(&b2), 16, size * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    if (posix_memalign(reinterpret_cast<void**>(&result2), 16, size * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }

    // Method #3: align std::vector with an allocation class.
    std::vector<float, aligned_allocator<float, 16>> a3(size);
    std::vector<float, aligned_allocator<float, 16>> b3(size);
    std::vector<float, aligned_allocator<float, 16>> result3(size);

    // Method #4: using __attribute__((aligned(16))) (gcc, clang) or alignas(16) (C++11).
    // Ensures stack-allocated arrays are aligned.
    // float __attribute__((aligned(16))) alignedArray3[4];
    alignas(16) float a4[size];
    alignas(16) float b4[size];
    alignas(16) float result4[size];

    // Method #5: using __mm_malloc() and __mm_free() (intel).
    float* a5 = static_cast<float*>(_mm_malloc(size * sizeof(float), 16));
    float* b5 = static_cast<float*>(_mm_malloc(size * sizeof(float), 16));
    float* result5 = static_cast<float*>(_mm_malloc(size * sizeof(float), 16));

    // Initialize arrays (example values).
    for (size_t i = 0; i < size; ++i) {
        float val_a = static_cast<float>(i + 1);
        float val_b = static_cast<float>(i + 1) * 0.1f;
        
        a1[i] = a2[i] = a3[i] = a4[i] = a5[i] = val_a;
        b1[i] = b2[i] = b3[i] = b4[i] = b5[i] = val_b;
    }

    // Perform aligned SIMD addition for each method
    simd_add_aligned(a1, b1, result1, size);
    simd_add_aligned(a2, b2, result2, size);
    simd_add_aligned(a3.data(), b3.data(), result3.data(), size);
    simd_add_aligned(a4, b4, result4, size);
    simd_add_aligned(a5, b5, result5, size);

    // Print results (only showing Method 1 for brevity)
    std::cout << "\nMethod 1 (std::aligned_alloc) result: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << result1[i] << " ";
    }
    std::cout << std::endl;

    // Print results (only showing Method 1 for brevity)
    std::cout << "\nMethod 2 (posis_memalign) result: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << result2[i] << " ";
    }
    std::cout << std::endl;

    // Print results (only showing Method 1 for brevity)
    std::cout << "\nMethod 3 (std::vector) result: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << result3[i] << " ";
    }
    std::cout << std::endl;

    // Print results (only showing Method 1 for brevity)
    std::cout << "\nMethod 4 (alignas) result: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << result4[i] << " ";
    }
    std::cout << std::endl;

    // Print results (only showing Method 1 for brevity)
    std::cout << "\nMethod 5 (_mm_aloc) result: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << result1[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory
    std::free(a1); std::free(b1); std::free(result1);
    std::free(a2); std::free(b2); std::free(result2);
    _mm_free(a5); _mm_free(b5); _mm_free(result5);
}

int main() {
    /*
        Single precision operations.
    */
    std::cout << "Explicit SIMD vector addition...\n";

    // Create two __m128 vectors.
    // Note that intel goes little-endian, so we set the numbers backwards.
    __m128 a = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f); // [1.0, 2.0, 3.0, 4.0]
    __m128 b = _mm_set_ps(8.0f, 7.0f, 6.0f, 5.0f); // [5.0, 6.0, 7.0, 8.0]

    // Perform parallel addition.
    __m128 result = _mm_add_ps(a, b);

    // Print the results.
    std::cout << "a      = ";
    print_m128(a);
    std::cout << "b      = ";
    print_m128(b);
    std::cout << "result = ";
    print_m128(result);


    std::cout << "\nSIMD addition of array-like objects...\n";
    std::vector<float> aVec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> bVec = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> resultVec(aVec.size());

    print_array(aVec, "aVec");
    print_array(bVec, "bVec");

    simd_add(aVec, bVec, resultVec);

    print_array(resultVec, "resultVec");

    /*
        Double preciison operations.
    */

    std::cout << "\nExplicit SIMD vector addition...\n";

    // Create two __m128 vectors.
    // Note that intel goes little-endian, so we set the numbers backwards.
    __m128d aDouble = _mm_set_pd(2.0, 1.0); // [1.0, 2.0]
    __m128d bDouble = _mm_set_pd(6.0, 5.0); // [5.0, 6.0]

    // Perform parallel addition.
    __m128d resultDouble = _mm_add_pd(aDouble, bDouble);

    // Print the results.
    std::cout << "aDouble = ";
    print_m128_intrinsic<__m128d, double>(aDouble);
    std::cout << "bDouble = ";
    print_m128_intrinsic<__m128d, double>(bDouble);
    std::cout << "resultDouble = ";
    print_m128_intrinsic<__m128d, double>(resultDouble);


    std::cout << "\nSIMD addition of array-like objects storing doubles...\n";
    std::vector<double> aVecD = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> bVecD = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<double> resultVecD(aVecD.size());

    print_array(aVecD, "aVecD");
    print_array(bVecD, "bVecD");

    simd_add_doubles(aVecD, bVecD, resultVecD);

    print_array(resultVecD, "resultVecD");

    /*
        Examples of aligned memory allocations.
    */
    test_simd_add_aligned();

    return 0;
}


void simd_add_aligned(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; i += 4) {
        __m128 va = _mm_load_ps(a + i);  // Note: Using _mm_load_ps instead of _mm_loadu_ps
        __m128 vb = _mm_load_ps(b + i);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_store_ps(result + i, vr);    // Note: Using _mm_store_ps instead of _mm_storeu_ps
    }
}

void simd_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result) {
    // The data() member function of vectors returns a pointer to the internal array element.
    // So X.data() + i is a way to step through the entire vector, the bit of magic is that the _mm_loadu_ps()
    // function knows to read from start up to the next 32 bits (4 bytes == 4 entries) of the vector of floats.
    for (size_t i = 0; i < a.size(); i += 4) {
        __m128 va = _mm_loadu_ps(a.data() + i);
        __m128 vb = _mm_loadu_ps(b.data() + i);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(result.data() + i, vr);
    }
}

void simd_add_doubles(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
   // A float is 4 bytes (32 bits), a double is 8 bytes (64 bits).
   // Where as an 128-bit register can fit 4 floats, it can only hold 2 doubles.
   for (size_t i = 0; i < a.size(); i += 2) {
        __m128d va = _mm_loadu_pd(a.data() + i);
        __m128d vb = _mm_loadu_pd(b.data() + i);
        __m128d vr = _mm_add_pd(va, vb);
        _mm_storeu_pd(result.data() + i, vr);
    }
}

template<typename T>
void print_array(const std::vector<T>& arr, const std::string& name) {
    std::cout << name << ": ";
    std::cout << std::setprecision(6) << std::fixed;
    for (const T& value : arr) {  // Used to be for (float f : arr) {...
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

template<typename SimdType, typename BaseType>
void print_m128_intrinsic(SimdType value) {
    BaseType* f = static_cast<BaseType*>(static_cast<void*>(&value));
    std::cout << std::setprecision(6) << std::fixed << "[";

    // Using the size ratio to figure out how many elements need to be printed.
    for (int i = 0; i < sizeof(SimdType) / sizeof(BaseType); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << f[i];
    }
    std::cout << "]" << std::endl;
}

void print_m128(__m128 value) {
    // float* f = (float*)&value;
    float* f = static_cast<float*>(static_cast<void*>(&value));
    std::cout << std::setprecision(6) << std::fixed
              << "[" << f[0] << ", " << f[1] << ", " << f[2] << ", " << f[3] << "]" << std::endl;
}

void print_m128d(__m128d value) {
    // double* f = (double*)&value;
    double* f = static_cast<double*>(static_cast<void*>(&value));
    std::cout << std::setprecision(6) << std::fixed
              << "[" << f[0] << ", " << f[1] << "]" << std::endl;
}