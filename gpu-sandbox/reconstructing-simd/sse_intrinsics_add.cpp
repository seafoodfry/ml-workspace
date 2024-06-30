/*
    g++ -msse sse_intrinsics_add.cpp -o sse_intrinsics_add.out
*/
#include <iostream>
#include <iomanip>
#include <vector>
#include <xmmintrin.h>
#include <emmintrin.h>  // For double-precision, 64 bit/8 byte, precision ops.


void print_m128(__m128 value);
void print_m128d(__m128d value);
void simd_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result);
void simd_add_doubles(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result);

template<typename T>
void print_array(const std::vector<T>& arr, const std::string& name);



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
    print_m128d(aDouble);
    std::cout << "bDouble = ";
    print_m128d(bDouble);
    std::cout << "resultDouble = ";
    print_m128d(resultDouble);


    std::cout << "\nSIMD addition of array-like objects storing doubles...\n";
    std::vector<double> aVecD = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> bVecD = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<double> resultVecD(aVecD.size());

    print_array(aVecD, "aVecD");
    print_array(bVecD, "bVecD");

    simd_add_doubles(aVecD, bVecD, resultVecD);

    print_array(resultVecD, "resultVecD");

    return 0;
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