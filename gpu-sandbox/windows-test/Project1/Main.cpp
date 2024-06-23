#include <iostream>
#include <windows.h>


int main() {
	std::cout << "Hello, windows!" << std::endl;

    // Test Windows-specific functionality
    SYSTEMTIME st;
    GetSystemTime(&st);
    std::cout << "Current UTC time: "
        << st.wYear << "-"
        << st.wMonth << "-"
        << st.wDay << " "
        << st.wHour << ":"
        << st.wMinute << ":"
        << st.wSecond << std::endl;

	std::cin.get();

	return 0;
}