// c_function.cpp
# include <iostream>

extern "C" {
    int add(int a, int b) {
        return a + b;
    }
}