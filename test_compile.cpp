
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef _WIN32
#define EXPORT_SYM __declspec(dllexport)
#else
#define EXPORT_SYM __attribute__((visibility("default")))
#endif

extern "C" {
    EXPORT_SYM int test_func(double* x, int dim) {
        return dim;
    }
}
