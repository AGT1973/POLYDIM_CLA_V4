
#include <cstddef>
#ifdef _WIN32
#define EXPORT_SYM __declspec(dllexport)
#else
#define EXPORT_SYM __attribute__((visibility("default")))
#endif

extern "C" {
    EXPORT_SYM void add_one(void* stream, void** buffers, const char* opaque, size_t opaque_len, void* status) {
        double* in = reinterpret_cast<double*>(buffers[0]);
        double* out = reinterpret_cast<double*>(buffers[1]);
        // Dummy size assumption for test (let's say 4)
        for (int i = 0; i < 4; ++i) {
            out[i] = in[i] + 1.0;
        }
    }
}
