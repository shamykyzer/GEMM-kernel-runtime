#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace tile_runtime {

template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        std::size_t bytes = n * sizeof(T);
        // aligned_alloc requires size to be a multiple of alignment
        bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
#if defined(_MSC_VER)
        void* ptr = _aligned_malloc(bytes, Alignment);
#else
        void* ptr = std::aligned_alloc(Alignment, bytes);
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};

}  // namespace tile_runtime
