#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_KERNEL_MACROS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_KERNEL_MACROS_H_
// Provide a dummy attribute check for compilers which don't provide one.
#ifdef __has_attribute
#define SNN_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define SNN_HAS_ATTRIBUTE(x) 0
#endif  // __has_attribute
// Provide an always inline attribute to use for device code. This ensures that
// all functions in the kernel are inlined, so the optimiser can better
// understand the whoe kernel.
#if defined(__SYCL_DEVICE_ONLY__) && SNN_HAS_ATTRIBUTE(always_inline)
#define SNN_ALWAYS_INLINE __attribute__((always_inline))
#else
#define SNN_ALWAYS_INLINE
#endif
// Suggest to the compiler to unroll loops, typically on the device this leads
// to performance benefits, but make sure that this is benchmarked.
#ifdef __SYCL_DEVICE_ONLY__
#define SNN_PRAGMA_UNROLL \
  _Pragma("clang loop unroll(enable) interleave(enable)")
#else
#define SNN_PRAGMA_UNROLL
#endif  // __SYCL_DEVICE_ONLY__
#endif  // TENSORFLOW_KERNELS_CONV_OPS_KERNEL_MACROS_H_
