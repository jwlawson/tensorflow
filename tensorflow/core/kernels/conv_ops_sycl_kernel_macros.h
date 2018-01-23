#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_KERNEL_MACROS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_KERNEL_MACROS_H_

#ifdef __SYCL_DEVICE_ONLY__
#define SNN_PRAGMA_UNROLL _Pragma("clang loop unroll(enable) interleave(enable)")
#else
#define SNN_PRAGMA_UNROLL
#endif  // __SYCL_DEVICE_ONLY__
#endif  // TENSORFLOW_KERNELS_CONV_OPS_KERNEL_MACROS_H_
