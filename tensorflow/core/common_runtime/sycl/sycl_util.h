/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_

#include "tensorflow/core/common_runtime/device.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// For DMA helper
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
inline void const* GetBase(const Tensor* src) { return DMAHelper::base(src); }
inline void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

inline void SYCLmemcpy(Eigen::SyclDevice const& device,
                       Tensor const& src_tensor, Tensor* dst_tensor) {
  const size_t size = src_tensor.TotalBytes();
  void* dst_ptr = GetBase(dst_tensor);
  void const* src_ptr = GetBase(&src_tensor);
  device.memcpy(dst_ptr, src_ptr, size);
}
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
