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
#error This file must only be included when building with SYCL support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_SYCL_H_
#define TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_SYCL_H_

#include "tensorflow/core/kernels/pooling_ops_common.h"

namespace tensorflow {

typedef Eigen::SyclDevice SYCLDevice;

// MaxPool2D SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output tensor.
//
// For each output element, find the corresponding input window and run over
// all values in the window to find the maximum value. This value is then
// copied into that output element.
template <typename T, typename Index = int>
class MaxPool2DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool2DSYCL(const Index n_outputs, const PoolParameters& params,
                const read_accessor input_accessor,
                write_accessor output_accessor)
      : n_outputs_(n_outputs),
        p_(params),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    Index index = item.get_linear_id();
    if (index < n_outputs_) {
      Index n = index;
      const Index d = n % p_.depth_;
      n /= p_.depth_;
      Index cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
      const Index cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
      cstart = std::max(cstart, 0);
      n /= p_.out_cols_;
      Index rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
      const Index rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = std::max(rstart, 0);
      n /= p_.out_rows_;
      T maxval = Eigen::NumTraits<T>::lowest();
      const T* input_data_n =
          input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
      for (Index r = rstart; r < rend; ++r) {
        for (Index c = cstart; c < cend; ++c) {
          const Index idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
          if (input_data_n[idx] > maxval) {
            maxval = input_data_n[idx];
          }
        }
      }
      output_data[index] = maxval;
    }
  }

 private:
  const Index n_outputs_;
  const SYCL2DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct LaunchMaxPoolingOpSYCL {
  using Index = int;
  static void launch(OpKernelContext* context, Tensor* output,
                     const Tensor& tensor_in, const PoolParameters& params) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const Index output_size = output->NumElements();
    const Index n_threads = output_size;

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPool2DSYCL<T> max_pool(output_size, params, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), max_pool);
    });
  }
};
// MaxPoolGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output backprop tenor (i.e. the number of elements
// in the input data tensor).
//
// For each output backprop element we compute the possible window of values in
// the input backprop tensor which might contribute to this element. Then for
// each error in this window, compute the corresponding input window which was
// pooled into that element in the output. Walk through this input window to
// determine whether the input value is the first maximum value, and so the
// error should be propagated back to the corresponding backprop element.
template <typename T, typename Index = int>
class MaxPoolGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPoolGradSYCL(const Index n_outputs, const PoolParameters& params,
                  const read_accessor input_data_accessor,
                  const read_accessor output_data_accessor,
                  const read_accessor input_backprop_accessor,
                  write_accessor output_backprop_accessor)
      : n_outputs_(n_outputs),
        p_(params),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    const T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    const T* input_backprop =
        ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const Index index = item.get_linear_id();
    if (index < n_outputs_) {
      T output_value = 0;
      Index n = index;
      const Index d = n % p_.depth_;
      n /= p_.depth_;
      const Index c = (n % p_.in_cols_) + p_.pad_cols_;
      const Index poolcstart =
          (c < p_.window_cols_) ? 0
                                : (c - p_.window_cols_) / p_.stride_cols_ + 1;
      const Index poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
      n /= p_.in_cols_;
      const Index r = (n % p_.in_rows_) + p_.pad_rows_;
      const Index poolrstart =
          (r < p_.window_rows_) ? 0
                                : (r - p_.window_rows_) / p_.stride_rows_ + 1;
      const Index poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
      n /= p_.in_rows_;
      const Index index_no_n =
          index - n * p_.in_cols_ * p_.in_rows_ * p_.depth_;

      const T* input_data_n =
          input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
      const T* output_data_n =
          output_data + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;
      const T* input_backprop_n =
          input_backprop + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;

      for (Index poolr = poolrstart; poolr < poolrend; ++poolr) {
        Index rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
        const Index rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
        rstart = std::max(rstart, 0);

        for (Index poolc = poolcstart; poolc < poolcend; ++poolc) {
          Index cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
          const Index cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
          cstart = std::max(cstart, 0);

          const Index output_data_idx =
              (poolr * p_.out_cols_ + poolc) * p_.depth_ + d;
          bool should_continue = true;
          bool is_max = (input_data[index] == output_data_n[output_data_idx]);
          for (Index win_r = rstart; win_r < rend && should_continue; ++win_r) {
            for (Index win_c = cstart; win_c < cend && should_continue;
                 ++win_c) {
              const Index input_data_idx =
                  (win_r * p_.in_cols_ + win_c) * p_.depth_ + d;
              if (input_data_idx == index_no_n) {
                should_continue = false;
              } else if (input_data_n[input_data_idx] ==
                         output_data_n[output_data_idx]) {
                should_continue = false;
                is_max = false;
              }
            }
          }
          if (is_max) {
            output_value += input_backprop_n[output_data_idx];
          }
        }
      }
      output_backprop[index] = output_value;
    }
  }

 private:
  const Index n_outputs_;
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T, typename Index = int>
class MaxPoolGradArgmaxSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
  using tmp_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::discard_read_write,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPoolGradArgmaxSYCL(const Index n_outputs, const PoolParameters& params,
                        const read_accessor input_data_accessor,
                        const read_accessor output_data_accessor,
                        const read_accessor input_backprop_accessor,
                        tmp_accessor argmax_accessor,
                        write_accessor output_backprop_accessor)
      : n_outputs_(n_outputs),
        p_(params),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        argmax_accessor_(argmax_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::nd_item<1> item) {
    const T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    const T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    const T* input_backprop =
        ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);
    Index* argmax_data = ConvertToActualTypeSycl(Index, argmax_accessor_);

    const Index index = item.get_global_linear_id();
    if (index < p_.batch_ * p_.out_rows_ * p_.out_cols_ * p_.depth_) {
      Index n = index;
      const Index d = n % p_.depth_;
      n /= p_.depth_;
      Index cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
      const Index cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
      cstart = std::max(cstart, 0);
      n /= p_.out_cols_;
      Index rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
      const Index rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = std::max(rstart, 0);
      n /= p_.out_rows_;
      Index max_idx = -1;
      for (Index r = rstart; r < rend && max_idx < 0; ++r) {
        for (Index c = cstart; c < cend && max_idx < 0; ++c) {
          const Index idx =
              ((n * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          if (input_data[idx] == output_data[index]) {
            max_idx = idx;
          }
        }
      }
      argmax_data[index] = max_idx;
    }
    item.barrier(cl::sycl::access::fence_space::global_space);
    if (index < n_outputs_) {
      T output_value = 0;
      Index n = index;
      const Index d = n % p_.depth_;
      n /= p_.depth_;
      const Index c = (n % p_.in_cols_) + p_.pad_cols_;
      const Index poolcstart =
          (c < p_.window_cols_) ? 0
                                : (c - p_.window_cols_) / p_.stride_cols_ + 1;
      const Index poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
      n /= p_.in_cols_;
      const Index r = (n % p_.in_rows_) + p_.pad_rows_;
      const Index poolrstart =
          (r < p_.window_rows_) ? 0
                                : (r - p_.window_rows_) / p_.stride_rows_ + 1;
      const Index poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
      n /= p_.in_rows_;

      const T* input_backprop_n =
          input_backprop + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;
      const Index* argmax_data_n =
          argmax_data + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;

      for (Index poolr = poolrstart; poolr < poolrend; ++poolr) {
        for (Index poolc = poolcstart; poolc < poolcend; ++poolc) {
          const Index idx = (poolr * p_.out_cols_ + poolc) * p_.depth_ + d;
          // If no argmax was found then the value could be -1, but the index
          // is always non-negative so this will never cause an invalid read.
          if (argmax_data_n[idx] == index) {
            output_value += input_backprop_n[idx];
          }
        }
      }
      output_backprop[index] = output_value;
    }
  }

 private:
  const Index n_outputs_;
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  tmp_accessor argmax_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPoolingGradSYCL {
  using Index = int;
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const PoolParameters& params, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const Index output_size = output->NumElements();
    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_threads =
        output_size + (workgroup_size - (output_size % workgroup_size));
    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    if (use_argmax(params)) {
      Tensor argmax_tensor;
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<int>::v(),
                                          tensor_out.shape(), &argmax_tensor));
      auto argmax_buffer =
          device.get_sycl_buffer(argmax_tensor.template flat<Index>().data());

      device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
        auto input_data_access =
            input_data_buffer.template get_access<cl::sycl::access::mode::read>(
                cgh);
        auto output_data_access =
            output_data_buffer
                .template get_access<cl::sycl::access::mode::read>(cgh);
        auto input_backprop_access =
            input_backprop_buffer
                .template get_access<cl::sycl::access::mode::read>(cgh);
        auto output_backprop_access =
            output_backprop_buffer
                .template get_access<cl::sycl::access::mode::write>(cgh);
        auto argmax_access = argmax_buffer.template get_access<
            cl::sycl::access::mode::discard_read_write>(cgh);
        MaxPoolGradArgmaxSYCL<T> max_pool(
            output_size, params, input_data_access, output_data_access,
            input_backprop_access, argmax_access, output_backprop_access);

        cgh.parallel_for(
            cl::sycl::nd_range<1>(cl::sycl::range<1>(n_threads),
                                  cl::sycl::range<1>(workgroup_size)),
            max_pool);
      });
    } else {
      device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
        auto input_data_access =
            input_data_buffer.template get_access<cl::sycl::access::mode::read>(
                cgh);
        auto output_data_access =
            output_data_buffer
                .template get_access<cl::sycl::access::mode::read>(cgh);
        auto input_backprop_access =
            input_backprop_buffer
                .template get_access<cl::sycl::access::mode::read>(cgh);
        auto output_backprop_access =
            output_backprop_buffer
                .template get_access<cl::sycl::access::mode::write>(cgh);
        MaxPoolGradSYCL<T> max_pool(output_size, params, input_data_access,
                                    output_data_access, input_backprop_access,
                                    output_backprop_access);
        cgh.parallel_for(cl::sycl::range<1>(output_size), max_pool);
      });
    }
  }

 private:
  static bool use_argmax(const PoolParameters& params) {
    return false;
//    return params.window_rows > 5 && params.window_cols > 5 &&
//           params.row_stride == 1 && params.col_stride == 1;
  }
};
// MaxPoolGradGrad SYCL kernel. Expects the number of threads to be equal to
// the number of elements in the output backprop tensor, i.e. the number of
// elements in the output tensor.
//
// For each element in the output backprop tensor, find the corresponding input
// window, and compare the input and output data to find the index of the
// maximum value in the input tensor. This is then the index of the gradient to
// pass through to the output backprop tensor.
template <typename T, typename Index = int>
class MaxPoolGradGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPoolGradGradSYCL(const Index n_outputs, const PoolParameters& params,
                      const read_accessor input_data_accessor,
                      const read_accessor output_data_accessor,
                      const read_accessor input_backprop_accessor,
                      write_accessor output_backprop_accessor)
      : n_outputs_(n_outputs),
        p_(params),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    Index index = item.get_linear_id();
    if (index < n_outputs_) {
      Index n = index;
      Index d = n % p_.depth_;
      n /= p_.depth_;
      Index cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
      Index cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
      cstart = std::max(cstart, 0);
      n /= p_.out_cols_;
      Index rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
      Index rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = std::max(rstart, 0);
      n /= p_.out_rows_;
      Index maxidx = -1;
      bool should_stop = false;
      const T* input_data_n =
          input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
      for (Index r = rstart; r < rend && !should_stop; ++r) {
        for (Index c = cstart; c < cend && !should_stop; ++c) {
          Index idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
          if (output_data[index] == input_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
      if (maxidx != -1) {
        output_backprop[index] =
            input_backprop[n * p_.in_rows_ * p_.in_cols_ * p_.depth_ + maxidx];
      }
    }
  }

 private:
  const Index n_outputs_;
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPoolingGradGradOpSYCL {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& out_backprop, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const int num_threads = output->NumElements();

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto output_data_access =
          output_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPoolGradGradSYCL<T> maxpoolgradgrad(
          num_threads, params, input_data_access, output_data_access,
          input_backprop_access, output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), maxpoolgradgrad);
    });
  }
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_SYCL_H_
