/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/pooling_ops_3d.h"

#include <array>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cudnn_pooling_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_3d_gpu.h"
#endif
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

Pool3dParameters::Pool3dParameters(OpKernelContext* context,
                                   const std::vector<int32>& ksize,
                                   const std::vector<int32>& stride,
                                   Padding padding, TensorFormat data_format,
                                   const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 4 dimensions.
  OP_REQUIRES(context, tensor_in_shape.dims() == 5,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_planes = GetTensorDim(ksize, data_format, '0');
  window_rows = GetTensorDim(ksize, data_format, '1');
  window_cols = GetTensorDim(ksize, data_format, '2');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  plane_stride = GetTensorDim(stride, data_format, '0');
  row_stride = GetTensorDim(stride, data_format, '1');
  col_stride = GetTensorDim(stride, data_format, '2');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 3D pooling across plane/width/height. Depthwise
  // pooling is not supported.
  OP_REQUIRES(
      context, depth_window == 1 && depth_stride == 1,
      errors::Unimplemented(
          "Pooling3d only supports pooling across plane/width/height."));

  OP_REQUIRES_OK(context, GetWindowedOutputSize(tensor_in_planes, window_planes,
                                                plane_stride, padding,
                                                &out_plane, &pad_planes));
  OP_REQUIRES_OK(context,
                 GetWindowedOutputSize(tensor_in_rows, window_rows, row_stride,
                                       padding, &out_height, &pad_rows));
  OP_REQUIRES_OK(context,
                 GetWindowedOutputSize(tensor_in_cols, window_cols, col_stride,
                                       padding, &out_width, &pad_cols));
}

TensorShape Pool3dParameters::forward_output_shape() {
  return ShapeFromFormat(data_format, tensor_in_batch,
                         {{out_plane, out_height, out_width}}, depth);
}

enum PoolingType { MAX, AVG };

template <typename Device, typename T, PoolingType Type>
struct LaunchPoolingOp;

template <typename T>
struct LaunchPoolingOp<CPUDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    output->tensor<T, 5>().device(context->eigen_device<CPUDevice>()) =
        Eigen::CuboidAvgPooling(tensor_in.tensor<T, 5>(), window[0], window[1],
                                window[2], stride[0], stride[1], stride[2],
                                BrainPadding2EigenPadding(padding_type));
  }
};

template <typename T>
struct LaunchPoolingOp<CPUDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    output->tensor<T, 5>().device(context->eigen_device<CPUDevice>()) =
        Eigen::CuboidMaxPooling(tensor_in.tensor<T, 5>(), window[0], window[1],
                                window[2], stride[0], stride[1], stride[2],
                                BrainPadding2EigenPadding(padding_type));
  }
};

template <typename Device, typename T, PoolingType Type>
class Pooling3DOp : public UnaryOp<T> {
 public:
  explicit Pooling3DOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->device_type() == DEVICE_CPU) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("Default Pooling3DOp only supports NDHWC ",
                                  "on device type ",
                                  DeviceTypeString(context->device_type())));
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                 GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                 GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    const int64 depth = GetTensorDim(tensor_in, data_format_, 'C');
    const int64 in_batch = GetTensorDim(tensor_in, data_format_, 'N');

    // Dimension order for these arrays is: x, y, z.
    std::array<int64, 3> input_size{
        {GetTensorDim(tensor_in, data_format_, '2'),
         GetTensorDim(tensor_in, data_format_, '1'),
         GetTensorDim(tensor_in, data_format_, '0')}};
    std::array<int64, 3> window{{GetTensorDim(ksize_, data_format_, '2'),
                                 GetTensorDim(ksize_, data_format_, '1'),
                                 GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64, 3> stride{{GetTensorDim(stride_, data_format_, '2'),
                                 GetTensorDim(stride_, data_format_, '1'),
                                 GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64, 3> padding, out;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, window, stride,
                                            padding_, &out, &padding));

    TensorShape out_shape = ShapeFromFormat(data_format_, in_batch,
                                            {{out[2], out[1], out[0]}}, depth);
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    LaunchPoolingOp<Device, T, Type>::launch(context, tensor_in, window, stride,
                                             padding, data_format_, padding_,
                                             output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchMaxPooling3dGradOp;

template <typename T>
struct LaunchMaxPooling3dGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    output->flat<T>().setZero();
    for (int64 p = 0; p < out_backprop.dim_size(3); ++p) {
      // Calculate broadcast size for planes/rows/cols. For SAME padding,
      // current index could be in the padding area, and
      //   p * stride_planes + window_planes
      // could be beyond the input tensor's boundary. In such cases, change
      // the starting index and reduce the broadcast size.
      //
      // The same procedure is repeated for every spatial dimension in the
      // nested loops below.
      int pindex, psize;
      std::array<int64, 3> input_size{{tensor_in.dim_size(3),
                                       tensor_in.dim_size(2),
                                       tensor_in.dim_size(1)}};
      OP_REQUIRES_OK(context,
                     GetBroadcastSize(p, input_size[0], window[0], stride[0],
                                      padding[0], &pindex, &psize));
      for (int64 r = 0; r < out_backprop.dim_size(2); ++r) {
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, input_size[1], window[1], stride[1],
                                        padding[1], &rindex, &rsize));
        for (int64 c = 0; c < out_backprop.dim_size(1); ++c) {
          int cindex, csize;
          OP_REQUIRES_OK(
              context, GetBroadcastSize(c, input_size[2], window[2], stride[2],
                                        padding[2], &cindex, &csize));
          TensorSlice src{{0, -1}, {c, 1}, {r, 1}, {p, 1}, {0, -1}};
          TensorSlice dst{{0, -1},
                          {cindex, csize},
                          {rindex, rsize},
                          {pindex, psize},
                          {0, -1}};
          Eigen::DSizes<Eigen::DenseIndex, 5> src_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> src_sizes;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_sizes;
          src.FillIndicesAndSizes<5>(out_backprop.shape(), &src_indices,
                                     &src_sizes);
          dst.FillIndicesAndSizes<5>(tensor_in.shape(), &dst_indices,
                                     &dst_sizes);

#if !defined(EIGEN_HAS_INDEX_LIST)
          Eigen::array<int, 5> bcast = {1, csize, rsize, psize, 1};
#else
          Eigen::IndexList<Eigen::type2index<1>, int, int, int,
                           Eigen::type2index<1> >
              bcast;
          bcast.set(1, csize);
          bcast.set(2, rsize);
          bcast.set(3, psize);
#endif

          // Slice from tensor_in.
          Eigen::Tensor<T, 5, Eigen::RowMajor> tensor_in_slice(dst_sizes);
          tensor_in_slice.device(context->eigen_cpu_device()) =
              tensor_in.tensor<T, 5>().slice(dst_indices, dst_sizes);

          // Slice from tensor_out.
          Eigen::Tensor<T, 5, Eigen::RowMajor> tensor_out_slice(src_sizes);
          tensor_out_slice.device(context->eigen_cpu_device()) =
              tensor_out.tensor<T, 5>().slice(src_indices, src_sizes);

          // Backprop slice.
          Eigen::Tensor<T, 5, Eigen::RowMajor> out_backprop_slice(src_sizes);
          out_backprop_slice.device(context->eigen_cpu_device()) =
              out_backprop.tensor<T, 5>().slice(src_indices, src_sizes);

          // The true backprop slice: if an element is the max, choose
          // the backprop slice; otherwise set to 0.
          Eigen::Tensor<T, 5, Eigen::RowMajor> select_slice(dst_sizes);
          Eigen::Tensor<T, 5, Eigen::RowMajor> mat0(dst_sizes);
          mat0.setZero();
          select_slice =
              ((tensor_in_slice - tensor_out_slice.broadcast(bcast)).abs() <
               tensor_in_slice.constant(1e-5))
                  .select(out_backprop_slice.broadcast(bcast), mat0);

          output->tensor<T, 5>()
              .slice(dst_indices, dst_sizes)
              .device(context->eigen_cpu_device()) += select_slice;
        }
      }
    }
  }
};

template <class Device, class T>
class MaxPooling3dGradOp : public OpKernel {
 public:
  explicit MaxPooling3dGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->device_type() == DEVICE_CPU) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Default MaxPooling3dGradOp only supports NDHWC ",
              "on device type ", DeviceTypeString(context->device_type())));
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                 GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                 GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 5,
                errors::InvalidArgument("tensor_out must be 5-dimensional"));
    OP_REQUIRES(context, out_backprop.dims() == 5,
                errors::InvalidArgument("out_backprop must be 5-dimensional"));

    const TensorShape& output_shape = tensor_in.shape();
    Tensor* input_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &input_backprop));
    std::array<int64, 3> input_size{
        {GetTensorDim(output_shape, data_format_, '2'),
         GetTensorDim(output_shape, data_format_, '1'),
         GetTensorDim(output_shape, data_format_, '0')}};
    std::array<int64, 3> window{{GetTensorDim(ksize_, data_format_, '2'),
                                 GetTensorDim(ksize_, data_format_, '1'),
                                 GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64, 3> stride{{GetTensorDim(stride_, data_format_, '2'),
                                 GetTensorDim(stride_, data_format_, '1'),
                                 GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, window, stride,
                                            padding_, &out, &padding));
    LaunchMaxPooling3dGradOp<Device, T>::launch(
        context, tensor_in, tensor_out, out_backprop, window, stride, out,
        padding, data_format_, input_backprop);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchAvgPooling3dGradOp;

template <typename T>
struct LaunchAvgPooling3dGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& output_shape,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    output->flat<T>().setZero();
    std::array<int64, 3> input_size = {{tensor_in_shape.dim_size(3),
                                        tensor_in_shape.dim_size(2),
                                        tensor_in_shape.dim_size(1)}};
    for (int64 p = 0; p < out_backprop.dim_size(3); ++p) {
      // Calculate broadcast size for planes/rows/cols. For SAME padding,
      // current index could be in the padding area, and
      //   p * stride_planes + window_planes
      // could be beyond the input tensor's boundary. In such cases, change
      // the starting index and reduce the broadcast size.
      //
      // The same procedure is repeated for every spatial dimension in the
      // nested loops below.
      int pindex, psize;
      OP_REQUIRES_OK(context,
                     GetBroadcastSize(p, input_size[0], window[0], stride[0],
                                      padding[0], &pindex, &psize));
      for (int64 r = 0; r < out_backprop.dim_size(2); ++r) {
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, input_size[1], window[1], stride[1],
                                        padding[1], &rindex, &rsize));
        for (int64 c = 0; c < out_backprop.dim_size(1); ++c) {
          int cindex, csize;
          OP_REQUIRES_OK(
              context, GetBroadcastSize(c, input_size[2], window[2], stride[2],
                                        padding[2], &cindex, &csize));
          TensorSlice src{{0, -1}, {c, 1}, {r, 1}, {p, 1}, {0, -1}};
          TensorSlice dst{{0, -1},
                          {cindex, csize},
                          {rindex, rsize},
                          {pindex, psize},
                          {0, -1}};
          Eigen::DSizes<Eigen::DenseIndex, 5> src_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> src_sizes;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_sizes;
          src.FillIndicesAndSizes<5>(out_backprop.shape(), &src_indices,
                                     &src_sizes);
          dst.FillIndicesAndSizes<5>(tensor_in_shape, &dst_indices, &dst_sizes);
#if !defined(EIGEN_HAS_INDEX_LIST)
          Eigen::array<int, 5> bcast = {1, csize, rsize, psize, 1};
#else
          Eigen::IndexList<Eigen::type2index<1>, int, int, int,
                           Eigen::type2index<1> >
              bcast;
          bcast.set(1, csize);
          bcast.set(2, rsize);
          bcast.set(3, psize);
#endif
          Eigen::Tensor<T, 5, Eigen::RowMajor> slices(src_sizes);
          slices.device(context->eigen_cpu_device()) =
              out_backprop.tensor<T, 5>().slice(src_indices, src_sizes);
          // Divide by the size of the actual patch (psize * rsize * csize).
          float divide_size = rsize * csize * psize * 1.0f;
          slices *= slices.constant(1.0f / divide_size);

          output->tensor<T, 5>()
              .slice(dst_indices, dst_sizes)
              .device(context->eigen_cpu_device()) += slices.broadcast(bcast);
        }
      }
    }
  }
};

template <class Device, class T>
class AvgPooling3dGradOp : public OpKernel {
 public:
  explicit AvgPooling3dGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->device_type() == DEVICE_CPU) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Default AvgPooling3dGradOp only supports NDHWC ",
              "on device type ", DeviceTypeString(context->device_type())));
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                 GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                 GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 5,
        errors::InvalidArgument("tensor_in must be 1-dimensional and 5 "
                                "elements"));
    OP_REQUIRES(context, out_backprop.dims() == 5,
                errors::InvalidArgument("out_backprop must be 5-dimensional"));

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // Dimension order for these arrays is x, y, z.
    std::array<int64, 3> input_size{
        {GetTensorDim(output_shape, data_format_, '2'),
         GetTensorDim(output_shape, data_format_, '1'),
         GetTensorDim(output_shape, data_format_, '0')}};
    std::array<int64, 3> window{{GetTensorDim(ksize_, data_format_, '2'),
                                 GetTensorDim(ksize_, data_format_, '1'),
                                 GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64, 3> stride{{GetTensorDim(stride_, data_format_, '2'),
                                 GetTensorDim(stride_, data_format_, '1'),
                                 GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64, 3> padding, out;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, window, stride,
                                            padding_, &out, &padding));

    LaunchAvgPooling3dGradOp<Device, T>::launch(
        context, output_shape, out_backprop, window, stride, out, padding,
        data_format_, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchMaxPooling3dGradGradOp;

template <typename T>
struct LaunchMaxPooling3dGradGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, const Pool3dParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& tensor_top_diff,
                     Tensor* tensor_bottom_diff) {
    OP_REQUIRES(
        context, params.data_format == FORMAT_NHWC,
        errors::InvalidArgument("Default MaxPooling3dGradGradOp only supports",
                                "NDHWC on CPU device type"));

    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;

    ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                               params.tensor_in_planes * params.tensor_in_cols *
                                   params.tensor_in_rows *
                                   params.tensor_in_batch);
    ConstEigenMatrixMap out_mat(tensor_out.flat<T>().data(), params.depth,
                                params.out_plane * params.out_width *
                                    params.out_height * params.tensor_in_batch);
    ConstEigenMatrixMap top_diff_mat(
        tensor_top_diff.flat<T>().data(), params.depth,
        params.tensor_in_planes * params.tensor_in_cols *
            params.tensor_in_rows * params.tensor_in_batch);
    EigenMatrixMap bottom_diff_mat(
        tensor_bottom_diff->flat<T>().data(), params.depth,
        params.out_plane * params.out_width * params.out_height *
            params.tensor_in_batch);

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());

    auto shard = [&params, &in_mat, &out_mat, &top_diff_mat, &bottom_diff_mat](
                     int64 start, int64 limit) {
      const int32 depth = params.depth;
      const int32 in_planes = params.tensor_in_planes;
      const int32 in_rows = params.tensor_in_rows;
      const int32 in_cols = params.tensor_in_cols;
      const int32 pad_planes = params.pad_planes;
      const int32 pad_rows = params.pad_rows;
      const int32 pad_cols = params.pad_cols;
      const int32 window_planes = params.window_planes;
      const int32 window_rows = params.window_rows;
      const int32 window_cols = params.window_cols;
      const int32 plane_stride = params.plane_stride;
      const int32 row_stride = params.row_stride;
      const int32 col_stride = params.col_stride;
      const int32 out_plane = params.out_plane;
      const int32 out_height = params.out_height;
      const int32 out_width = params.out_width;

      {
        // Initializes the output grad backprop tensor with 0.
        const int32 output_image_size =
            out_plane * out_height * out_width * params.depth;
        EigenMatrixMap bottom_diff_shard(
            bottom_diff_mat.data() + start * output_image_size, 1,
            (limit - start) * output_image_size);
        bottom_diff_shard.setZero();
      }

      for (int b = start; b < limit; ++b) {
        for (int pp = 0; pp < out_plane; ++pp) {
          for (int ph = 0; ph < out_height; ++ph) {
            for (int pw = 0; pw < out_width; ++pw) {
              // (p_start, p_end) * (h_start, h_end) * (w_start, w_end) is the
              // range that the input vector projects to.
              int p_start = pp * plane_stride - pad_planes;
              const int p_end = std::min(p_start + window_planes, in_planes);
              int h_start = ph * row_stride - pad_rows;
              const int h_end = std::min(h_start + window_rows, in_rows);
              int w_start = pw * col_stride - pad_cols;
              const int w_end = std::min(w_start + window_cols, in_cols);
              p_start = std::max(p_start, 0);
              h_start = std::max(h_start, 0);
              w_start = std::max(w_start, 0);
              const int out_index =
                  ((b * out_plane + pp) * out_height + ph) * out_width + pw;
              // Find value corresponding to the input maximum in top_diff.
              for (int d = 0; d < depth; ++d) {
                const T& output_ref = out_mat.coeffRef(d, out_index);
                bool should_stop = false;
                for (int p = p_start; p < p_end && !should_stop; ++p) {
                  for (int h = h_start; h < h_end && !should_stop; ++h) {
                    for (int w = w_start; w < w_end && !should_stop; ++w) {
                      const int in_index =
                          ((b * in_planes + p) * in_rows + h) * in_cols + w;
                      const T& input_ref = in_mat.coeffRef(d, in_index);
                      if (output_ref == input_ref) {
                        T& bottom_diff_ref =
                            bottom_diff_mat.coeffRef(d, out_index);
                        bottom_diff_ref = top_diff_mat.coeffRef(d, in_index);
                        should_stop = true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    };
    const int64 shard_cost =
        params.out_plane * params.out_height * params.out_width * params.depth *
        params.window_planes * params.window_rows * params.window_cols;
    Shard(worker_threads.num_threads, worker_threads.workers,
          params.tensor_in_batch, shard_cost, shard);
  }
};

template <class Device, class T>
class MaxPooling3dGradGradOp : public OpKernel {
 public:
  explicit MaxPooling3dGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    const int32 ksize_c = GetTensorDim(ksize_, data_format_, 'C');
    const int32 stride_c = GetTensorDim(stride_, data_format_, 'C');
    OP_REQUIRES(context, ksize_c == 1 && stride_c == 1,
                errors::Unimplemented("MaxPooling3dGradGrad is not yet "
                                      "supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling3d, tensor_in should have 5 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 5,
                errors::InvalidArgument("tensor_out must be 5-dimensional"));
    // For maxpooling3d, out_grad_backprop should have 5 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 5,
        errors::InvalidArgument("out_grad_backprop must be 5-dimensional"));

    Pool3dParameters params{context,  ksize_,       stride_,
                            padding_, data_format_, tensor_in.shape()};

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {2}, 0, tensor_out.shape(), &output));

    LaunchMaxPooling3dGradGradOp<Device, T>::launch(
        context, params, tensor_in, tensor_out, out_grad_backprop, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_KERNELS(D, T)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MaxPool3D").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      Pooling3DOp<D##Device, T, MAX>);                                     \
  REGISTER_KERNEL_BUILDER(Name("MaxPool3DGrad")                            \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<T>("TInput"),                \
                          MaxPooling3dGradOp<D##Device, T>);               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MaxPool3DGradGrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      MaxPooling3dGradGradOp<D##Device, T>);                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("AvgPool3D").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      Pooling3DOp<D##Device, T, AVG>);                                     \
  REGISTER_KERNEL_BUILDER(Name("AvgPool3DGrad")                            \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("orig_input_shape"),             \
                          AvgPooling3dGradOp<D##Device, T>);

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T)
TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA

template <typename T>
struct LaunchPoolingOp<GPUDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    DnnPooling3dOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kAverage, window,
        stride, padding, data_format, tensor_in, output);
  }
};

template <typename T>
struct LaunchPoolingOp<GPUDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    DnnPooling3dOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kMaximum, window,
        stride, padding, data_format, tensor_in, output);
  }
};

template <typename T>
struct LaunchMaxPooling3dGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* input_backprop) {
    const TensorShape output_shape = tensor_in.shape();
    DnnPooling3dGradOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kMaximum, window,
        stride, padding, out, data_format, out_backprop, output_shape,
        &tensor_in, &tensor_out, input_backprop);
  }
};

template <typename T>
struct LaunchAvgPooling3dGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    DnnPooling3dGradOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kAverage, window,
        stride, padding, out, data_format, out_backprop, tensor_in_shape,
        nullptr, nullptr, output);
  }
};

template <typename T>
struct LaunchMaxPooling3dGradGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context, const Pool3dParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& tensor_top_diff,
                     Tensor* tensor_bottom_diff) {
    bool status = functor::MaxPool3dGradBackward<T>()(
        params.data_format, tensor_in.flat<T>().data(),
        tensor_out.flat<T>().data(), params.tensor_in_batch, params.out_plane,
        params.out_height, params.out_width, params.depth,
        params.tensor_in_planes, params.tensor_in_rows, params.tensor_in_cols,
        params.window_planes, params.window_rows, params.window_cols,
        params.plane_stride, params.row_stride, params.col_stride,
        params.pad_planes, params.pad_rows, params.pad_cols,
        tensor_top_diff.flat<T>().data(), tensor_bottom_diff->flat<T>().data(),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPool3dGradBackward"));
    }
  }
};

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_float(REGISTER_GPU_KERNELS) TF_CALL_half(REGISTER_GPU_KERNELS)
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// Helper struct to contain the various pool parameters used in the SYCL
// pooling kernels. Similar to the Pool3dParameters, but with a number of
// convenient constructors.
struct SYCL3DPoolParams {
  SYCL3DPoolParams(const int depth, const int batch, const int in_planes,
                   const int in_rows, const int in_cols,
                   const std::array<int64, 3>& out_shape,
                   const std::array<int64, 3>& window,
                   const std::array<int64, 3>& stride,
                   const std::array<int64, 3>& padding)
      : depth_(depth),
        batch_(batch),
        in_planes_(in_planes),
        in_rows_(in_rows),
        in_cols_(in_cols),
        window_planes_(window[2]),
        window_rows_(window[1]),
        window_cols_(window[0]),
        stride_planes_(stride[2]),
        stride_rows_(stride[1]),
        stride_cols_(stride[0]),
        out_planes_(out_shape[2]),
        out_rows_(out_shape[1]),
        out_cols_(out_shape[0]),
        pad_planes_(padding[2]),
        pad_rows_(padding[1]),
        pad_cols_(padding[0]) {}

  SYCL3DPoolParams(const int depth, const int batch, const int in_planes,
                   const int in_rows, const int in_cols, const int out_planes,
                   const int out_rows, const int out_cols,
                   const std::array<int64, 3>& window,
                   const std::array<int64, 3>& stride,
                   const std::array<int64, 3>& padding)
      : depth_(depth),
        batch_(batch),
        in_planes_(in_planes),
        in_rows_(in_rows),
        in_cols_(in_cols),
        window_planes_(window[2]),
        window_rows_(window[1]),
        window_cols_(window[0]),
        stride_planes_(stride[2]),
        stride_rows_(stride[1]),
        stride_cols_(stride[0]),
        out_planes_(out_planes),
        out_rows_(out_rows),
        out_cols_(out_cols),
        pad_planes_(padding[2]),
        pad_rows_(padding[1]),
        pad_cols_(padding[0]) {}

  SYCL3DPoolParams(const Pool3dParameters& params)
      : depth_(params.depth),
        batch_(params.tensor_in_batch),
        in_planes_(params.tensor_in_planes),
        in_rows_(params.tensor_in_rows),
        in_cols_(params.tensor_in_cols),
        window_planes_(params.window_planes),
        window_rows_(params.window_rows),
        window_cols_(params.window_cols),
        stride_planes_(params.plane_stride),
        stride_rows_(params.row_stride),
        stride_cols_(params.col_stride),
        out_planes_(params.out_plane),
        out_rows_(params.out_height),
        out_cols_(params.out_width),
        pad_planes_(params.pad_planes),
        pad_rows_(params.pad_rows),
        pad_cols_(params.pad_cols) {}

  const int depth_;
  const int batch_;
  const int in_planes_;
  const int in_rows_;
  const int in_cols_;

  const int window_planes_;
  const int window_rows_;
  const int window_cols_;

  const int stride_planes_;
  const int stride_rows_;
  const int stride_cols_;

  const int out_planes_;
  const int out_rows_;
  const int out_cols_;

  const int pad_planes_;
  const int pad_rows_;
  const int pad_cols_;
};
// MaxPool3d SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output tensor.
//
// For each output element, find the corresponding input window and run over
// all values in the window to find the maximum value. This value is then
// copied into that output element.
template <typename T>
class MaxPool3DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool3DSYCL(const int depth, const int batch, const int in_planes,
              const int in_rows, const int in_cols, const int out_planes,
              const int out_rows, const int out_cols,
              const std::array<int64, 3>& window,
              const std::array<int64, 3>& stride,
              const std::array<int64, 3>& padding,
              const read_accessor input_accessor,
              write_accessor output_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, out_planes, out_rows,
           out_cols, window, stride, padding),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    T maxval = Eigen::NumTraits<T>::lowest();
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend; ++p) {
      for (int r = rstart; r < rend; ++r) {
        for (int c = cstart; c < cend; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          if (input_data_n[idx] > maxval) {
            maxval = input_data_n[idx];
          }
        }
      }
    }
    output_data[index] = maxval;
  }

 private:
  const SYCL3DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct LaunchPoolingOp<SYCLDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int out_planes = GetTensorDim(*output, data_format, '0');
    const int out_rows = GetTensorDim(*output, data_format, '1');
    const int out_cols = GetTensorDim(*output, data_format, '2');
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in, data_format, '2');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int num_threads = output->NumElements();

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPool3DSYCL<T> max_pool(depth, batch, in_planes, in_rows, in_cols,
                              out_planes, out_rows, out_cols, window, stride,
                              padding, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), max_pool);
    });
  }
};

// Need an atomic add for the MaxPool3DGrad kernel. For the device, this need a
// pointer to global memory which isn't understood by the host. The host should
// never be calling this method, but we provide the header so that the host
// compiler can compile the MaxPool3DGrad functor.
#ifdef __SYCL_DEVICE_ONLY__
template <typename T>
void SyclAtomicAdd(__attribute__((address_space(1))) T* address,
                   const T increment);
#else
template <typename T>
void SyclAtomicAdd(T* address, const T increment);
#endif  // __SYCL_DEVICE_ONLY__
// MaxPool3DGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the pooled output tensor (i.e. the number of elements
// in the backprop input tensor).
//
// For each gradient in the input backprop tensor we compare the input data to
// the output data to find the max value in the input window. This gradient is
// then added to the corresponding output gradient. We need to perform the
// addition atomically as a single value may be the maximum of a number of
// input windows.
template <typename T>
class MaxPool3DGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool3DGradSYCL(const int depth, const int batch, const int in_planes,
                  const int in_rows, const int in_cols,
                  const std::array<int64, 3>& output_shape,
                  const std::array<int64, 3>& window,
                  const std::array<int64, 3>& stride,
                  const std::array<int64, 3>& padding,
                  const read_accessor input_data_accessor,
                  const read_accessor output_data_accessor,
                  const read_accessor input_backprop_accessor,
                  write_accessor output_backprop_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, output_shape, window, stride,
           padding),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    int maxidx = -1;
    bool should_stop = false;
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int r = rstart; r < rend && !should_stop; ++r) {
        for (int c = cstart; c < cend && !should_stop; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          if (output_data[index] == input_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    if (maxidx != -1) {
      SyclAtomicAdd(
          output_backprop +
              n * p_.in_planes_ * p_.in_rows_ * p_.in_cols_ * p_.depth_ +
              maxidx,
          input_backprop[index]);
    }
  }

 private:
  const SYCL3DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPooling3dGradOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    output->template flat<T>().setZero().device(device);
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in, data_format, '2');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int output_size = out_backprop.NumElements();

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
      MaxPool3DGradSYCL<T> max_pool(
          depth, batch, in_planes, in_rows, in_cols, out, window, stride,
          padding, input_data_access, output_data_access, input_backprop_access,
          output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(output_size), max_pool);
    });
  }
};
#ifdef __SYCL_DEVICE_ONLY__
// Use the OpenCL atomic uint operations to provide a floating point atomic add.
// For the device we use the atomic compare-exchange builtin to keep trying to
// add to the memory in a thread safe way. The union is needed as these
// builtins are not availble for floating point types, only integer types, so
// we do the addition on the float and the memory update on the uint.
//
// TODO(jwlawson): Remove once we have different type accessors for SYCL buffers
// Providing a way to cast the types of buffers or accessors has been proposed
// as a SYCL extension, so once this is available we can use and atomic
// accessor and remove this.
template <>
void SyclAtomicAdd<float>(__attribute__((address_space(1))) float* address,
                          const float increment) {
  union {
    uint32_t u32;
    float f32;
  } next, expected, current;
  current.f32 = *address;
  __attribute__((address_space(1))) uint32_t* uint_addr =
      reinterpret_cast<__attribute__((address_space(1))) uint32_t*>(address);
  do {
    expected.f32 = current.f32;
    next.f32 = expected.f32 + increment;
    current.u32 =
        _Z14atomic_cmpxchgPVU3AS1jjj(uint_addr, expected.u32, next.u32);
  } while (current.u32 != expected.u32);
}
template <>
void SyclAtomicAdd<double>(__attribute__((address_space(1))) double* address,
                           const double increment) {
  union {
    uint64_t u64;
    double d64;
  } next, expected, current;
  current.d64 = *address;
  __attribute__((address_space(1))) uint64_t* uint_addr =
      reinterpret_cast<__attribute__((address_space(1))) uint64_t*>(address);
  do {
    expected.d64 = current.d64;
    next.d64 = expected.d64 + increment;
    current.d64 = _Z12atom_cmpxchgPVU3AS1mmm(uint_addr, expected.u64, next.u64);
  } while (current.u64 != expected.u64);
}
#else
// Provide a dummy implementation for the host compiler. This code will not be
// seen by the SYCL device, and so should not be run.
template <>
void SyclAtomicAdd<float>(float* address, const float increment) {
  LOG(FATAL) << "MaxPool3DGradSYCL should only be run on a SYCL device";
}
template <>
void SyclAtomicAdd<double>(double* address, const double increment) {
  LOG(FATAL) << "MaxPool3DGradSYCL should only be run on a SYCL device";
}
#endif  // __SYCL_DEVICE_ONLY__
// MaxPool3DGradGrad SYCL kernel. Expects the number of threads to be equal to
// the number of elements in the output backprop tensor, i.e. the number of
// elements in the output tensor.
//
// For each element in the output backprop tensor, find the corresponding input
// window, and compare the input and output data to find the index of the
// maximum value in the input tensor. This is then the index of the gradient to
// pass through to the output backprop tensor.
template <typename T>
class MaxPool3DGradGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool3DGradGradSYCL(const Pool3dParameters& params,
                      const read_accessor input_data_accessor,
                      const read_accessor output_data_accessor,
                      const read_accessor input_backprop_accessor,
                      write_accessor output_backprop_accessor)
      : p_(params),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    int maxidx = -1;
    bool should_stop = false;
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int r = rstart; r < rend && !should_stop; ++r) {
        for (int c = cstart; c < cend && !should_stop; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          if (output_data[index] == input_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    if (maxidx != -1) {
      output_backprop[index] = input_backprop[n * p_.in_planes_ * p_.in_rows_ *
                                                  p_.in_cols_ * p_.depth_ +
                                              maxidx];
    }
  }

 private:
  const SYCL3DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPooling3dGradGradOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context, const Pool3dParameters& params,
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
      MaxPool3DGradGradSYCL<T> functor(params, input_data_access,
                                     output_data_access, input_backprop_access,
                                     output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    });
  }
};
// AvgPool3D SYCL kernel. Expects the number of threads to be equal to the number of elements in the output tensor.
//
// For each output value find the corresponding input window, and run through
// the window accumulating the values to form an average. We divide each value
// before accumulating to prevent the accumulator from becoming significantly
// bigger than the values we are adding and so decrease any errors.
template <typename T>
class AvgPool3DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  AvgPool3DSYCL(const int depth, const int batch, const int in_planes,
              const int in_rows, const int in_cols, const int out_planes,
              const int out_rows, const int out_cols,
              const std::array<int64, 3>& window,
              const std::array<int64, 3>& stride,
              const std::array<int64, 3>& padding,
              const read_accessor input_accessor,
              write_accessor output_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, out_planes, out_rows,
           out_cols, window, stride, padding),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    T accum = T(0);
    T count =
        static_cast<T>((pend - pstart) * (rend - rstart) * (cend - cstart));
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend; ++p) {
      for (int r = rstart; r < rend; ++r) {
        for (int c = cstart; c < cend; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          accum += input_data_n[idx] / count;
        }
      }
    }
    output_data[index] = accum;
  }

 private:
  const SYCL3DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct LaunchPoolingOp<SYCLDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int out_planes = GetTensorDim(*output, data_format, '0');
    const int out_rows = GetTensorDim(*output, data_format, '1');
    const int out_cols = GetTensorDim(*output, data_format, '2');
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in, data_format, '2');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int num_threads = output->NumElements();

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      AvgPool3DSYCL<T> avg_pool(depth, batch, in_planes, in_rows, in_cols,
                              out_planes, out_rows, out_cols, window, stride,
                              padding, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), avg_pool);
    });
  }
};
// AvgPool3DGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output backprop tensor, i.e. the number of
// elements in the input tensor.
//
// For each output backprop index find a window in the input backprop tensor
// which corresponds to all the values of the output which were affected by the
// input value at this index. Then for each gradient in this window, compute
// the size of the input window which was averaged to give this output, and use
// this size to scale the gradient accordingly. Add this scaled gradient to the
// output backprop value.
template <typename T>
class AvgPool3DGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  AvgPool3DGradSYCL(const int depth, const int batch, const int in_planes,
                  const int in_rows, const int in_cols,
                  const std::array<int64, 3>& out_shape,
                  const std::array<int64, 3>& window,
                  const std::array<int64, 3>& stride,
                  const std::array<int64, 3>& padding,
                  const read_accessor input_backprop_accessor,
                  write_accessor output_backprop_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, out_shape, window, stride,
           padding),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const int index = item.get_linear_id();
    int n = index;
    const int d = n % p_.depth_;
    n /= p_.depth_;
    const int c = (n % p_.in_cols_) + p_.pad_cols_;
    const int poolcstart =
        (c < p_.window_cols_) ? 0 : (c - p_.window_cols_) / p_.stride_cols_ + 1;
    const int poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
    n /= p_.in_cols_;
    const int r = (n % p_.in_rows_) + p_.pad_rows_;
    const int poolrstart =
        (r < p_.window_rows_) ? 0 : (r - p_.window_rows_) / p_.stride_rows_ + 1;
    const int poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
    n /= p_.in_rows_;
    const int p = (n % p_.in_planes_) + p_.pad_planes_;
    const int poolpstart =
        (p < p_.window_planes_)
            ? 0
            : (p - p_.window_planes_) / p_.stride_planes_ + 1;
    const int poolpend = std::min(p / p_.stride_planes_ + 1, p_.out_planes_);
    n /= p_.in_planes_;

    T gradient = T(0);
    const T* input_backprop_n =
        input_backprop +
        n * p_.out_planes_ * p_.out_cols_ * p_.out_rows_ * p_.depth_;
    for (int poolp = poolpstart; poolp < poolpend; ++poolp) {
      int pstart = poolp * p_.stride_planes_ - p_.pad_planes_;
      const int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
      pstart = std::max(pstart, 0);
      const int plane_window_size = pend - pstart;
      for (int poolr = poolrstart; poolr < poolrend; ++poolr) {
        int rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
        const int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
        rstart = std::max(rstart, 0);
        const int row_window_size = rend - rstart;
        for (int poolc = poolcstart; poolc < poolcend; ++poolc) {
          const int idx =
              ((poolp * p_.out_rows_ + poolr) * p_.out_cols_ + poolc) *
                  p_.depth_ +
              d;
          int cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
          const int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
          cstart = std::max(cstart, 0);
          const int col_window_size = cend - cstart;
          const int window_size =
              plane_window_size * row_window_size * col_window_size;
          gradient += input_backprop_n[idx] / static_cast<T>(window_size);
        }
      }
    }
    output_backprop[index] = gradient;
  }

 private:
  const SYCL3DPoolParams p_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchAvgPooling3dGradOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& output_shape,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    output->template flat<T>().setZero().device(device);
    const int batch = GetTensorDim(tensor_in_shape, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
    const int depth = GetTensorDim(tensor_in_shape, data_format, 'C');

    const int num_threads = output->NumElements();

    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      AvgPool3DGradSYCL<T> functor(depth, batch, in_planes, in_rows, in_cols,
                                 output_shape, window, stride, padding,
                                 input_backprop_access, output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    });
  }
};
#define REGISTER_SYCL_KERNELS(T) REGISTER_KERNELS(SYCL, T)
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNELS)
#undef REGISTER_SYCL_KERNELS
#endif  // TENSORFLOW_USE_SYCL

#undef REGISTER_KERNELS

}  // namespace tensorflow
