//#include <sstream>
//#include <TH/TH.h>
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/CUDAGeneratorImpl.h>
//#include <c10/cuda/CUDAFunctions.h>
//#include <c10/cuda/CUDACachingAllocator.h>

//#include <torch/csrc/cuda/THCP.h>
//#include <torch/csrc/CudaIPCTypes.h>
//#include <torch/csrc/utils/pybind.h>
//#include <torch/csrc/utils/cuda_lazy_init.h>
//#include <torch/csrc/utils/python_strings.h>
//#include <torch/csrc/cuda/python_comm.h>
//#include <torch/csrc/Generator.h>
//#include <torch/csrc/python_headers.h>

#include <torch/torch.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "op.h"
#include <iostream>


//void cuda_mem_prefetch_async(torch::List<torch::Tensor> tensors, PyObject *obj) {
//  HANDLE_TH_ERRORS
//  THPUtils_assert(PyLong_Check(obj), "invalid stream");
//  uint64_t bits = PyLong_AsUnsignedLongLong(obj);
//  if (bits == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
//    throw python_error();
//  }
//  auto stream = at::cuda::CUDAStream::unpack(bits);
//  auto device = static_cast<int>(c10::cuda::current_device());
//  if (device != stream.device_index()) {
//    THCPModule_setDevice(stream.device_index());
//  }
//  std::cout << "device: " << device << std::endl; 
//  
//  for (int i = 0; i < tensors.size(); i++) {
//    auto tensor = tensors.get(i);
//    std::cout << tensor << std::endl;
//    std::cout << tensor.dim() << std::endl;
//    std::cout << tensor.nbytes() << std::endl;
//    cudaMemPrefetchAsync(&tensor, tensor.nbytes(), device, stream);
//  }
//}

torch::List<torch::Tensor> cuda_mem_prefetch_async(torch::List<torch::Tensor> tensors) {
  int total_bytes;

  for (int i = 0; i < tensors.size(); i++) {
    auto tensor = tensors.get(i);
    //std::cout << "tensor: " << tensor << std::endl;
    //std::cout << "dim: " << tensor.dim() << std::endl;
    //std::cout << "nbytes: " << tensor.nbytes() << std::endl;
    total_bytes += tensor.nbytes();
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMemPrefetchAsync(&tensors, total_bytes, 1, stream);
  cudaStreamDestroy(stream);

  return tensors;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("cuda_mem_prefetch_async", cuda_mem_prefetch_async);
}