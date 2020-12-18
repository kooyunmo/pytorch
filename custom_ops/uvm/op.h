#include <torch/script.h>

// clang-format off
#  if defined(_WIN32)
#    if defined(custom_ops_EXPORTS)
#      define CUSTOM_OP_API __declspec(dllexport)
#    else
#      define CUSTOM_OP_API __declspec(dllimport)
#    endif
#  else
#    define CUSTOM_OP_API
#  endif
// clang-format on

CUSTOM_OP_API torch::List<torch::Tensor> cuda_mem_prefetch_async(torch::List<torch::Tensor> tensors);