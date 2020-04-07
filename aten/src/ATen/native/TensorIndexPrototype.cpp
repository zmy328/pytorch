#include <ATen/native/TensorIndexPrototype.h>

#include <map>
#include <numeric>
#include <iostream>
#include <vector>
#include <string>

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

DEFINE_DISPATCH(gather_new_stub);

void resize_result(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index, bool broadcast_index = false){
    if (!broadcast_index){
        result.resize_(index.sizes());
        return;
    }

    auto sizes = self.sizes().vec();
    dim = maybe_wrap_dim(dim, self.dim());
    sizes[dim] = index.numel();
    result.resize_(sizes);
}

Tensor& gather_out_cpu_new_(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index,
                            bool sparse_grad, bool broadcast_index = false) {
  resize_result(result, self, dim, index, broadcast_index);
  gather_new_stub(result.device().type(), result, self, dim, index, broadcast_index);
  return result;
}

Tensor gather_cpu_new_(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  Tensor result = at::empty({0}, self.options());
  return gather_out_cpu_new_(result, self, dim, index, sparse_grad);
}

Tensor index_select_out_cpu_new_(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index){
    return gather_out_cpu_new_(result, self, dim, index, /*sparse_grad=*/false, /*broadcast_index=*/true);
}

Tensor index_select_cpu_new_(const Tensor & self, int64_t dim, const Tensor & index){
    Tensor result = at::empty({0}, self.options());
    return index_select_out_cpu_new_(result, self, dim, index);
}

}} // namespace at::native
