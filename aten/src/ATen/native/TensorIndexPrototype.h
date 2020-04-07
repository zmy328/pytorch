#pragma once

// Unify index_* and scatter/gather implementations.
//   Currently very much in the prototype stage.

#include <string>

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using gather_fn = void (*)(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool check_shape_and_dim);
DECLARE_DISPATCH(gather_fn, gather_new_stub);

}} // namespace at::native
