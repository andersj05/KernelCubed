#include <torch/extension.h>

#include <cmath>

torch::Tensor qwen_decode_attention_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    double scale);

torch::Tensor qwen_decode_attention(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    double scale) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
  TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
  TORCH_CHECK(
      query.dim() == 3, "query must have shape [1, query_heads, 128]");
  TORCH_CHECK(
      key_cache.dim() == 3,
      "key_cache must have shape [tokens, kv_heads, 128]");
  TORCH_CHECK(
      value_cache.dim() == 3,
      "value_cache must have shape [tokens, kv_heads, 128]");
  TORCH_CHECK(query.size(0) == 1, "only single-token decode is supported");
  TORCH_CHECK(query.size(2) == 128, "this prototype requires head_dim=128");
  TORCH_CHECK(
      key_cache.sizes() == value_cache.sizes(),
      "key and value cache shapes must match");
  TORCH_CHECK(
      key_cache.size(0) > 0, "KV cache must contain at least one token");
  TORCH_CHECK(
      key_cache.size(2) == 128, "this prototype requires head_dim=128");
  TORCH_CHECK(
      query.size(1) % key_cache.size(1) == 0,
      "query_heads must be divisible by kv_heads");
  TORCH_CHECK(
      query.size(1) == 2 * key_cache.size(1),
      "this Qwen prototype requires two query heads per KV head");
  TORCH_CHECK(
      query.scalar_type() == key_cache.scalar_type(),
      "query and key cache dtypes must match");
  TORCH_CHECK(
      query.scalar_type() == value_cache.scalar_type(),
      "query and value cache dtypes must match");
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Half ||
          query.scalar_type() == at::ScalarType::BFloat16,
      "only float16 and bfloat16 are supported");
  TORCH_CHECK(
      query.get_device() == key_cache.get_device() &&
          query.get_device() == value_cache.get_device(),
      "query, key cache, and value cache must be on the same GPU");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");

  return qwen_decode_attention_cuda(query, key_cache, value_cache, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "decode_attention",
      &qwen_decode_attention,
      "Qwen head_dim=128 grouped-query decode attention (CUDA)",
      py::arg("query"),
      py::arg("key_cache"),
      py::arg("value_cache"),
      py::arg("scale"));
}
