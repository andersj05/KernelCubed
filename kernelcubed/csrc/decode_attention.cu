#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>

namespace {

constexpr int kHeadDim = 128;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void qwen_decode_attention_warp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    scalar_t* __restrict__ output,
    int sequence_length,
    int query_heads,
    int kv_heads,
    float scale) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kValuesPerLane = kHeadDim / kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_head = blockIdx.x * kWarpsPerBlock + warp;
  if (query_head >= query_heads) {
    return;
  }

  const int group_size = query_heads / kv_heads;
  const int kv_head = query_head / group_size;
  float query_values[kValuesPerLane];
  float accumulators[kValuesPerLane] = {0.0f};
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const int dimension = lane + index * kWarpSize;
    query_values[index] = static_cast<float>(
        query[query_head * kHeadDim + dimension]);
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  for (int token = 0; token < sequence_length; ++token) {
    const int cache_base = (token * kv_heads + kv_head) * kHeadDim;
    float dot = 0.0f;
#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dimension = lane + index * kWarpSize;
      dot += query_values[index] *
          static_cast<float>(key_cache[cache_base + dimension]);
    }
    dot = warp_sum(dot);

    float old_factor = 0.0f;
    float new_factor = 0.0f;
    if (lane == 0) {
      const float score = dot * scale;
      const float next_max = fmaxf(running_max, score);
      old_factor = running_max == -FLT_MAX
          ? 0.0f
          : __expf(running_max - next_max);
      new_factor = __expf(score - next_max);
      running_sum = running_sum * old_factor + new_factor;
      running_max = next_max;
    }
    old_factor = __shfl_sync(0xffffffff, old_factor, 0);
    new_factor = __shfl_sync(0xffffffff, new_factor, 0);

#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dimension = lane + index * kWarpSize;
      accumulators[index] = accumulators[index] * old_factor +
          static_cast<float>(value_cache[cache_base + dimension]) *
              new_factor;
    }
  }

  float inverse_sum = lane == 0 ? 1.0f / running_sum : 0.0f;
  inverse_sum = __shfl_sync(0xffffffff, inverse_sum, 0);
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const int dimension = lane + index * kWarpSize;
    output[query_head * kHeadDim + dimension] =
        static_cast<scalar_t>(accumulators[index] * inverse_sum);
  }
}

template <typename scalar_t>
__global__ void qwen_decode_attention_split_gqa2_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    float* __restrict__ partials,
    int sequence_length,
    int query_heads,
    int kv_heads,
    int partitions,
    float scale) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kValuesPerLane = kHeadDim / kWarpSize;
  constexpr int kStateSize = kHeadDim + 2;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int kv_head = blockIdx.x * kWarpsPerBlock + warp;
  const int partition = blockIdx.y;
  const int query_head_0 = kv_head * 2;
  const int query_head_1 = query_head_0 + 1;
  if (kv_head >= kv_heads || query_head_1 >= query_heads) {
    return;
  }

  const int tokens_per_partition =
      (sequence_length + partitions - 1) / partitions;
  const int token_begin = partition * tokens_per_partition;
  const int token_end =
      min(sequence_length, token_begin + tokens_per_partition);
  float query_values_0[kValuesPerLane];
  float query_values_1[kValuesPerLane];
  float accumulators_0[kValuesPerLane] = {0.0f};
  float accumulators_1[kValuesPerLane] = {0.0f};
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const int dimension = lane + index * kWarpSize;
    query_values_0[index] = static_cast<float>(
        query[query_head_0 * kHeadDim + dimension]);
    query_values_1[index] = static_cast<float>(
        query[query_head_1 * kHeadDim + dimension]);
  }

  float running_max_0 = -FLT_MAX;
  float running_max_1 = -FLT_MAX;
  float running_sum_0 = 0.0f;
  float running_sum_1 = 0.0f;
  for (int token = token_begin; token < token_end; ++token) {
    const int cache_base = (token * kv_heads + kv_head) * kHeadDim;
    float dot_0 = 0.0f;
    float dot_1 = 0.0f;
#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dimension = lane + index * kWarpSize;
      const float key_value =
          static_cast<float>(key_cache[cache_base + dimension]);
      dot_0 += query_values_0[index] * key_value;
      dot_1 += query_values_1[index] * key_value;
    }
    dot_0 = warp_sum(dot_0);
    dot_1 = warp_sum(dot_1);

    float old_factor_0 = 0.0f;
    float old_factor_1 = 0.0f;
    float new_factor_0 = 0.0f;
    float new_factor_1 = 0.0f;
    if (lane == 0) {
      const float score_0 = dot_0 * scale;
      const float score_1 = dot_1 * scale;
      const float next_max_0 = fmaxf(running_max_0, score_0);
      const float next_max_1 = fmaxf(running_max_1, score_1);
      old_factor_0 = running_max_0 == -FLT_MAX
          ? 0.0f
          : __expf(running_max_0 - next_max_0);
      old_factor_1 = running_max_1 == -FLT_MAX
          ? 0.0f
          : __expf(running_max_1 - next_max_1);
      new_factor_0 = __expf(score_0 - next_max_0);
      new_factor_1 = __expf(score_1 - next_max_1);
      running_sum_0 = running_sum_0 * old_factor_0 + new_factor_0;
      running_sum_1 = running_sum_1 * old_factor_1 + new_factor_1;
      running_max_0 = next_max_0;
      running_max_1 = next_max_1;
    }
    old_factor_0 = __shfl_sync(0xffffffff, old_factor_0, 0);
    old_factor_1 = __shfl_sync(0xffffffff, old_factor_1, 0);
    new_factor_0 = __shfl_sync(0xffffffff, new_factor_0, 0);
    new_factor_1 = __shfl_sync(0xffffffff, new_factor_1, 0);

#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dimension = lane + index * kWarpSize;
      const float value =
          static_cast<float>(value_cache[cache_base + dimension]);
      accumulators_0[index] =
          accumulators_0[index] * old_factor_0 + value * new_factor_0;
      accumulators_1[index] =
          accumulators_1[index] * old_factor_1 + value * new_factor_1;
    }
  }

  const int state_base_0 =
      (partition * query_heads + query_head_0) * kStateSize;
  const int state_base_1 =
      (partition * query_heads + query_head_1) * kStateSize;
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const int dimension = lane + index * kWarpSize;
    partials[state_base_0 + dimension] = accumulators_0[index];
    partials[state_base_1 + dimension] = accumulators_1[index];
  }
  if (lane == 0) {
    partials[state_base_0 + kHeadDim] = running_max_0;
    partials[state_base_0 + kHeadDim + 1] = running_sum_0;
    partials[state_base_1 + kHeadDim] = running_max_1;
    partials[state_base_1 + kHeadDim + 1] = running_sum_1;
  }
}

template <typename scalar_t>
__global__ void qwen_decode_attention_merge_kernel(
    const float* __restrict__ partials,
    scalar_t* __restrict__ output,
    int query_heads,
    int partitions) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kValuesPerLane = kHeadDim / kWarpSize;
  constexpr int kStateSize = kHeadDim + 2;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_head = blockIdx.x * kWarpsPerBlock + warp;
  if (query_head >= query_heads) {
    return;
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float accumulators[kValuesPerLane] = {0.0f};
  for (int partition = 0; partition < partitions; ++partition) {
    const int state_base =
        (partition * query_heads + query_head) * kStateSize;
    float old_factor = 0.0f;
    float new_factor = 0.0f;
    if (lane == 0) {
      const float partial_max = partials[state_base + kHeadDim];
      const float partial_sum = partials[state_base + kHeadDim + 1];
      const float next_max = fmaxf(running_max, partial_max);
      old_factor = running_max == -FLT_MAX
          ? 0.0f
          : __expf(running_max - next_max);
      new_factor = __expf(partial_max - next_max);
      running_sum =
          running_sum * old_factor + partial_sum * new_factor;
      running_max = next_max;
    }
    old_factor = __shfl_sync(0xffffffff, old_factor, 0);
    new_factor = __shfl_sync(0xffffffff, new_factor, 0);

#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dimension = lane + index * kWarpSize;
      accumulators[index] = accumulators[index] * old_factor +
          partials[state_base + dimension] * new_factor;
    }
  }

  float inverse_sum = lane == 0 ? 1.0f / running_sum : 0.0f;
  inverse_sum = __shfl_sync(0xffffffff, inverse_sum, 0);
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const int dimension = lane + index * kWarpSize;
    output[query_head * kHeadDim + dimension] =
        static_cast<scalar_t>(accumulators[index] * inverse_sum);
  }
}

template <typename scalar_t>
void launch_qwen_decode_attention(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    torch::Tensor& output,
    const torch::Tensor& workspace,
    float scale) {
  const int sequence_length = static_cast<int>(key_cache.size(0));
  const int query_heads = static_cast<int>(query.size(1));
  const int kv_heads = static_cast<int>(key_cache.size(1));
  constexpr int kTokensPerPartition = 128;
  constexpr int kMaximumPartitions = 64;
  const dim3 head_grid((query_heads + 3) / 4);
  const dim3 kv_head_grid((kv_heads + 3) / 4);
  const dim3 block(kHeadDim);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (sequence_length <= 256) {
    qwen_decode_attention_warp_kernel<scalar_t>
        <<<head_grid, block, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key_cache.const_data_ptr<scalar_t>(),
            value_cache.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            sequence_length,
            query_heads,
            kv_heads,
            scale);
  } else {
    const int requested_partitions =
        (sequence_length + kTokensPerPartition - 1) / kTokensPerPartition;
    const int partitions = requested_partitions < kMaximumPartitions
        ? requested_partitions
        : kMaximumPartitions;
    const dim3 split_grid(kv_head_grid.x, partitions);
    qwen_decode_attention_split_gqa2_kernel<scalar_t>
        <<<split_grid, block, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key_cache.const_data_ptr<scalar_t>(),
            value_cache.const_data_ptr<scalar_t>(),
            workspace.mutable_data_ptr<float>(),
            sequence_length,
            query_heads,
            kv_heads,
            partitions,
            scale);
    qwen_decode_attention_merge_kernel<scalar_t>
        <<<head_grid, block, 0, stream>>>(
            workspace.const_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            query_heads,
            partitions);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor qwen_decode_attention_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& workspace,
    double scale) {
  auto output = torch::empty_like(query);
  if (query.scalar_type() == at::ScalarType::Half) {
    launch_qwen_decode_attention<at::Half>(
        query,
        key_cache,
        value_cache,
        output,
        workspace,
        static_cast<float>(scale));
  } else {
    launch_qwen_decode_attention<at::BFloat16>(
        query,
        key_cache,
        value_cache,
        output,
        workspace,
        static_cast<float>(scale));
  }
  return output;
}
