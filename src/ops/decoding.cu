/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/decoding.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

template<int BLOCK_SIZE>
__global__ void softmax_argmax_kernel_sharded(
  const half* __restrict__ input,
  half* __restrict__ output,
  float* __restrict__ max_buffer,      // [seq_len] for allreduce
  int* __restrict__ max_idx_buffer,    // [seq_len] for argmax indices
  float* __restrict__ sum_buffer,      // [seq_len] for allreduce
  const int vocab_size_per_shard,
  const int vocab_offset,              // Starting vocab index for this shard
  const int seq_len) {
  
  const int seq_pos = blockIdx.x;
  if (seq_pos >= seq_len) return;
  
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int num_warps = BLOCK_SIZE / 32;
  
  extern __shared__ char shared_mem_bytes[];
  float* warp_max = (float*)shared_mem_bytes;
  int* warp_max_idx = (int*)(warp_max + num_warps);
  float* warp_sum = (float*)(warp_max_idx + num_warps);
  
  // Phase 1: Find local max and its index
  float thread_max = -INFINITY;
  int thread_max_idx = 0;
  
  for (int i = tid; i < vocab_size_per_shard; i += BLOCK_SIZE) {
      float val = __half2float(input[seq_pos * vocab_size_per_shard + i]);
      if (val > thread_max) {
          thread_max = val;
          thread_max_idx = vocab_offset + i;  // Global vocab index
      }
  }
  
  // Warp reduction for max with index
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
      float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
      int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
      
      if (other_max > thread_max || (other_max == thread_max && other_idx < thread_max_idx)) {
          thread_max = other_max;
          thread_max_idx = other_idx;
      }
  }
  
  if (lane_id == 0) {
      warp_max[warp_id] = thread_max;
      warp_max_idx[warp_id] = thread_max_idx;
  }
  __syncthreads();
  
  // Final reduction across warps
  if (tid < num_warps) {
      thread_max = warp_max[tid];
      thread_max_idx = warp_max_idx[tid];
  } else {
      thread_max = -INFINITY;
      thread_max_idx = INT_MAX;
  }
  
  if (warp_id == 0) {
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
          float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
          int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
          
          if (other_max > thread_max || (other_max == thread_max && other_idx < thread_max_idx)) {
              thread_max = other_max;
              thread_max_idx = other_idx;
          }
      }
      
      if (lane_id == 0) {
          max_buffer[seq_pos] = thread_max;
          max_idx_buffer[seq_pos] = thread_max_idx;
      }
  }
}

// Custom reduction operation for max with index
__device__ void reduce_max_with_idx(float* max_val, int* max_idx, float other_val, int other_idx) {
    if (other_val > *max_val || (other_val == *max_val && other_idx < *max_idx)) {
        *max_val = other_val;
        *max_idx = other_idx;
    }
}

template<int BLOCK_SIZE>
__global__ void softmax_compute_kernel_sharded(
  const half* __restrict__ input,
  half* __restrict__ output,
  const float* __restrict__ max_buffer,  // Global max from allreduce
  float* __restrict__ sum_buffer,
  const int vocab_size_per_shard,
  const int seq_len) {
  
  const int seq_pos = blockIdx.x;
  if (seq_pos >= seq_len) return;
  
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int num_warps = BLOCK_SIZE / 32;
  
  extern __shared__ float warp_sum[];
  
  float max_val = max_buffer[seq_pos];
  float thread_sum = 0.0f;
  
  // Compute exp(x - max) and local sum
  for (int i = tid; i < vocab_size_per_shard; i += BLOCK_SIZE) {
      float val = __half2float(input[seq_pos * vocab_size_per_shard + i]);
      float exp_val = expf(val - max_val);
      thread_sum += exp_val;
      output[seq_pos * vocab_size_per_shard + i] = __float2half(exp_val);
  }
  
  // Warp reduction for sum
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
      thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }
  
  if (lane_id == 0) warp_sum[warp_id] = thread_sum;
  __syncthreads();
  
  // Final reduction
  if (tid < num_warps) thread_sum = warp_sum[tid];
  else thread_sum = 0.0f;
  
  if (warp_id == 0) {
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
          thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
      }
      if (lane_id == 0) sum_buffer[seq_pos] = thread_sum;
  }
}

template<int BLOCK_SIZE>
__global__ void softmax_normalize_kernel_sharded(
  half* __restrict__ output,
  const float* __restrict__ sum_buffer,  // Global sum from allreduce
  const int vocab_size_per_shard,
  const int seq_len) {
  
  const int seq_pos = blockIdx.x;
  if (seq_pos >= seq_len) return;
  
  const int tid = threadIdx.x;
  float inv_sum = 1.0f / sum_buffer[seq_pos];
  
  // Normalize
  for (int i = tid; i < vocab_size_per_shard; i += BLOCK_SIZE) {
      float exp_val = __half2float(output[seq_pos * vocab_size_per_shard + i]);
      output[seq_pos * vocab_size_per_shard + i] = __float2half(exp_val * inv_sum);
  }
}

// Kernel to handle custom reduction for max with argmax
__global__ void reduce_max_with_argmax_kernel(
  float* max_buffer,
  int* max_idx_buffer,
  int seq_len,
  int n_shards) {
  
  int seq_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq_pos >= seq_len) return;
  
  // Each thread handles one sequence position
  // The buffers contain [shard0_val, shard1_val, ...] for each sequence
  float global_max = max_buffer[seq_pos * n_shards];
  int global_idx = max_idx_buffer[seq_pos * n_shards];
  
  for (int shard = 1; shard < n_shards; shard++) {
      float shard_max = max_buffer[seq_pos * n_shards + shard];
      int shard_idx = max_idx_buffer[seq_pos * n_shards + shard];
      
      if (shard_max > global_max || (shard_max == global_max && shard_idx < global_idx)) {
          global_max = shard_max;
          global_idx = shard_idx;
      }
  }
  
  // Write back the global max and index
  max_buffer[seq_pos] = global_max;
  max_idx_buffer[seq_pos] = global_idx;
}

struct SoftmaxShardedContext {
    float* max_buffer;
    int* max_idx_buffer;
    float* sum_buffer;
    float* all_max_buffer;
    int* all_idx_buffer;
    int seq_len;
    int n_shards;
};

SoftmaxShardedContext* create_softmax_context(int seq_len, ncclComm_t nccl_comm) {
    SoftmaxShardedContext* ctx = new SoftmaxShardedContext;
    ctx->seq_len = seq_len;
    ncclCommCount(nccl_comm, &ctx->n_shards);
    
    cudaMalloc(&ctx->max_buffer, seq_len * sizeof(float));
    cudaMalloc(&ctx->max_idx_buffer, seq_len * sizeof(int));
    cudaMalloc(&ctx->sum_buffer, seq_len * sizeof(float));
    cudaMalloc(&ctx->all_max_buffer, seq_len * ctx->n_shards * sizeof(float));
    cudaMalloc(&ctx->all_idx_buffer, seq_len * ctx->n_shards * sizeof(int));
    
    return ctx;
}

void softmax_argmax_sharded_with_context(
  const half* input,
  half* output,
  int* argmax_indices,
  int vocab_size_per_shard,
  int vocab_offset,
  SoftmaxShardedContext* ctx,
  cudaStream_t stream,
  ncclComm_t nccl_comm) {
  
  const int block_size = 256;
  const int grid_size = ctx->seq_len;
  size_t shared_mem_size = (block_size / 32) * sizeof(float) * 2 + (block_size / 32) * sizeof(int);
  
  // Step 1: Find local max and argmax
  softmax_argmax_kernel_sharded<block_size><<<grid_size, block_size, shared_mem_size, stream>>>(
      input, output, ctx->max_buffer, ctx->max_idx_buffer, ctx->sum_buffer, 
      vocab_size_per_shard, vocab_offset, ctx->seq_len
  );
  
  // Steps 2-5: Same as before but using pre-allocated buffers from context
  ncclAllReduce(ctx->max_buffer, ctx->max_buffer, ctx->seq_len, ncclFloat32, ncclMax, nccl_comm, stream);
  
  ncclAllGather(ctx->max_buffer, ctx->all_max_buffer, ctx->seq_len, ncclFloat32, nccl_comm, stream);
  ncclAllGather(ctx->max_idx_buffer, ctx->all_idx_buffer, ctx->seq_len, ncclInt32, nccl_comm, stream);
  
  int reduce_blocks = (ctx->seq_len + 255) / 256;
  reduce_max_with_argmax_kernel<<<reduce_blocks, 256, 0, stream>>>(
      ctx->all_max_buffer, ctx->all_idx_buffer, ctx->seq_len, ctx->n_shards
  );
  
  cudaMemcpyAsync(argmax_indices, ctx->all_idx_buffer, ctx->seq_len * sizeof(int), 
                  cudaMemcpyDeviceToDevice, stream);
  
  softmax_compute_kernel_sharded<block_size><<<grid_size, block_size, shared_mem_size, stream>>>(
      input, output, ctx->max_buffer, ctx->sum_buffer, vocab_size_per_shard, ctx->seq_len
  );
  
  ncclAllReduce(ctx->sum_buffer, ctx->sum_buffer, ctx->seq_len, ncclFloat32, ncclSum, nccl_comm, stream);
  
  softmax_normalize_kernel_sharded<block_size><<<grid_size, block_size, 0, stream>>>(
      output, ctx->sum_buffer, vocab_size_per_shard, ctx->seq_len
  );
}

void destroy_softmax_context(SoftmaxShardedContext* ctx) {
    cudaFree(ctx->max_buffer);
    cudaFree(ctx->max_idx_buffer);
    cudaFree(ctx->sum_buffer);
    cudaFree(ctx->all_max_buffer);
    cudaFree(ctx->all_idx_buffer);
    delete ctx;
}

// Optimized version using warp primitives
template<int BLOCK_SIZE, typename DT>
__global__ void softmax_argmax_kernel(
  const DT* __restrict__ input,
  DT* __restrict__ output,
  int* __restrict__ argmax_indices,
  const int vocab_size,
  const int seq_len) {
  
  const int seq_pos = blockIdx.x;
  if (seq_pos >= seq_len) return;
  
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int num_warps = BLOCK_SIZE / 32;
  
  // Shared memory
  extern __shared__ char shared_mem_bytes[];
  float* warp_max = (float*)shared_mem_bytes;
  int* warp_max_idx = (int*)(warp_max + num_warps);
  float* warp_sum = (float*)(warp_max_idx + num_warps);
  
  // Phase 1: Find max
  float thread_max = -INFINITY;
  int thread_max_idx = 0;
  
  for (int vocab_idx = tid; vocab_idx < vocab_size; vocab_idx += BLOCK_SIZE) {
      float val;
      if constexpr (std::is_same_v<DT, half>) {
        val = __half2float(input[seq_pos * vocab_size + vocab_idx]);
      } else {
        val = static_cast<float>(input[seq_pos * vocab_size + vocab_idx]);
      }
      if (val > thread_max) {
          thread_max = val;
          thread_max_idx = vocab_idx;
      }
  }
  
  // Warp-level reduction for max
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
      float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
      int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
      
      if (other_max > thread_max || (other_max == thread_max && other_idx < thread_max_idx)) {
          thread_max = other_max;
          thread_max_idx = other_idx;
      }
  }
  
  // Store warp results
  if (lane_id == 0) {
      warp_max[warp_id] = thread_max;
      warp_max_idx[warp_id] = thread_max_idx;
  }
  __syncthreads();
  
  // Final reduction across warps
  if (tid < num_warps) {
      thread_max = warp_max[tid];
      thread_max_idx = warp_max_idx[tid];
  } else {
      thread_max = -INFINITY;
      thread_max_idx = INT_MAX;
  }
  
  if (warp_id == 0) {
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
          float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
          int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
          
          if (other_max > thread_max || (other_max == thread_max && other_idx < thread_max_idx)) {
              thread_max = other_max;
              thread_max_idx = other_idx;
          }
      }
      
      if (lane_id == 0) {
          warp_max[0] = thread_max;
          warp_max_idx[0] = thread_max_idx;
      }
  }
  __syncthreads();
  
  float max_val = warp_max[0];
  int max_idx = warp_max_idx[0];
  
  // Phase 2: Compute exp and sum
  float thread_sum = 0.0f;
  
  for (int vocab_idx = tid; vocab_idx < vocab_size; vocab_idx += BLOCK_SIZE) {
      float val;
      if constexpr (std::is_same_v<DT, half>) {
        val = __half2float(input[seq_pos * vocab_size + vocab_idx]);
      } else {
        val = static_cast<float>(input[seq_pos * vocab_size + vocab_idx]);
      }
      float exp_val = expf(val - max_val);
      thread_sum += exp_val;
      if constexpr (std::is_same_v<DT, half>) {
        output[seq_pos * vocab_size + vocab_idx] = __float2half(exp_val);
      } else {
        output[seq_pos * vocab_size + vocab_idx] = static_cast<DT>(exp_val);
      }
  }
  
  // Warp-level reduction for sum
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
      thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }
  
  if (lane_id == 0) {
      warp_sum[warp_id] = thread_sum;
  }
  __syncthreads();
  
  // Final reduction
  if (tid < num_warps) {
      thread_sum = warp_sum[tid];
  } else {
      thread_sum = 0.0f;
  }
  
  if (warp_id == 0) {
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
          thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
      }
      
      if (lane_id == 0) {
          warp_sum[0] = thread_sum;
      }
  }
  __syncthreads();
  
  float total_sum = warp_sum[0];
  float inv_sum = 1.0f / total_sum;
  
  // Phase 3: Normalize
  for (int vocab_idx = tid; vocab_idx < vocab_size; vocab_idx += BLOCK_SIZE) {
      float exp_val;
      if constexpr (std::is_same_v<DT, half>) {
        exp_val = __half2float(output[seq_pos * vocab_size + vocab_idx]);
        output[seq_pos * vocab_size + vocab_idx] = __float2half(exp_val * inv_sum);
      } else {
        exp_val = static_cast<float>(output[seq_pos * vocab_size + vocab_idx]);
        output[seq_pos * vocab_size + vocab_idx] = static_cast<DT>(exp_val * inv_sum);
      }
  }
  
//   Store argmax
  if (tid == 0) {
    argmax_indices[seq_pos] = max_idx;
  }
}

// Wrapper function template
template<typename DT>
void softmax_argmax(
  const DT* input,
  DT* output,
  int* argmax_indices,
  int vocab_size,
  int seq_len,
  cudaStream_t stream) {

  const int block_size = 256;
  const int grid_size = seq_len;  // One block per sequence position
  
  
  // Use optimized kernel for larger vocabularies
  size_t shared_mem_size_opt = (block_size / 32) * sizeof(float) * 2 + (block_size / 32) * sizeof(int);
  softmax_argmax_kernel<block_size, DT><<<grid_size, block_size, shared_mem_size_opt, stream>>>(
      input, output, argmax_indices, vocab_size, seq_len
  );  
}

// Placeholder kernel implementations
template <typename DT>
void Decoding::inference_kernel(DecodingMeta const *m,
                                BatchConfig const *bc,
                                DT const *input_ptr,
                                DT *softmax_output_ptr,
                                int *argmax_output_ptr,
                                int num_classes,
                                int vocab_offset,
                                float *loss,
                                cudaStream_t stream) {

  // Old non-sharded version (commented out)
  // softmax_argmax(input_ptr, softmax_output_ptr, argmax_output_ptr, num_classes, bc->num_active_tokens(), stream);
  
  // New sharded version - only supports half precision for now
  if constexpr (std::is_same_v<DT, half>) {
    // Use sharded softmax with NCCL communication
    int vocab_size_per_shard = num_classes;  // This is already the local shard size
    
    softmax_argmax_sharded_with_context(
        input_ptr,
        softmax_output_ptr, 
        argmax_output_ptr,
        vocab_size_per_shard,
        vocab_offset,
        m->softmax_context,
        stream,
        m->handle.ncclComm);
  } else {
    // Fall back to non-sharded version for non-half types
    softmax_argmax(input_ptr, softmax_output_ptr, argmax_output_ptr, num_classes, bc->num_active_tokens(), stream);
  }
}

void store_peft_token_ids(DecodingMeta *m, BatchConfig const *bc) {
  assert(peft_finetuning_enabled(m->peft_support_mode));

  int num_ft_tokens = bc->num_finetuning_fwd_tokens();
  int i = bc->finetuning_request_index();
  int tokens_previous_requests =
      bc->requestsInfo[i].first_token_offset_in_batch;
  int prev_steps_tokens = bc->requestsInfo[i].first_token_depth_in_request;
  assert(bc->requestsInfo[i].num_tokens_in_batch == num_ft_tokens);

  // shift labels by 1 position to the left (ignore first token label)
  for (int j = 0; j < num_ft_tokens - 1; j++) {
    m->peft_token_ids[prev_steps_tokens + j] =
        bc->tokensInfo[tokens_previous_requests + j + 1].token_id;
  }
}

template <typename DT>
void store_peft_activations(DecodingMeta *m,
                            BatchConfig const *bc,
                            int num_classes,
                            DT *softmax_output_ptr,
                            cudaStream_t stream) {
  assert(peft_finetuning_enabled(m->peft_support_mode));
  assert(m->output_grad_ptr != nullptr);

  int num_ft_tokens = bc->num_finetuning_fwd_tokens();
  int i = bc->finetuning_request_index();
  int tokens_previous_requests =
      bc->requestsInfo[i].first_token_offset_in_batch;
  int prev_steps_tokens = bc->requestsInfo[i].first_token_depth_in_request;
  assert(bc->requestsInfo[i].num_tokens_in_batch == num_ft_tokens);

  size_t batch_offset = num_classes * tokens_previous_requests;
  size_t req_offset = num_classes * prev_steps_tokens;
  size_t data_size = num_classes * num_ft_tokens * sizeof(DT);
  assert(m->allocated_peft_buffer_size >= data_size);
  checkCUDA(cudaMemcpyAsync(static_cast<DT *>(m->output_grad_ptr) + req_offset,
                            softmax_output_ptr + batch_offset,
                            data_size,
                            cudaMemcpyDeviceToDevice,
                            stream));
}

/*static*/
void Decoding::inference_kernel_wrapper(DecodingMeta *m,
                                         BatchConfig const *bc,
                                         GenericTensorAccessorR const &input,
                                         GenericTensorAccessorW const &softmax_output,
                                         GenericTensorAccessorW const &argmax_output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  if (bc->num_active_tokens() <= 0) {
    return;
  }
  
  int num_classes = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int vocab_offset = input.domain.lo()[0];  // Starting vocab index for this shard
  float loss = 0.0f;

  if (input.data_type == DT_HALF) {
    Decoding::inference_kernel<half>(m,
                                     bc,
                                     input.get_half_ptr(),
                                     softmax_output.get_half_ptr(),
                                     argmax_output.get_int32_ptr(),
                                     num_classes,
                                     vocab_offset,
                                     &loss,
                                     stream);
  } else if (input.data_type == DT_FLOAT) {
    Decoding::inference_kernel<float>(m,
                                      bc,
                                      input.get_float_ptr(),
                                      softmax_output.get_float_ptr(),
                                      argmax_output.get_int32_ptr(),
                                      num_classes,
                                      vocab_offset,
                                      &loss,
                                      stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (bc->num_finetuning_fwd_requests() > 0) {
    store_peft_token_ids(m, bc);
    // Store softmax activations for PEFT backward pass
    if (input.data_type == DT_HALF) {
      store_peft_activations(m, bc, num_classes, softmax_output.get_half_ptr(), stream);
    } else if (input.data_type == DT_FLOAT) {
      store_peft_activations(m, bc, num_classes, softmax_output.get_float_ptr(), stream);
    }
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[Decoding] forward time = %.2lfms\n", elapsed);
  }
}

template <typename DT>
__global__ void sparse_categorical_crossentropy_loss_peft_backward(
    DT *input_grad,
    DT const *output_grad,
    BatchConfig::TokenId const *token_ids,
    int num_tokens,
    int num_classes) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_classes) {
    int class_idx = i % num_classes;
    int token_idx = i / num_classes;
    input_grad[i] = output_grad[i];
    if (class_idx == token_ids[token_idx]) {
      input_grad[i] = input_grad[i] - (DT)1.0f;
    }
  }
}

template <typename DT>
void Decoding::peft_bwd_kernel(DecodingMeta const *m,
                               BatchConfig const *bc,
                               DT *input_grad_ptr,
                               int num_classes,
                               cudaStream_t stream) {
  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();

  int num_bwd_tokens = bc->requestsInfo[i].num_tokens_in_batch - 1;

  DT scale_factor = 1.0 / (bc->requestsInfo[i].num_tokens_in_batch);
  // ignore last token
  checkCUDA(cudaMemsetAsync(input_grad_ptr + num_bwd_tokens * num_classes,
                            0,
                            num_classes * sizeof(DT),
                            stream));
  checkCUDA(cudaMemcpyAsync(m->handle.workSpace,
                            m->peft_token_ids,
                            sizeof(BatchConfig::TokenId) * num_bwd_tokens,
                            cudaMemcpyHostToDevice,
                            stream));
  sparse_categorical_crossentropy_loss_peft_backward<<<
      GET_BLOCKS(num_bwd_tokens * num_classes),
      CUDA_NUM_THREADS,
      0,
      stream>>>(input_grad_ptr,
                static_cast<DT *>(m->output_grad_ptr),
                static_cast<BatchConfig::TokenId const *>(m->handle.workSpace),
                num_bwd_tokens,
                num_classes);
  // scale
  scale_kernel<<<GET_BLOCKS(num_bwd_tokens * num_classes),
                 CUDA_NUM_THREADS,
                 0,
                 stream>>>(
      input_grad_ptr, num_bwd_tokens * num_classes, DT(0.0), scale_factor);
}

/*static*/
void Decoding::peft_bwd_kernel_wrapper(DecodingMeta *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  int num_classes = input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  if (m->input_type[0] == DT_FLOAT) {
    Decoding::peft_bwd_kernel<float>(
        m, bc, input_grad.get_float_ptr(), num_classes, stream);
  } else if (m->input_type[0] == DT_HALF) {
    Decoding::peft_bwd_kernel<half>(
        m, bc, input_grad.get_half_ptr(), num_classes, stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[Decoding] peft_bwd time = %.2fms\n", elapsed);
  }
}

DecodingMeta::DecodingMeta(FFHandler handler,
                           Decoding const *decoding,
                           Legion::Domain const &input_domain,
                           MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, decoding) {
  beam_search = decoding->beam_search;
  
  if (peft_finetuning_enabled(peft_support_mode)) {
    allocated_peft_buffer_size =
        input_domain.get_volume() * data_type_size(decoding->data_type);
    gpu_mem_allocator.create_legion_instance(
        reserveInst, allocated_peft_buffer_size, "DecodingMeta");
    output_grad_ptr =
        gpu_mem_allocator.allocate_instance_untyped(allocated_peft_buffer_size);
  } else {
    allocated_peft_buffer_size = 0;
    output_grad_ptr = nullptr;
  }
  
  // Simple allocations for required buffers
  probs = nullptr; // Not needed for basic decoding
  d_loss = nullptr; // Not needed for basic decoding
  parent_output_buffer = nullptr; // Only needed for beam search if implemented
  
  // Create softmax context for sharded computation
  // Use a reasonable max sequence length based on input domain
  int max_seq_len = input_domain.get_dim() > 1 ? 
    (input_domain.hi()[input_domain.get_dim()-1] - input_domain.lo()[input_domain.get_dim()-1] + 1) : 1024;
  softmax_context = create_softmax_context(max_seq_len, handler.ncclComm);
  
  std::strcpy(op_name, decoding->name);
}

DecodingMeta::~DecodingMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
  // Destroy softmax context
  if (softmax_context != nullptr) {
    destroy_softmax_context(softmax_context);
  }
}

// Explicit template instantiations
template void softmax_argmax<half>(
  const half* input,
  half* output,
  int* argmax_indices,
  int vocab_size,
  int seq_len,
  cudaStream_t stream);

template void softmax_argmax<float>(
  const float* input,
  float* output,
  int* argmax_indices,
  int vocab_size,
  int seq_len,
  cudaStream_t stream);

// Explicit template instantiations for peft_bwd_kernel
template void Decoding::peft_bwd_kernel<half>(
    DecodingMeta const *m,
    BatchConfig const *bc,
    half *input_grad_ptr,
    int num_classes,
    cudaStream_t stream);

template void Decoding::peft_bwd_kernel<float>(
    DecodingMeta const *m,
    BatchConfig const *bc,
    float *input_grad_ptr,
    int num_classes,
    cudaStream_t stream);

} // namespace FlexFlow