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

#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Domain;

SoftmaxMeta::SoftmaxMeta(FFHandler handler,
                         Softmax const *softmax,
                         Domain const &input_domain,
                         bool is_last_op,
                         MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, softmax) {
  dim = softmax->dim;
  profiling = softmax->profiling;
  inference_debugging = softmax->inference_debugging;
  if (peft_finetuning_enabled(peft_support_mode) && is_last_op) {
    allocated_peft_buffer_size =
        input_domain.get_volume() * data_type_size(softmax->data_type);
    gpu_mem_allocator.create_legion_instance(
        reserveInst, allocated_peft_buffer_size, "SoftmaxMeta");
    output_grad_ptr =
        gpu_mem_allocator.allocate_instance_untyped(allocated_peft_buffer_size);
  } else {
    allocated_peft_buffer_size = 0;
    output_grad_ptr = nullptr;
  }
  std::strcpy(op_name, softmax->name);
}

namespace Kernels {
namespace Softmax {

void forward_kernel_wrapper(SoftmaxMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  assert(false && "Not supported");
}

void backward_kernel_wrapper(SoftmaxMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  assert(false && "Not supported");
}

void inference_kernel_wrapper(SoftmaxMeta *m,
                              BatchConfig const *bc,
                              bool is_last_op,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
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
  int num_classes = output.domain.hi()[0] - output.domain.lo()[0] + 1;
  if (m->output_type[0] == DT_FLOAT) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_float_ptr(),
                               output.get_float_ptr(),
                               num_classes,
                               stream);
    if (is_last_op && bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations(
          m, bc, num_classes, output.get_float_ptr(), stream);
    }
  } else if (m->output_type[0] == DT_HALF) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_half_ptr(),
                               output.get_half_ptr(),
                               num_classes,
                               stream);
    if (is_last_op && bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations(
          m, bc, num_classes, output.get_half_ptr(), stream);
    }
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] inference time = %.2fms\n", m->op_name, elapsed);
  }
}

void peft_bwd_kernel_wrapper(SoftmaxMeta const *m,
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
  if (m->output_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel(
        m, bc, input_grad.get_float_ptr(), num_classes, stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel(
        m, bc, input_grad.get_half_ptr(), num_classes, stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] inference time = %.2fms\n", m->op_name, elapsed);
  }
}

namespace Internal {

// Optimized version using warp primitives
template<int BLOCK_SIZE>
__global__ void softmax_argmax_kernel(
  const half* __restrict__ input,
  half* __restrict__ output,
  // int* __restrict__ argmax_indices,
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
      float val = __half2float(input[seq_pos * vocab_size + vocab_idx]);
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
      float val = __half2float(input[seq_pos * vocab_size + vocab_idx]);
      float exp_val = expf(val - max_val);
      thread_sum += exp_val;
      output[seq_pos * vocab_size + vocab_idx] = __float2half(exp_val);
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
      float exp_val = __half2float(output[seq_pos * vocab_size + vocab_idx]);
      output[seq_pos * vocab_size + vocab_idx] = __float2half(exp_val * inv_sum);
  }
  
  // Store argmax
  // if (tid == 0) {
  //     argmax_indices[seq_pos] = max_idx;
  // }
}

// Wrapper function
void softmax_argmax(
  const half* input,
  half* output,
  // int* argmax_indices,
  int vocab_size,
  int seq_len,
  cudaStream_t stream) {

  const int block_size = 256;
  const int grid_size = seq_len;  // One block per sequence position
  
  
  // Use optimized kernel for larger vocabularies
  size_t shared_mem_size_opt = (block_size / 32) * sizeof(float) * 2 + (block_size / 32) * sizeof(int);
  softmax_argmax_kernel<block_size><<<grid_size, block_size, shared_mem_size_opt, stream>>>(
      input, output, /*argmax_indices,*/ vocab_size, seq_len
  );  
}

template <typename DT>
void inference_kernel_spatial_sharing(SoftmaxMeta const *m,
                                    BatchConfig const *bc,
                                    DT const *input_ptr,
                                    DT *output_ptr,
                                    int num_classes,
                                    cudaStream_t main_stream) {
  // launch finetuning fwd tokens kernel if there are any finetuning fwd tokens
  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaEventRecord(m->handle.peft_fwd_can_start, main_stream)); 
    checkCUDA(cudaStreamWaitEvent(m->handle.peft_fwd_stream, m->handle.peft_fwd_can_start, 0));

    softmax_argmax((const half*)input_ptr + num_classes * bc->num_inference_tokens(), 
                  (half*)output_ptr + num_classes * bc->num_inference_tokens(), 
                  num_classes, 
                  bc->num_finetuning_fwd_tokens(), 
                  m->handle.peft_fwd_stream);

    checkCUDA(cudaEventRecord(m->handle.peft_fwd_done, m->handle.peft_fwd_stream));
  }

  // launch inference kernel if there are inference tokens
  if (bc->num_inference_tokens() > 0) {

    softmax_argmax(
        (const half*)input_ptr, 
        (half*)output_ptr, 
        num_classes, 
        bc->num_inference_tokens(), 
        main_stream);
  }

  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaStreamWaitEvent(main_stream, m->handle.peft_fwd_done, 0));
  }
}

template <typename DT>
void inference_kernel(SoftmaxMeta const *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int num_classes,
                      cudaStream_t stream) {
  if (m->peft_support_mode == SPATIAL_SHARING || m->peft_support_mode == SPATIAL_SHARING_LIMITED) {
    inference_kernel_spatial_sharing(m, bc, input_ptr, output_ptr, num_classes, stream);
    return;
  }
  softmax_argmax((const half*)input_ptr, (half*)output_ptr, num_classes, bc->num_active_tokens(), stream);
}

template <typename DT>
void store_peft_activations(SoftmaxMeta *m,
                            BatchConfig const *bc,
                            int num_classes,
                            DT *output_ptr,
                            cudaStream_t stream) {
  assert(peft_finetuning_enabled(m->peft_support_mode));
  assert(m->output_grad_ptr != nullptr);

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

  size_t batch_offset = num_classes * tokens_previous_requests;
  size_t req_offset = num_classes * prev_steps_tokens;
  size_t data_size = num_classes * num_ft_tokens * sizeof(DT);
  assert(m->allocated_peft_buffer_size >= data_size);
  checkCUDA(cudaMemcpyAsync(static_cast<DT *>(m->output_grad_ptr) + req_offset,
                            output_ptr + batch_offset,
                            data_size,
                            cudaMemcpyDeviceToDevice,
                            stream));
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
void peft_bwd_kernel(SoftmaxMeta const *m,
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

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow
