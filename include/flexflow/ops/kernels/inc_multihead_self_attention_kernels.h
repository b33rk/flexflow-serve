#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H

#define QKV_WEIGHT_NUM 3
#define KV_WEIGHT_NUM 2

#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/inc_multihead_self_attention.h"

namespace FlexFlow {
namespace Kernels {
namespace IncMultiHeadAttention {

// flashinfer
// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t
    get_k_entry_offset_verify(int const token_idx,
                              int const page_idx,
                              int const num_heads,
                              int const head_dim) {
  size_t index = ((page_idx)*kPagesize * 2 + (token_idx % kPagesize)) *
                 head_dim * num_heads;
  return index;
}

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t
    get_v_entry_offset_verify(int const token_idx,
                              int const page_idx,
                              int const num_heads,
                              int const head_dim) {
  size_t index =
      ((page_idx)*kPagesize * 2 + kPagesize + (token_idx % kPagesize)) *
      head_dim * num_heads;
  return index;
}

// // kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_k_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const num_heads,
                                                     int const head_dim) {
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
          token_idx % kPagesize) * /* page slot index */
         num_heads *
         head_dim;
}

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_v_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const num_heads,
                                                     int const head_dim) {
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
          kPagesize + token_idx % kPagesize) * /* page slot index */
         num_heads *
         head_dim;
}
// [For the tokens in batch]
// Update the kv cache, and compact the q array.
// Source: qkv projeciton array of tokens in the batch.
// Destination: q&kv ptr took by the attention kernel.
// Note that the q&k here are the value after applying with position encoding.
void update_qkv_in_batch(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         ffStream_t stream);
template <typename DT>
void update_kv_cache_kernel_flashinfer(IncMultiHeadSelfAttentionMeta const *m,
                                       BatchConfig const *bc,
                                       ffStream_t stream);
template <typename DT>
void produce_output(IncMultiHeadSelfAttentionMeta const *m,
                    BatchConfig const *bc,
                    DT *output_ptr,
                    ffStream_t stream);

template <typename DT>
void compute_attention_kernel_prompt(IncMultiHeadSelfAttentionMeta *m,
                                     BatchConfig const *bc,
                                     int shard_id,
                                     ffStream_t stream);

template <typename DT>
void apply_scaling_and_rotary(IncMultiHeadSelfAttentionMeta const *m,
                              BatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              ffStream_t stream);

template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id);

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
template <typename DT>
void run_batched_matmul(IncMultiHeadSelfAttentionMeta const *meta,
                        cublasHandle_t handle,
                        cublasOperation_t transa,
                        cublasOperation_t transb,
                        int m,
                        int n,
                        int k,
                        void const *alpha,
                        const DT *A,
                        cudaDataType Atype,
                        int lda,
                        long long int strideA,
                        const DT *B,
                        cudaDataType Btype,
                        int ldb,
                        long long int strideB,
                        void const *beta,
                        DT *C,
                        cudaDataType Ctype,
                        int ldc,
                        long long int strideC,
                        int batchCount,
                        cudaDataType computeType,
                        cublasGemmAlgo_t algo,
                        ffStream_t stream,
                        int batch_ratio_a = 1,
                        int batch_ratio_b = 1,
                        int batch_ratio_c = 1,
                        bool bwd = false);
#else
template <typename DT>
void run_batched_matmul(IncMultiHeadSelfAttentionMeta const *meta,
                        hipblasHandle_t handle,
                        hipblasOperation_t transa,
                        hipblasOperation_t transb,
                        int m,
                        int n,
                        int k,
                        void const *alpha,
                        const DT *A,
                        hipblasDatatype_t Atype,
                        int lda,
                        long long int strideA,
                        const DT *B,
                        hipblasDatatype_t Btype,
                        int ldb,
                        long long int strideB,
                        void const *beta,
                        DT *C,
                        hipblasDatatype_t Ctype,
                        int ldc,
                        long long int strideC,
                        int batchCount,
                        hipblasDatatype_t computeType,
                        hipblasGemmAlgo_t algo,
                        hipStream_t stream,
                        int batch_ratio_a = 1,
                        int batch_ratio_b = 1,
                        int batch_ratio_c = 1,
                        bool bwd = false);
#endif

} // namespace IncMultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
