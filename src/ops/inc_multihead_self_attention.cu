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
#include "cuComplex.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/utils/cuda_helper.h"
#include <math_constants.h>

// flashinfer & paged attention
#include "flashinfer/decode_attention_decl.cuh"
#include "flashinfer/prefill_attention_decl.cuh"
#include "flexflow/page_manager.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

// flashinfer & paged attention
using flashinfer::BatchDecodeHandler;
using flashinfer::BatchDecodeWithPagedKVCacheWrapperDispatched;
using flashinfer::BatchPrefillHandler;
using flashinfer::BatchPrefillWithPagedKVCacheWrapperDispatched;
using flashinfer::LogitsPostHook;
using flashinfer::MaskMode;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

std::string get_fwd_dbg_folder(IncMultiHeadSelfAttentionMeta const *m,
                               int shard_id) {
  std::string op_name_without_uid =
      IncMultiHeadSelfAttention::get_op_name_without_uid(m);
  fs::path dst_filepath = get_dst_folder("fwd", m->decoding_step, shard_id);
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

template <typename DT>
__global__ void store_kv_cache(DT const *devQKVProjArray,
                               DT *kCache_ptr,
                               DT *vCache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int num_tokens,
                               int max_seq_len,
                               int head_dim,
                               int num_q_heads,
                               int num_kv_heads) {
  CUDA_KERNEL_LOOP(i, num_tokens * head_dim * num_kv_heads) {
    // devQKVProjArray: [head_dim, tot_num_heads, num_tokens]
    // kCache_ptr: [head_dim, num_kv_heads, max_seq_len, 1]
    // vCache_ptr: [head_dim, num_kv_heads, max_seq_len, 1]

    // i is iterating over one set of key/val projections from the input
    int token_idx = i / (head_dim * num_kv_heads);
    int head_idx = (i / head_dim) % num_kv_heads;
    int offset = i % head_dim;

    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int key_src_idx = token_idx * head_dim * tot_num_heads +
                      head_dim * num_q_heads + head_dim * head_idx + offset;
    int val_src_idx = key_src_idx + head_dim * num_kv_heads;

    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    int dst_idx =
        tok_id * head_dim * num_kv_heads + head_idx * head_dim + offset;

    kCache_ptr[dst_idx] = devQKVProjArray[key_src_idx];
    vCache_ptr[dst_idx] = devQKVProjArray[val_src_idx];
  }
}

template <typename DT>
__global__ void store_query_cache(DT const *devQKVProjArray,
                                  DT *qCache_ptr,
                                  int num_tokens_in_batch,
                                  int first_token_offset_in_batch,
                                  int first_token_depth_in_request,
                                  int head_dim,
                                  int num_q_heads,
                                  int num_kv_heads) {
  CUDA_KERNEL_LOOP(i, num_tokens_in_batch * head_dim * num_q_heads) {
    int hidden_size = head_dim * num_q_heads;
    int tot_num_heads = num_q_heads + 2 * num_kv_heads;

    int token_idx = i / hidden_size;
    int offset = i % hidden_size;
    int src_idx =
        (first_token_offset_in_batch + token_idx) * (head_dim * tot_num_heads) +
        offset;

    qCache_ptr[first_token_depth_in_request * hidden_size + i] =
        devQKVProjArray[src_idx];
  }
}

template <typename DT>
__global__ void fill_entries_above_diagonal(DT *matrix,
                                            size_t num_rows,
                                            size_t num_cols,
                                            size_t num_q_heads,
                                            size_t entries_above_diagonal,
                                            DT value) {
  CUDA_KERNEL_LOOP(i, entries_above_diagonal * num_q_heads) {
    size_t head_idx = i / entries_above_diagonal;
    size_t entry_idx = i % entries_above_diagonal;
    size_t y = (-1 + sqrt(8 * (float)entry_idx + 1)) / 2;
    size_t x = entry_idx - y * (y + 1) / 2;
    y += (num_cols - num_rows) + 1;
    matrix[head_idx * num_rows * num_cols + num_cols * y + x] = value;
  }
}

bool is_finetuning_bwd_request(BatchConfig const *bc, int request_id) {
  return bc->requestsInfo[request_id].finetuning_request &&
         bc->requestsInfo[request_id].finetuning_backward_phase;
}

bool is_decoding_request(BatchConfig const *bc, int request_id) {
  return !bc->requestsInfo[request_id].finetuning_request &&
         !bc->requestsInfo[request_id].prompt_phase;
}

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
                        cudaStream_t stream,
                        int batch_ratio_a,
                        int batch_ratio_b,
                        int batch_ratio_c,
                        bool bwd) {
  if (batch_ratio_a == 1 && batch_ratio_b == 1 && batch_ratio_c == 1) {
    checkCUDA(cublasGemmStridedBatchedEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A,
                                         Atype,
                                         lda,
                                         strideA,
                                         B,
                                         Btype,
                                         ldb,
                                         strideB,
                                         beta,
                                         C,
                                         Ctype,
                                         ldc,
                                         strideC,
                                         batchCount,
                                         computeType,
                                         algo));
  } else {
    const DT **h_A_array = new const DT *[batchCount];
    const DT **h_B_array = new const DT *[batchCount];
    DT **h_C_array = new DT *[batchCount];
    for (int batch = 0; batch < batchCount; batch++) {
      h_A_array[batch] = A + (batch / batch_ratio_a) * strideA;
      h_B_array[batch] = B + (batch / batch_ratio_b) * strideB;
      h_C_array[batch] = C + (batch / batch_ratio_c) * strideC;
    }
    assert(sizeof(DT *) == sizeof(void *));
    if (!bwd) {
      // Copy pointer arrays to device
      checkCUDA(cudaMemcpyAsync(meta->d_A_array,
                                h_A_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_B_array,
                                h_B_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_C_array,
                                h_C_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));

      checkCUDA(cublasGemmBatchedEx(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    meta->d_A_array,
                                    Atype,
                                    lda,
                                    meta->d_B_array,
                                    Btype,
                                    ldb,
                                    beta,
                                    meta->d_C_array,
                                    Ctype,
                                    ldc,
                                    batchCount,
                                    computeType,
                                    algo));
    } else {
      checkCUDA(cudaMemcpyAsync(meta->d_A_array2,
                                h_A_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_B_array2,
                                h_B_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_C_array2,
                                h_C_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));

      checkCUDA(cublasGemmBatchedEx(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    meta->d_A_array2,
                                    Atype,
                                    lda,
                                    meta->d_B_array2,
                                    Btype,
                                    ldb,
                                    beta,
                                    meta->d_C_array2,
                                    Ctype,
                                    ldc,
                                    batchCount,
                                    computeType,
                                    algo));
    }
  }
}

template <typename DT>
__global__ void store_softmax_activation(DT const *qk_prods_softmax,
                                         DT *softmax_activation_buffer,
                                         int num_new_tokens,
                                         int total_tokens,
                                         int max_finetuning_seq_len,
                                         int num_q_heads) {
  CUDA_KERNEL_LOOP(i, num_new_tokens * total_tokens * num_q_heads) {
    // qk_prods_softmax: [num_new_tokens, total_tokens, num_q_heads]
    // softmax activation buffer: [MAX_FINETUNING_LENGTH(num_new_tokens),
    // MAX_FINETUNING_LENGTH(total_tokens), num_q_heads]
    int tokens_previous_steps = total_tokens - num_new_tokens;
    int new_tokens_idx = i % num_new_tokens;
    int total_tokens_idx = (i / num_new_tokens) % total_tokens;
    int head_idx = i / (num_new_tokens * total_tokens);
    int src_idx = head_idx * num_new_tokens * total_tokens +
                  total_tokens_idx * num_new_tokens + new_tokens_idx;
    int dst_idx = head_idx * max_finetuning_seq_len * max_finetuning_seq_len +
                  total_tokens_idx * max_finetuning_seq_len +
                  (tokens_previous_steps + new_tokens_idx);

    softmax_activation_buffer[dst_idx] = qk_prods_softmax[src_idx];
  }
}

template <typename DT>
void compute_attention_kernel_peft(IncMultiHeadSelfAttentionMeta *m,
                                   BatchConfig const *bc,
                                   DT *attn_heads,
                                   int shard_id,
                                   cudaStream_t peft_stream) {
  if (bc->num_finetuning_fwd_tokens() <= 0) {
    return;
  }

  checkCUDA(cublasSetStream(m->handle.peft_blas, peft_stream));
  checkCUDNN(cudnnSetStream(m->handle.peft_dnn, peft_stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  cudaDataType_t compute_type = cublas_data_type;

  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  // int tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;

  assert(bc->num_finetuning_fwd_tokens() > 0);
  int req_idx = bc->finetuning_request_index();
  assert(!bc->request_completed[req_idx]);
  assert(bc->requestsInfo[req_idx].finetuning_request &&
         !bc->requestsInfo[req_idx].finetuning_backward_phase);

  int num_new_tokens = bc->requestsInfo[req_idx].num_tokens_in_batch;
  int total_tokens = bc->requestsInfo[req_idx].first_token_depth_in_request +
                     bc->requestsInfo[req_idx].num_tokens_in_batch;
  assert(num_new_tokens > 0 && total_tokens > 0);

  // Copy query to m->query_activation_buffer for BWD
  // int max_peft_tokens = bc->requestsInfo[i].max_length;
  int max_peft_tokens = BatchConfig::max_sequence_length();
  size_t activation_size_needed =
      sizeof(DT) * max_peft_tokens * m->num_q_heads * m->qProjSize;
  if (activation_size_needed != m->allocated_peft_buffer_size1) {
    std::cout << "activation_size_needed: " << activation_size_needed
              << std::endl;
    std::cout << "m->allocated_peft_buffer_size1: "
              << m->allocated_peft_buffer_size1 << std::endl;
    std::cout << "max_peft_tokens: " << max_peft_tokens << std::endl;
    std::cout << "m->num_q_heads: " << m->num_q_heads << std::endl;
    std::cout << "m->qProjSize: " << m->qProjSize << std::endl;
    std::cout << "BatchConfig::max_sequence_length()"
              << BatchConfig::max_sequence_length() << std::endl;
    std::cout << "sizeof(DT)" << sizeof(DT) << std::endl;
  }
  assert(activation_size_needed == m->allocated_peft_buffer_size1);
  int parallelism = m->qProjSize * m->num_q_heads * num_new_tokens;
  int tokens_previous_steps = total_tokens - num_new_tokens;
  int tokens_previous_requests =
      bc->requestsInfo[req_idx].first_token_offset_in_batch;
  store_query_cache<<<GET_BLOCKS(parallelism),
                      min(CUDA_NUM_THREADS, parallelism),
                      0,
                      peft_stream>>>(
      static_cast<DT *>(m->devQKVProjArray),
      static_cast<DT *>(m->query_activation_buffer),
      num_new_tokens,
      tokens_previous_requests,
      tokens_previous_steps,
      m->qProjSize,
      m->num_q_heads,
      m->num_kv_heads);

  // Step 1: compute query-key product QK.T/sqrt(d_k)
  {
    DT alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      // Scale by sqrt(d_k) as per the original attention paper
      alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
    }
    // after transpositions
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    // before transpositions
    int lda = m->qProjSize * m->num_q_heads;
    int ldb = m->kProjSize * m->num_kv_heads;
    int ldc = num_new_tokens;
    // N.B. strides are applied before transpose operations
    int strideA = m->qProjSize;
    int strideB = m->kProjSize;
    int strideC = num_new_tokens * total_tokens;

    // matrix A: query_activation_buffer
    // matrix A's layout: [qProjSize, num_q_heads, tot_peft_tokens]
    // Skip over entries from previous PEFT fwd steps
    DT const *A = static_cast<DT *>(m->query_activation_buffer) +
                  tokens_previous_steps * m->qProjSize * m->num_q_heads;
    // matrix B: key cache (peft)
    // matrix B's layout: [kProjSize, num_kv_heads, total_tokens]
    // To get B, skip over K entries from previous requests (all heads +
    // padding)
    DT const *B = static_cast<DT *>(m->keyCachePeft);
    // matrix C: qk_prods (current req only)
    // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT *C = static_cast<DT *>(m->handle.workSpace);
    run_batched_matmul<DT>(m,
                           m->handle.peft_blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           m_,
                           n,
                           k,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           strideA,
                           B,
                           cublas_data_type,
                           ldb,
                           strideB,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           strideC,
                           m->num_q_heads,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                           peft_stream,
                           1,
                           m->num_q_heads / m->num_kv_heads,
                           1);
    if (m->inference_debugging) {
      std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".qk_prods";
      save_tensor(static_cast<DT const *>(m->handle.workSpace),
                  num_new_tokens * total_tokens * m->num_q_heads,
                  fpath.c_str());
    }
  }
  // Step 2: Add alibi position bias to qk production
  // matrix C: qk_prods
  // matrix C's layout: [num_new_tokens, total_tokens, num_heads]
  // To get C, skip over QK.T products from previous requests
  {
    // matrix C: qk_prods (current req only)
    // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT *C = static_cast<DT *>(m->handle.workSpace);
    if (*m->position_bias) {
      size_t parallelism = m->num_q_heads * total_tokens * num_new_tokens;
      apply_position_bias_qkprd<<<GET_BLOCKS(parallelism),
                                  min((size_t)CUDA_NUM_THREADS, parallelism),
                                  0,
                                  peft_stream>>>(C,
                                                 num_new_tokens,
                                                 total_tokens,
                                                 m->num_q_heads,
                                                 m->global_num_q_heads,
                                                 shard_id);
    }
  }

  // Step 3: Apply causal mask. Fill all elements above diagonal in qk prods
  // with -inf to force causal attention.
  {
    assert(num_new_tokens <= total_tokens);
    size_t entries_above_diagonal = num_new_tokens * (num_new_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      // matrix C: qk_prods (current req only)
      // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
      DT *C = static_cast<DT *>(m->handle.workSpace);
      size_t parallelism = m->num_q_heads * entries_above_diagonal;
      fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    peft_stream>>>(C,
                                                   num_new_tokens,
                                                   total_tokens,
                                                   m->num_q_heads,
                                                   entries_above_diagonal,
                                                   static_cast<DT>(-INFINITY));
    }
    if (m->inference_debugging) {
      std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".qk_prods.masked";
      save_tensor(static_cast<DT const *>(m->handle.workSpace),
                  num_new_tokens * total_tokens * m->num_q_heads,
                  fpath.c_str());
    }
  }

  // Step 4: Compute Softmax(QK.T/sqrt(d_k))
  {
    // Before modifying the parameters below, make sure to read the following
    // description of the CUDNN_TENSOR_NCHW tensor layout, from
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t:
    // This tensor format specifies that the data is laid out in the following
    // order: batch size, feature maps, rows, columns. The strides are
    // implicitly defined in such a way that the data are contiguous in memory
    // with no padding between images, feature maps, rows, and columns; the
    // columns are the inner dimension and the images are the outermost
    // dimension.
    int n_param = m->num_q_heads;
    int c_param = total_tokens;
    int h_param = 1;
    int w_param = num_new_tokens;
    checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                          CUDNN_TENSOR_NCHW,
                                          cudnn_data_type,
                                          n_param,
                                          c_param,
                                          h_param,
                                          w_param));
    float softmax_alpha = 1.0f, softmax_beta = 0.0f;
    // matrix C: qk_prods (current req only)
    // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT *C = static_cast<DT *>(m->handle.workSpace);
    // matrix C_softmax: qk_prods_softmax (current req only)
    // matrix C_softmax's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
    // The softmax operation below is executed according to the
    // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
    // softmax operation is computed per spatial location (H,W) per image (N)
    // across dimension C.
    checkCUDNN(cudnnSoftmaxForward(m->handle.peft_dnn,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &softmax_alpha,
                                   m->qk_tensor,
                                   C,
                                   &softmax_beta,
                                   m->qk_tensor,
                                   C_softmax));
    // Copy C_softmax to m->softmax_activation_buffer for PEFT backward
    int max_peft_tokens = BatchConfig::max_sequence_length();
    int max_dataset_entry_size = bc->requestsInfo[req_idx].max_length;
    size_t activation_size_needed =
        sizeof(DT) * max_peft_tokens * max_peft_tokens * m->num_q_heads;
    assert(activation_size_needed == m->allocated_peft_buffer_size2);
    int parallelism = m->num_q_heads * total_tokens * num_new_tokens;
    store_softmax_activation<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               peft_stream>>>(
        static_cast<DT *>(m->qk_prods_softmax),
        static_cast<DT *>(m->softmax_activation_buffer),
        num_new_tokens,
        total_tokens,
        max_dataset_entry_size,
        m->num_q_heads);

    if (m->inference_debugging) {
      std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".qk_prods_softmax";
      save_tensor(static_cast<DT const *>(m->qk_prods_softmax),
                  num_new_tokens * total_tokens * m->num_q_heads,
                  fpath.c_str());
    }
  }

  // Step 5: Matmul softmax(QK.T/sqrt(d_k)) by V. Implemented as V @
  // softmax(QK.T/sqrt(d_k)).T
  {
    DT alpha = 1.0f, beta = 0.0f;
    // after transpositions
    int m_ = m->vProjSize;
    int n = num_new_tokens;
    int k = total_tokens;
    // before transpositions
    int lda = m_ * m->num_kv_heads;
    int ldb = n;
    int ldc = m_ * m->num_q_heads;
    // N.B. strides are applied before transpose operations
    int strideA = m->vProjSize;
    int strideB = num_new_tokens * total_tokens;
    int strideC = m->vProjSize;
    // matrix A: value cache (peft)
    // matrix A's layout: [vProjSize, num_kv_heads, total_tokens]
    // To get A, skip over V.T entries from previous requests (all heads +
    // padding)
    DT *A = static_cast<DT *>(m->valueCachePeft);
    // matrix B: qk_prods_softmax (current req only)
    // matrix B's layout: [num_new_tokens, total_tokens, num_q_heads]
    // To get B, skip over softmax(QK.T/sqrt(d_k)) entries from previous
    // requests (all heads)
    DT *B = static_cast<DT *>(m->qk_prods_softmax);
    // matrix C: attn heads
    // matrix C's layout: [vProjSize, num_q_heads, num_new_tokens]
    // To get C, skip over softmax(QK.T/sqrt(d_k))V products from previous
    // requests
    // store the result attn heads, also skip the genration tokens
    DT *C = static_cast<DT *>(attn_heads) +
            (bc->requestsInfo[req_idx].first_token_offset_in_batch) *
                m->num_q_heads * m->vProjSize;
    run_batched_matmul<DT>(m,
                           m->handle.peft_blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           m_,
                           n,
                           k,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           strideA,
                           B,
                           cublas_data_type,
                           ldb,
                           strideB,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           strideC,
                           m->num_q_heads,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                           peft_stream,
                           m->num_q_heads / m->num_kv_heads,
                           1,
                           1);
    if (m->inference_debugging) {
      std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".qk_prods_softmax";
      save_tensor(static_cast<DT const *>(attn_heads),
                  num_new_tokens * m->num_q_heads * m->vProjSize,
                  fpath.c_str());
    }
  }
}

// only used by MPT model. https://arxiv.org/abs/2108.12409
template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_total_tokens * num_heads) {
    // get head_idx,
    int head_idx = i / (num_tokens * num_total_tokens) + (num_heads * shard_id);
    int position_idx = (i / num_tokens) % num_total_tokens;
    position_idx = position_idx + 1 - num_total_tokens;
    // 8 is alibi_bias_max in
    // https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json
    float base = (float)(head_idx + 1) * 8 / global_num_q_heads;
    float slopes = 1.0 / pow(2, base);
    // if(i == 0){
    //   printf("see position: %d, %f, %f, %f\n", position_idx, base, slopes,
    //   position_idx * slopes);
    // }
    input_ptr[i] += static_cast<DT>(position_idx * slopes);
  }
}

template <typename DT>
__global__ void scaling_query_kernel(DT *input_ptr,
                                     int qProjSize,
                                     int num_tokens,
                                     int num_q_heads,
                                     int num_kv_heads,
                                     float scaling_factor) {
  CUDA_KERNEL_LOOP(i, (qProjSize * num_q_heads) * num_tokens) {
    int token_idx = i / (qProjSize * num_q_heads);
    int offset = i % (qProjSize * num_q_heads);
    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int idx = token_idx * qProjSize * tot_num_heads + offset;
    input_ptr[idx] *= scaling_factor;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_fwd(DT *input_ptr,
                               cuFloatComplex *complex_input,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               float rope_theta,
                               bool llama3_rope,
                               float factor,
                               float low_freq_factor,
                               float high_freq_factor,
                               int original_max_position_embeddings,
                               int proj_size,
                               int num_tokens,
                               int num_q_heads,
                               int num_kv_heads) {
  int half_proj = proj_size / 2;
  int q_proj_work = num_tokens * num_q_heads * half_proj;
  int kv_proj_work = num_tokens * num_kv_heads * half_proj;
  int tot_num_heads = num_q_heads + 2 * num_kv_heads;
  CUDA_KERNEL_LOOP(i, q_proj_work + kv_proj_work) {
    bool q_tensor = i < q_proj_work;
    int num_heads = q_tensor ? num_q_heads : num_kv_heads;

    int real_i = q_tensor ? i : i - q_proj_work;
    int token_idx = real_i / (half_proj * num_heads);
    int pair_idx = real_i % half_proj;
    int head_idx = (real_i / half_proj) % num_heads;

    // input_ptr: [proj_size, tot_num_heads, num_tokens]
    int real_part_index = token_idx * proj_size * tot_num_heads +
                          (q_tensor ? 0 : proj_size * num_q_heads) +
                          head_idx * proj_size + pair_idx;
    int complex_part_index = real_part_index + half_proj;
    complex_input[i] = {(float)input_ptr[real_part_index],
                        (float)input_ptr[complex_part_index]};

    float inv_freq =
        1.0 / pow(rope_theta, (float)2.0 * pair_idx / proj_size); // θ_i

    if (llama3_rope) {
      float pi = CUDART_PI_F;
      float wavelen = 2 * pi / inv_freq;
      float low_freq_wavelen =
          original_max_position_embeddings / low_freq_factor;
      float high_freq_wavelen =
          original_max_position_embeddings / high_freq_factor;
      if (wavelen > low_freq_wavelen) {
        inv_freq = inv_freq / factor;
      }
      float smooth_factor =
          (original_max_position_embeddings / wavelen - low_freq_factor) /
          (high_freq_factor - low_freq_factor);
      if (!(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen)) {
        inv_freq = ((1 - smooth_factor) * inv_freq / factor +
                    smooth_factor * inv_freq);
      }
    }

    int pos = tokenInfos[token_idx].abs_depth_in_request;
    inv_freq = inv_freq * pos;

    cuFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = (DT)complex_input[i].x;
    input_ptr[complex_part_index] = (DT)complex_input[i].y;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_bwd(DT *input_ptr,
                               cuFloatComplex *complex_input,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               float rope_theta,
                               bool llama3_rope,
                               float factor,
                               float low_freq_factor,
                               float high_freq_factor,
                               int original_max_position_embeddings,
                               int proj_size,
                               int num_tokens,
                               int num_q_heads,
                               int num_kv_heads) {
  int half_proj = proj_size / 2;
  int q_proj_work = num_tokens * num_q_heads * half_proj;
  int kv_proj_work = num_tokens * num_kv_heads * half_proj;
  // int tot_num_heads = num_q_heads + 2 * num_kv_heads;
  CUDA_KERNEL_LOOP(i, q_proj_work + kv_proj_work) {
    // compute indexes to visit first half proj_size of each of q/k tensor.
    // devQKVProj has shape [num_tokens, proj_size, tot_num_heads] in peft_bwd
    bool q_tensor = i < q_proj_work;
    int num_heads = q_tensor ? num_q_heads : num_kv_heads;
    int real_i = q_tensor ? i : i - q_proj_work;

    int token_idx = real_i % num_tokens;
    int pair_idx = (real_i / num_tokens) % half_proj;
    int head_idx = real_i / (num_tokens * half_proj);
    assert(head_idx < num_heads);

    int complex_part_index =
        (q_tensor ? 0 : num_tokens * proj_size * num_q_heads) +
        head_idx * proj_size * num_tokens + pair_idx * num_tokens + token_idx;
    int real_part_index = complex_part_index + num_tokens * half_proj;

    complex_input[i] = {(float)input_ptr[real_part_index],
                        (float)input_ptr[complex_part_index]};

    float inv_freq =
        1.0 / pow(rope_theta, (float)2.0 * pair_idx / proj_size); // θ_i

    if (llama3_rope) {
      float pi = CUDART_PI_F;
      float wavelen = 2 * pi / inv_freq;
      float low_freq_wavelen =
          original_max_position_embeddings / low_freq_factor;
      float high_freq_wavelen =
          original_max_position_embeddings / high_freq_factor;
      if (wavelen > low_freq_wavelen) {
        inv_freq = inv_freq / factor;
      }
      float smooth_factor =
          (original_max_position_embeddings / wavelen - low_freq_factor) /
          (high_freq_factor - low_freq_factor);
      if (!(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen)) {
        inv_freq = ((1 - smooth_factor) * inv_freq / factor +
                    smooth_factor * inv_freq);
      }
    }

    int pos = tokenInfos[token_idx].abs_depth_in_request;
    inv_freq = inv_freq * pos;

    cuFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = (DT)complex_input[i].x;
    input_ptr[complex_part_index] = (DT)complex_input[i].y;
  }
}

template <typename DT>
void apply_scaling_and_rotary(IncMultiHeadSelfAttentionMeta const *m,
                              BatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              cudaStream_t inf_stream) {

  checkCUDA(cublasSetStream(m->handle.blas, inf_stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, inf_stream));
  assert(m->qProjSize == m->kProjSize && m->qProjSize == m->vProjSize);

  int num_tokens = bc->num_active_tokens();
  int parallelism = m->kProjSize * num_tokens * m->num_q_heads;

  if (m->scaling_query) {
    scaling_query_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           inf_stream>>>(output_ptr,
                                         m->qProjSize,
                                         num_tokens,
                                         m->num_q_heads,
                                         m->num_kv_heads,
                                         m->scaling_factor);
  }

  // Step 3: apply rotary embedding if needed
  if (m->rotary_embedding_meta->apply_rotary_embedding) {
    /*q&k*/
    int half_proj = m->qProjSize / 2;
    int q_proj_work = num_tokens * m->num_q_heads * half_proj;
    int kv_proj_work = num_tokens * m->num_kv_heads * half_proj;
    parallelism = q_proj_work + kv_proj_work;
    apply_rotary_embedding_fwd<<<GET_BLOCKS(parallelism),
                                 min(CUDA_NUM_THREADS, parallelism),
                                 0,
                                 inf_stream>>>(
        output_ptr,
        m->complex_input,
        m->token_infos,
        m->rotary_embedding_meta->rope_theta,
        (m->rotary_embedding_meta->rope_type == "llama3"),
        m->rotary_embedding_meta->factor,
        m->rotary_embedding_meta->low_freq_factor,
        m->rotary_embedding_meta->high_freq_factor,
        m->rotary_embedding_meta->original_max_position_embeddings,
        m->qProjSize,
        num_tokens,
        m->num_q_heads,
        m->num_kv_heads);
  }
}

template <typename DT>
__global__ void update_kv_cache_kernel_flashinfer_kernel(
    DT *qkv_proj_array,
    half *qTmp_ptr,
    half *kvCache_ptr,
    int32_t *kv_indptr,
    int32_t *kv_page_indices,
    bool const *request_completed,
    int peft_req_idx,
    BatchConfig::PerTokenInfo const *tokenInfos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int num_new_tokens) {
  int tot_num_heads = num_q_heads + 2 * num_kv_heads;
  // iterate over the whole qkv_proj_array
  CUDA_KERNEL_LOOP(i, num_new_tokens * (tot_num_heads * head_dim)) {
    // qkv_proj_array: [head_dim, tot_num_heads, num_new_tokens]
    // qTmp_ptr: [head_dim, num_q_heads, num_new_tokens]
    // kvCache_ptr: [head_dim, num_kv_heads, page_size, 2, max_num_pages]
    int proj_offset = i % head_dim;
    int head_idx = (i / head_dim) % tot_num_heads;
    int token_idx = i / (tot_num_heads * head_dim);
    assert(proj_offset < head_dim && "Invalid proj_offset");
    assert(head_idx < tot_num_heads && "Invalid head_idx");
    assert(token_idx < num_new_tokens && "Invalid token_idx");

    int token_abs_idx = tokenInfos[token_idx].abs_depth_in_request;
    int const req_idx = tokenInfos[token_idx].request_index;

    assert(req_idx != peft_req_idx &&
           "Attempting to use inference KV cache for PEFT tokens");

    int req_idx_compact = 0;
    for (int j = 0; j < req_idx; j++) {
      if (!request_completed[j]) {
        req_idx_compact++;
      }
    }
    assert(req_idx_compact >= 0 && req_idx_compact <= req_idx &&
           "Invalid request index");

    if (head_idx < num_q_heads) {
      // copy value into qTmp_ptr
      int offset = head_idx * head_dim + proj_offset;
      assert(offset >= 0 && offset < num_q_heads * head_dim &&
             "Q-tmp offset out of bounds");
      qTmp_ptr[token_idx * head_dim * num_q_heads + offset] = qkv_proj_array[i];
    } else {
      int logical_page_idx = token_abs_idx / kPagesize;
      int page_idx =
          kv_page_indices[kv_indptr[req_idx_compact] + logical_page_idx];
      int to_k_idx = get_k_entry_offset_verify(
          token_abs_idx, page_idx, num_kv_heads, head_dim);
      int to_v_idx = get_v_entry_offset_verify(
          token_abs_idx, page_idx, num_kv_heads, head_dim);
      if (head_idx - num_q_heads < num_kv_heads) {
        // key
        int offset = (head_idx - num_q_heads) * head_dim + proj_offset;
        assert(offset >= 0 && offset < num_kv_heads * head_dim &&
               "K-cache offset out of bounds");
        kvCache_ptr[to_k_idx + offset] = qkv_proj_array[i];
      } else {
        // value
        int offset =
            (head_idx - num_q_heads - num_kv_heads) * head_dim + proj_offset;
        assert(offset >= 0 && offset < num_kv_heads * head_dim &&
               "V-cache offset out of bounds");
        kvCache_ptr[to_v_idx + offset] = qkv_proj_array[i];
      }
    }
  }
}

// template <typename DT>
// __global__ void update_kv_cache_kernel_flashinfer_kernel(
//     DT *qkv_proj_array,
//     half *qTmp_ptr,
//     half *kvCache_ptr,
//     int32_t *kv_indptr,
//     int32_t *kv_page_indices,
//     bool const *request_completed,
//     int peft_req_idx,
//     BatchConfig::PerTokenInfo const *tokenInfos,
//     int num_q_heads,
//     int num_kv_heads,
//     int head_dim,
//     int num_new_tokens) {
//   int const q_hidden_size = num_q_heads * head_dim;
//   int const kv_hidden_size = num_kv_heads * head_dim;

//   int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int const token_idx = thread_idx / q_hidden_size;
//   int const offset = thread_idx % q_hidden_size;
//   if (token_idx >= num_new_tokens) {
//     return;
//   }
//   int const req_idx = tokenInfos[token_idx].request_index;
//   int token_abs_idx = tokenInfos[token_idx].abs_depth_in_request;
//   // calculate the compact request index in the easiest way
//   // TODO: recheck
//   int req_idx_compact = -1;
//   int cnt = 0;
//   while (cnt < req_idx + 1) {
//     if (!request_completed[cnt] && cnt != peft_req_idx) {
//       req_idx_compact++;
//     }
//     cnt++;
//   }
//   assert(req_idx_compact >= 0 && "Invalid request index");
//   size_t from_idx = token_idx * (q_hidden_size + temp_kv_hidden_size * 2);
//   qTmp_ptr[token_idx * q_hidden_size + offset] =
//       static_cast<half>(qkv_proj_array[from_idx + offset]);
//   if (offset < kv_hidden_size) {
//     int start = kv_indptr[req_idx_compact];
//     int end = kv_indptr[req_idx_compact + 1] - 1;
//     assert(start <= end && "Invalid kv_indptr");
//     assert(start + (token_abs_idx / kPagesize) <= end && "Invalid page
//     index"); int page_idx = kv_page_indices[start + (token_abs_idx /
//     kPagesize)]; size_t to_k_idx = get_k_entry_offset_verify(
//                token_abs_idx, page_idx, num_kv_heads, head_dim),
//            to_v_idx = get_v_entry_offset_verify(
//                token_abs_idx, page_idx, num_kv_heads, head_dim);
//     // key and value cache should be stored interleaved
//     int const stride = num_q_heads / num_kv_heads;
//     int const kv_offset =
//         offset / head_dim * stride * head_dim + offset % head_dim;
//     kvCache_ptr[to_k_idx + offset] =
//         static_cast<half>(qkv_proj_array[from_idx + q_hidden_size +
//         kv_offset]);
//     kvCache_ptr[to_v_idx + offset] =
//         static_cast<half>(qkv_proj_array[from_idx + q_hidden_size +
//                                          temp_kv_hidden_size + kv_offset]);
//   }
// }

template <typename DT>
void update_kv_cache_kernel_flashinfer(IncMultiHeadSelfAttentionMeta const *m,
                                       BatchConfig const *bc,
                                       cudaStream_t stream) {
  // printf("entered update_qkv_in_batch_verify\n");
  int num_new_tokens = bc->num_inference_tokens();
  if (num_new_tokens == 0) {
    return;
  }
  int tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  int parallelism = m->qProjSize * tot_num_heads * num_new_tokens;
  int peft_req_idx = (bc->num_finetuning_fwd_tokens() > 0)
                         ? bc->finetuning_request_index()
                         : -1;
  int32_t *kv_indptr = m->handle.incr_attention_metadata->kv_indptr;
  int32_t *kv_indices = m->handle.incr_attention_metadata->kv_indices;
  update_kv_cache_kernel_flashinfer_kernel<<<GET_BLOCKS(parallelism),
                                             min(CUDA_NUM_THREADS, parallelism),
                                             0,
                                             stream>>>(
      static_cast<DT *>(m->devQKVProjArray),
      static_cast<half *>(m->queryTmp),
      static_cast<half *>(m->kvCache),
      kv_indptr,
      kv_indices,
      m->request_completed,
      peft_req_idx,
      m->token_infos,
      m->num_q_heads,
      m->num_kv_heads,
      m->qProjSize,
      num_new_tokens);
}

template <typename DT>
void update_kv_cache_kernel_peft(IncMultiHeadSelfAttentionMeta const *m,
                                 BatchConfig const *bc,
                                 cudaStream_t stream) {
  int num_tokens = bc->num_finetuning_fwd_tokens();
  if (num_tokens <= 0) {
    return;
  }

  int tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  int head_dim = m->qProjSize;
  int i = bc->finetuning_request_index();
  int tokens_previous_requests =
      bc->requestsInfo[i].first_token_offset_in_batch;
  DT *qkv_ptr = static_cast<DT *>(m->devQKVProjArray) +
                m->qProjSize * tot_num_heads * tokens_previous_requests;

  int parallelism = head_dim * tot_num_heads * num_tokens;
  // devQKVProj has shape [qProjSize, tot_num_heads, num_new_tokens]
  store_kv_cache<<<GET_BLOCKS(parallelism),
                   min(CUDA_NUM_THREADS, parallelism),
                   0,
                   stream>>>(qkv_ptr,
                             static_cast<DT *>(m->keyCachePeft),
                             static_cast<DT *>(m->valueCachePeft),
                             m->token_infos,
                             num_tokens,
                             BatchConfig::max_sequence_length(),
                             head_dim,
                             m->num_q_heads,
                             m->num_kv_heads);
}

template <typename DT>
void flashinfer_incr_attention(IncMultiHeadSelfAttentionMeta *m,
                               BatchConfig const *bc,
                               int shard_id,
                               DT *output_ptr,
                               cudaStream_t stream) {

  // global constant parameters
  uint32_t const num_q_heads = m->num_q_heads;
  uint32_t const num_kv_heads = m->num_kv_heads;
  uint32_t const head_dim = m->qProjSize;
  uint32_t const batch_size = bc->num_inference_requests();
  float const sm_scale =
      (*m->qk_prod_scaling) ? 1.0f / sqrt(m->qProjSize) : 1.0f;
  assert(batch_size > 0);
  assert(num_q_heads > 0);
  assert(num_kv_heads > 0);
  assert(head_dim > 0);
  assert(bc->num_inference_tokens() > 0);

  half *q = static_cast<half *>(m->queryTmp),
       *kv = static_cast<half *>(m->kvCache), *o = (half *)output_ptr;
  assert(q != nullptr && "q is null!");
  assert(kv != nullptr && "kv is null!");
  assert(o != nullptr && "o is null!");
  assert(m->handle.incr_attention_metadata->q_indptr != nullptr &&
         "q_indptr is null!");
  assert(m->handle.incr_attention_metadata->kv_indices != nullptr &&
         "kv_indices is null!");
  assert(m->handle.incr_attention_metadata->kv_indptr != nullptr &&
         "kv_indptr is null!");
  assert(m->handle.incr_attention_metadata->kv_last_page_len != nullptr &&
         "kv_last_page_len is null!");

  paged_kv_t<PageStorage::kIndices, half, int32_t> paged_kv(
      num_kv_heads,
      kPagesize,
      head_dim,
      batch_size,
      QKVLayout::kNHD,
      kv,
      m->handle.incr_attention_metadata->kv_indices,
      m->handle.incr_attention_metadata->kv_indptr,
      m->handle.incr_attention_metadata->kv_last_page_len);

  if (m->inference_debugging) {
    bc->save_to_file(get_fwd_dbg_folder(m, shard_id) + ".batch_config");
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".q_indptr";
    save_tensor(
        static_cast<int32_t *>(m->handle.incr_attention_metadata->q_indptr),
        batch_size + 1,
        fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".kv_indptr";
    save_tensor(
        static_cast<int32_t *>(m->handle.incr_attention_metadata->kv_indptr),
        batch_size + 1,
        fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".kv_indices";

    int num_pages;
    checkCUDA(
        cudaMemcpy(&num_pages,
                   m->handle.incr_attention_metadata->kv_indptr + batch_size,
                   sizeof(int),
                   cudaMemcpyDeviceToHost));
    save_tensor(
        static_cast<int32_t *>(m->handle.incr_attention_metadata->kv_indices),
        num_pages,
        fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".kv_last_page_len";
    save_tensor(static_cast<int32_t *>(
                    m->handle.incr_attention_metadata->kv_last_page_len),
                batch_size,
                fpath.c_str());
  }

  assert(m->handle.incr_attention_metadata->prompt_handler_collections.count(
             batch_size) != 0 &&
         "Handler is not initialized");
  void *handler =
      m->handle.incr_attention_metadata->prompt_handler_collections[batch_size];
  // printf("obtained handler\n");
  assert(sizeof(DT) == 2 && "FlashInfer only supports half precision");
  DISPATCH_HEADDIM(head_dim, HEAD_DIM, {
     // printf("Launching BatchPrefillWithPagedKVCacheWrapperDispatched\n");
     cudaError_t result =
         BatchPrefillWithPagedKVCacheWrapperDispatched<PageStorage::kIndices,
                                                       HEAD_DIM,
                                                       LogitsPostHook::kNone,
                                                       PosEncodingMode::kNone,
                                                       false,
                                                       MaskMode::kCausal,
                                                       half,
                                                       half,
                                                       half,
                                                       int32_t>(
             static_cast<BatchPrefillHandler *>(handler),
             q,
             m->handle.incr_attention_metadata->q_indptr,
             /*q_offset=*/nullptr,
             paged_kv,
             /*custom_mask=*/nullptr,
             /*qk_indptr=*/nullptr,
             o,
             /*lse=*/nullptr,
             num_q_heads,
             /*window_left=*/-1,
             /*logits_soft_cap=*/0.f,
             sm_scale,
             /*rope_scale=*/1.f,
             /*rope_theta=*/static_cast<float>(1e4),
             stream);
     if (result != cudaSuccess) {
       throw std::runtime_error("Failed to run "
                                "IncrementalDecodingAttentionForwardKernel: " +
                                std::string(cudaGetErrorString(result)));
     }
   });
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *qkv_ptr,
                      DT *output_ptr,
                      cudaStream_t inf_stream,
                      cudaStream_t peft_stream) {

  // phase 0: copy calculated qkv into devQKVProjArray
  // [qProjSize, tot_num_heads, num_new_tokens]
  assert(m->qProjSize == m->kProjSize && m->qProjSize == m->vProjSize);
  size_t tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  size_t qkv_proj_size = m->qProjSize * tot_num_heads * bc->num_active_tokens();

  cudaMemcpyAsync(m->devQKVProjArray,
                  qkv_ptr,
                  qkv_proj_size * sizeof(DT),
                  cudaMemcpyDeviceToDevice,
                  inf_stream);

  if (m->inference_debugging) {
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".devQKVProjArray";
    save_tensor(static_cast<DT const *>(m->devQKVProjArray),
                qkv_proj_size,
                fpath.c_str());
  }

  // phase 1: Implement kernel to apply rotary embedding and scaling
  apply_scaling_and_rotary(
      m, bc, shard_id, static_cast<DT *>(m->devQKVProjArray), inf_stream);

  if (m->inference_debugging) {
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".post_rope";
    save_tensor(static_cast<DT const *>(m->devQKVProjArray),
                qkv_proj_size,
                fpath.c_str());
  }

  // peft stream can only start after
  if (bc->num_finetuning_fwd_tokens() > 0) {
    // wait until copy to devQKVProjArray and application of scaling & rotary
    // have finished
    cudaEvent_t prep_done;
    cudaEventCreate(&prep_done);
    cudaEventRecord(prep_done, inf_stream);
    cudaStreamWaitEvent(peft_stream, prep_done, 0);

    update_kv_cache_kernel_peft<DT>(m, bc, peft_stream);
    compute_attention_kernel_peft<DT>(m, bc, output_ptr, shard_id, peft_stream);

    assert(m->peft_token_infos != nullptr);
    assert(m->peft_token_infos_size == sizeof(BatchConfig::PerTokenInfo) *
                                           BatchConfig::max_sequence_length());
    int num_ft_tokens = bc->num_finetuning_fwd_tokens();
    int i = bc->finetuning_request_index();
    int tokens_previous_requests =
        bc->requestsInfo[i].first_token_offset_in_batch;
    int prev_steps_tokens = bc->requestsInfo[i].first_token_depth_in_request;
    for (int j = 0; j < num_ft_tokens; j++) {
      m->peft_token_infos[prev_steps_tokens + j] =
          bc->tokensInfo[tokens_previous_requests + j];
    }
  }

  // flashinfer sdpa
  assert(bc->num_finetuning_fwd_tokens() >= 0 &&
         bc->num_finetuning_bwd_tokens() >= 0);
  if (bc->num_inference_tokens() > 0) {
    update_kv_cache_kernel_flashinfer<DT>(m, bc, inf_stream);
    flashinfer_incr_attention<DT>(m, bc, shard_id, output_ptr, inf_stream);
  }

  // if (m->inference_debugging) {
  //   size_t key_cache_size = m->kProjSize * m->num_kv_heads *
  //                           BatchConfig::max_sequence_length() *
  //                           BatchConfig::max_requests_per_batch();
  //   std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".key_cache";
  //   save_tensor(
  //       static_cast<DT const *>(m->keyCache), key_cache_size, fpath.c_str());
  //   fpath = get_fwd_dbg_folder(m, shard_id) + ".value_cache";
  //   save_tensor(
  //       static_cast<DT const *>(m->valueCache), key_cache_size,
  //       fpath.c_str());
  // }
}

std::string get_peft_dbg_folder(IncMultiHeadSelfAttentionMeta const *m,
                                int shard_id) {
  std::string op_name_without_uid =
      IncMultiHeadSelfAttention::get_op_name_without_uid(m);
  fs::path dst_filepath = get_dst_folder("bwd", m->bwd_step, shard_id);
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

__global__ void transposeAdd_half_kernel(
    half *out, half const *in, int width, int height, half alpha, half beta) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = t_id; i < width * height; i += num_threads) {
    int row = i / width;
    int col = i % width;
    out[col * height + row] =
        alpha * in[row * width + col] + beta * out[col * height + row];
  }
}

__global__ void transposeAdd_float_kernel(float *out,
                                          float const *in,
                                          int width,
                                          int height,
                                          float alpha,
                                          float beta) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = t_id; i < width * height; i += num_threads) {
    int row = i / width;
    int col = i % width;
    out[col * height + row] =
        alpha * in[row * width + col] + beta * out[col * height + row];
  }
}

template <typename DT>
void transposeAdd(DT *out,
                  const DT *in,
                  int width,
                  int height,
                  float alpha,
                  float beta,
                  cudaStream_t stream) {
  assert(false && "Unsupported data type");
}

template <>
void transposeAdd<float>(float *out,
                         float const *in,
                         int width,
                         int height,
                         float alpha,
                         float beta,
                         cudaStream_t stream) {
  transposeAdd_float_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, alpha, beta);
}

template <>
void transposeAdd<half>(half *out,
                        half const *in,
                        int width,
                        int height,
                        float alpha,
                        float beta,
                        cudaStream_t stream) {
  transposeAdd_half_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, __float2half(alpha), __float2half(beta));
}

template <typename DT>
void peft_bwd_kernel(IncMultiHeadSelfAttentionMeta const *m,
                     BatchConfig const *bc,
                     int shard_id,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     cudaStream_t peft_stream) {
  assert(!m->offload);
  checkCUDA(cublasSetStream(m->handle.peft_blas, peft_stream));
  checkCUDNN(cudnnSetStream(m->handle.peft_dnn, peft_stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  cudaDataType_t compute_type = cublas_data_type;

  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();
  int num_tokens = bc->requestsInfo[i].num_tokens_in_batch;
  int num_total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                         bc->requestsInfo[i].num_tokens_in_batch;
  // Currently assume we are calculating gradients for all tokens
  // of a request
  assert(num_tokens == num_total_tokens);
  assert(num_total_tokens == bc->requestsInfo[i].max_length);
  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  // assert(bc->requestsInfo[i].first_token_offset_in_batch == 0);

  if (m->inference_debugging) {
    // save result to file for checking
    std::string filename =
        get_peft_dbg_folder(m, shard_id) + ".o_proj.input_gradient_0";
    save_tensor(output_grad_ptr,
                m->vProjSize * m->num_q_heads * num_tokens,
                filename.c_str());
  }

  // Step 1: compute gradients w.r.t. value
  {
    float alpha = 1.0f, beta = 0.0f;
    // matrix A: qk_prods_softmax
    // matrix A's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT const *A = static_cast<DT *>(m->qk_prods_softmax);
    // matrix B: attn_heads gradients
    // matrix B's layout: [vProjSize * num_q_heads, num_new_tokens]
    DT const *B = output_grad_ptr;
    // matrix C: gradients for value (saved as part of m->devQKVProjArray)
    // matrix C's layout: [num_tokens, qProjsize * num_q_heads, 3]
    // note that we first need to compute the gradients wrt each q_heads, then
    // we can sum the gradients corresponding to each group of q_heads to obtain
    // the gradients wrt each value head
    DT *C = static_cast<DT *>(m->devQKVProjArrayBWD) +
            2 * num_tokens *
                (m->qProjSize * m->num_q_heads); // skip over regions reserved
                                                 // for Q and K gradients
    // after transpositions
    int m_ = num_tokens; // total_tokens
    int n_ = m->vProjSize;
    int k_ = num_tokens; // num_new_tokens
    // before transpositions
    int lda = num_tokens; // num_new_tokens
    int ldb = m->vProjSize * m->num_q_heads;
    int ldc = num_tokens; // total_tokens
    // N.B. strides are applied before transpose operations
    int strideA = num_tokens * num_tokens; // num_new_tokens * total_tokens
    int strideB = m->vProjSize;
    int strideC = num_tokens * m->vProjSize;
    checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_T,
                                         m_,
                                         n_,
                                         k_,
                                         &alpha,
                                         A,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         m->num_q_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // save result to file for checking
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".v_proj.input_gradient_0";
      save_tensor(C, m_ * n_ * m->num_q_heads, filename.c_str());
      std::string filename2 =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax";
      save_tensor(A, m_ * k_ * m->num_q_heads, filename2.c_str());
    }
  }
  // Step 2: compute gradients w.r.t. the qk_prods_softmax tensor
  {
    float alpha = 1.0f, beta = 0.0f;
    // matrix A: attn_heads gradients
    // matrix A's layout: [vProjSize * num_q_heads, num_new_tokens]
    DT const *A = output_grad_ptr;
    // matrix B: value cache
    // matrix B's layout: [vProjSize * num_kv_heads, max_num_tokens, 1]
    DT const *B = static_cast<DT *>(m->valueCachePeft);

    // matrix C: qk_prods_softmax gradients
    // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT *C = static_cast<DT *>(m->qk_prods_softmax);
    // after transposition & striding
    int m_ = num_tokens; // num_new_tokens
    int n_ = num_tokens;
    int k_ = m->vProjSize;
    // before transposition and striding
    int lda = m->vProjSize * m->num_q_heads;
    int ldb = m->vProjSize * m->num_kv_heads;
    int ldc = num_tokens; // num_new_tokens
    int strideA = m->vProjSize;
    int strideB = m->vProjSize;
    int strideC = num_tokens * num_tokens; // num_new_tokens * total_tokens

    run_batched_matmul<DT>(m,
                           m->handle.peft_blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           m_,
                           n_,
                           k_,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           strideA,
                           B,
                           cublas_data_type,
                           ldb,
                           strideB,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           strideC,
                           m->num_q_heads,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                           peft_stream,
                           1,
                           m->num_q_heads / m->num_kv_heads,
                           1,
                           true);
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad";
      save_tensor(
          C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
      std::string filename2 = get_peft_dbg_folder(m, shard_id) + ".vcache";
      save_tensor(B,
                  m->vProjSize * m->num_kv_heads *
                      BatchConfig::max_sequence_length(),
                  filename2.c_str());
    }
  }
  // Step 3: softmax backpropagation
  {
    float alpha = 1.0f, beta = 0.0f;
    int n_param = m->num_q_heads;
    int c_param = num_tokens;
    int h_param = 1;
    int w_param = num_tokens;
    checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                          CUDNN_TENSOR_NCHW,
                                          cudnn_data_type,
                                          n_param,
                                          c_param,
                                          h_param,
                                          w_param));
    checkCUDNN(cudnnSoftmaxBackward(m->handle.peft_dnn,
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    m->qk_tensor,
                                    m->softmax_activation_buffer,
                                    m->qk_tensor,
                                    m->qk_prods_softmax,
                                    &beta,
                                    m->qk_tensor,
                                    m->handle.workSpace));

    if (m->inference_debugging) {
      DT *C = static_cast<DT *>(m->handle.workSpace);
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad_in";
      save_tensor(
          C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
      filename =
          get_peft_dbg_folder(m, shard_id) + ".softmax_activation_buffer";
      save_tensor(static_cast<DT *>(m->softmax_activation_buffer),
                  num_tokens * num_tokens * m->num_q_heads,
                  filename.c_str());
    }

    //  TODO: fill all elements above diagonal to force causal attention
    size_t entries_above_diagonal = num_tokens * (num_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      size_t parallelism = m->num_q_heads * entries_above_diagonal;
      fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    peft_stream>>>(
          static_cast<DT *>(m->handle.workSpace),
          num_tokens,
          num_tokens,
          m->num_q_heads,
          entries_above_diagonal,
          DT(0.0f));
    }
    if (m->inference_debugging) {
      DT *C = static_cast<DT *>(m->handle.workSpace);
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad_in.masked";
      save_tensor(
          C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
    }
  }
  // Step 4: compute gradients w.r.t. key
  {
    float alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = 1.0f / sqrt(m->kProjSize);
    }
    // matrix A: gradients w.r.t. qk_prods
    // matrix A's layout: [num_new_tokens, num_tokens, num_q_heads]
    DT const *A = static_cast<DT *>(m->handle.workSpace);
    // matrix B: query activation (in query_activation_buffer)
    // matrix B's layout: [m->qProjSize * num_q_heads, num_new_tokens]
    DT const *B = static_cast<DT *>(m->query_activation_buffer);
    // matrix C: gradients for key (saved as part of m->devQKVProjArrayBWD)
    // matrix C's layout: [num_tokens, qProjsize * num_q_heads, 3]
    // note that we first need to compute the gradients wrt each q_heads, then
    // we can sum the gradients corresponding to each group of q_heads to obtain
    // the gradients wrt each key head
    DT *C = static_cast<DT *>(m->devQKVProjArrayBWD) +
            num_tokens *
                (m->qProjSize *
                 m->num_q_heads); // skip over regions reserved for Q gradients
    // after transposition & striding
    int m_ = num_tokens;
    int n_ = m->kProjSize;
    int k_ = num_tokens; // num_new_tokens
    // before transposition and striding
    int lda = num_tokens; // num_new_tokens
    int ldb = m->kProjSize * m->num_q_heads;
    int ldc = num_tokens;
    int strideA = num_tokens * num_tokens;
    int strideB = m->kProjSize;
    int strideC = num_tokens * m->kProjSize;
    checkCUDA(cublasGemmStridedBatchedEx(m->handle.peft_blas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_T,
                                         m_,
                                         n_,
                                         k_,
                                         &alpha,
                                         A,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         m->num_q_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".query_activation";
      save_tensor(
          B, m->qProjSize * m->num_q_heads * num_tokens, filename.c_str());
      std::string filename2 =
          get_peft_dbg_folder(m, shard_id) + ".devkproj_pre";
      save_tensor(
          C, num_tokens * (m->qProjSize * m->num_q_heads), filename2.c_str());
    }
  }
  // Step 5: compute gradients w.r.t query
  {
    float alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = 1.0f / sqrt(m->kProjSize);
    }
    // matrix A: gradients w.r.t. qk_prods
    // matrix A's layout: [num_new_tokens, num_tokens, num_q_heads]
    DT const *A = static_cast<DT *>(m->handle.workSpace);
    // matrix B: key cache
    // matrix B's layout: [vProjSize * num_kv_heads, max_num_tokens, num_req]
    DT const *B = static_cast<DT *>(m->keyCachePeft);
    // matrix C: gradients for query (saved as part of m->devQKVProjArrayBWD)
    // matrix C's layout: [num_tokens, qProjsize * num_q_heads, 3]
    DT *C = static_cast<DT *>(m->devQKVProjArrayBWD);
    // after transposition & striding
    int m_ = num_tokens; // num_new_tokens
    int n_ = m->qProjSize;
    int k_ = num_tokens;
    // before transposition and striding
    int lda = num_tokens; // num_new_tokens
    int ldb = m->qProjSize * m->num_kv_heads;
    int ldc = num_tokens;
    int strideA = num_tokens * num_tokens;
    int strideB = m->qProjSize;
    int strideC = num_tokens * m->qProjSize;
    run_batched_matmul<DT>(m,
                           m->handle.peft_blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           m_,
                           n_,
                           k_,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           strideA,
                           B,
                           cublas_data_type,
                           ldb,
                           strideB,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           strideC,
                           m->num_q_heads,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                           peft_stream,
                           1,
                           m->num_q_heads / m->num_kv_heads,
                           1,
                           true);
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray_pre";
      save_tensor(
          C, num_tokens * m->qProjSize * m->num_q_heads * 3, filename.c_str());
    }
  }

  // Step 6: perform rotary position embeddings (RoPE) bwd
  {
    if (m->rotary_embedding_meta->apply_rotary_embedding) {
      checkCUDA(cudaMemcpyAsync(m->peft_token_infos_device,
                                m->peft_token_infos,
                                m->peft_token_infos_size,
                                cudaMemcpyHostToDevice,
                                peft_stream));
      assert(m->qProjSize == m->kProjSize);
      /*q&k*/
      int half_proj = m->qProjSize / 2;
      int q_proj_work = num_tokens * m->num_q_heads * half_proj;
      int kv_proj_work = num_tokens * m->num_kv_heads * half_proj;
      int parallelism = q_proj_work + kv_proj_work;
      apply_rotary_embedding_bwd<<<GET_BLOCKS(parallelism),
                                   min(CUDA_NUM_THREADS, parallelism),
                                   0,
                                   peft_stream>>>(
          static_cast<DT *>(m->devQKVProjArrayBWD),
          m->complex_input,
          m->peft_token_infos_device,
          m->rotary_embedding_meta->rope_theta,
          (m->rotary_embedding_meta->rope_type == "llama3"),
          m->rotary_embedding_meta->factor,
          m->rotary_embedding_meta->low_freq_factor,
          m->rotary_embedding_meta->high_freq_factor,
          m->rotary_embedding_meta->original_max_position_embeddings,
          m->qProjSize,
          num_tokens,
          m->num_q_heads,
          m->num_kv_heads);
      DT *C = static_cast<DT *>(m->devQKVProjArrayBWD);
      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray";
        save_tensor(C,
                    num_tokens * m->qProjSize * m->num_q_heads * 3,
                    filename.c_str());
      }
    }

    // matrix C: gradients for key (saved as part of m->devQKVProjArrayBWD)
    // matrix C's layout: [num_tokens, qProjsize * num_heads, 3]
    DT *C = static_cast<DT *>(m->devQKVProjArrayBWD) +
            num_tokens *
                (m->qProjSize *
                 m->num_q_heads); // skip over regions reserved for Q gradients
    if (m->inference_debugging) {
      std::string filename = get_peft_dbg_folder(m, shard_id) + ".devkproj";
      save_tensor(
          C, num_tokens * (m->qProjSize * m->num_q_heads), filename.c_str());
    }
  }

  // Step 7: compute gradients w.r.t. input
  {
    float alpha = 1.0f, beta = 0.0f;
    if (!m->reset_input_grads[0]) {
      beta = 1.0f;
    }
    // matrix B: gradients w.r.t. QKV (concatenated in devQKVArray)
    // matrix B's layout: [num_tokens, qProjsize * tot_num_heads]
    DT const *B = static_cast<DT *>(m->devQKVProjArrayBWD);
    // matrix C: gradients w.r.t. input
    // matrix C's layout: [qProjsize * tot_num_heads, num_tokens]
    DT *C = input_grad_ptr;
    int n_ = num_tokens;
    int k_ = m->qProjSize * (m->num_q_heads + 2 * m->num_kv_heads);

    // The original version uses existing result and attention's projection to
    // do further calculation in a way different than the usual dense layer,
    // they are off by a transpose. So an explicit transpose is needed here.
    // The add here is just for gradient accumulation.
    transposeAdd(C, B, n_, k_, alpha, beta, peft_stream);

    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".self_attn.input_gradient_0";
      save_tensor(
          C, num_tokens * m->qProjSize * m->num_q_heads, filename.c_str());
    }
  }
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  cudaStream_t inf_stream;
  checkCUDA(get_legion_stream(&inf_stream));
  cudaStream_t peft_stream;
  checkCUDA(get_legion_stream(&peft_stream));

  // cudaEvent_t t_start, t_end;
  // if (m->profiling) {
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);
  // }

  assert(input.data_type == output.data_type);

  if (input.data_type == DT_HALF) {
    Kernels::IncMultiHeadAttention::inference_kernel(m,
                                                     bc,
                                                     shard_id,
                                                     input.get_half_ptr(),
                                                     output.get_half_ptr(),
                                                     inf_stream,
                                                     peft_stream);
  } else if (input.data_type == DT_FLOAT) {
    Kernels::IncMultiHeadAttention::inference_kernel(m,
                                                     bc,
                                                     shard_id,
                                                     input.get_float_ptr(),
                                                     output.get_float_ptr(),
                                                     inf_stream,
                                                     peft_stream);
  } else {
    assert(false && "Unspported data type");
  }

  // if (m->profiling) {
  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   printf("IncMultiHeadSelfAttention forward time = %.9fms\n", elapsed);
  // }
}

/*static*/
void IncMultiHeadSelfAttention::peft_bwd_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(input_grad.data_type == output_grad.data_type);

  if (input_grad.data_type == DT_HALF) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_half_ptr(),
                                                    output_grad.get_half_ptr(),
                                                    stream);
  } else if (input_grad.data_type == DT_FLOAT) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_float_ptr(),
                                                    output_grad.get_float_ptr(),
                                                    stream);
  } else {
    assert(false && "Unspported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("IncMultiHeadSelfAttention PEFT backward time = %.9fms\n", elapsed);
  }
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
    MemoryAllocator &inf_mem_allocator,
    MemoryAllocator &kv_cache_mem_allocator,
    MemoryAllocator &peft_mem_allocator,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    INC_DECODING_MODE,
                                    attn,
                                    attn->qProjSize,
                                    attn->kProjSize,
                                    attn->vProjSize,
                                    attn->oProjSize,
                                    attn->rotary_embedding_meta,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->scaling_factor,
                                    inf_mem_allocator,
                                    kv_cache_mem_allocator,
                                    peft_mem_allocator,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
                                    attn->num_kv_cache_pages,
                                    attn->quantization_type,
                                    attn->offload) {}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    InferenceMode infer_mode,
    Op const *attn,
    int _qProjSize,
    int _kProjSize,
    int _vProjSize,
    int _oProjSize,
    RotaryEmbeddingMeta _rotary_embedding_meta,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _position_bias,
    float _scaling_factor,
    MemoryAllocator &inf_mem_allocator,
    MemoryAllocator &kv_cache_mem_allocator,
    MemoryAllocator &peft_mem_allocator,
    int _global_num_q_heads,
    int _global_num_kv_heads,
    int _num_q_heads,
    int _num_kv_heads,
    int _num_kv_cache_pages,
    DataType _quantization_type,
    bool _offload)
    : OpMeta(handler, attn) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  checkCUDNN(cudnnCreateTensorDescriptor(&qk_tensor));
  // assume dimensions match for now
  qProjSize = _qProjSize;
  kProjSize = _kProjSize;
  vProjSize = _vProjSize;
  oProjSize = _oProjSize;
  assert(qProjSize == kProjSize &&
         kProjSize == vProjSize); // required for attention QK.T matmul
  size_t size_of_dt = data_type_size(attn->data_type);
  quantization_type = _quantization_type;
  offload = _offload;

  global_num_q_heads = _global_num_q_heads;
  global_num_kv_heads = _global_num_kv_heads;
  num_q_heads = _num_q_heads;
  num_kv_heads = _num_kv_heads;

  rotary_embedding_meta =
      (RotaryEmbeddingMeta *)calloc(1, sizeof(RotaryEmbeddingMeta));
  *rotary_embedding_meta = _rotary_embedding_meta;
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  position_bias = (bool *)calloc(1, sizeof(bool));
  *position_bias = _position_bias;

  num_kv_cache_pages = _num_kv_cache_pages;
  assert(num_kv_cache_pages > 0 || enable_peft_finetuning);

  // spec decoding and peft finetuning are mutually exclusive
  if (enable_peft_finetuning) {
    assert(infer_mode == INC_DECODING_MODE);
  }

  size_t inf_instance_size = 0;
  size_t kv_cache_instance_size = 0;
  size_t peft_instance_size = 0;

  // Compute total GPU memory size needed
  {
    // 1. GQA pointers for batch matmul. Used by PEFT and spec_inc if
    // num_q_heads > num_kv_heads
    if (num_q_heads > num_kv_heads &&
        (infer_mode == BEAM_SEARCH_MODE || enable_peft_finetuning)) {
      assert(num_q_heads % num_kv_heads == 0 &&
             "num_q_heads must be divisible by num_kv_heads");
      assert(attn->data_type == DT_FLOAT ||
             attn->data_type == DT_HALF && "Unsupported data type");
      gqa_ptr_array_size = num_q_heads * sizeof(void *);
      if (infer_mode == BEAM_SEARCH_MODE) {
        inf_instance_size += 3 * gqa_ptr_array_size; // fwd
      } else if (enable_peft_finetuning) {
        inf_instance_size += 3 * gqa_ptr_array_size;  // fwd
        peft_instance_size += 3 * gqa_ptr_array_size; // bwd
      }
    }

    // 2. KV cache
    key_cache_size = value_cache_size =
        num_kv_heads * kProjSize * kPagesize * num_kv_cache_pages;
    if (infer_mode == BEAM_SEARCH_MODE || infer_mode == TREE_VERIFY_MODE) {
      // a K-ary tree max node is (k^n - 1) / 2
      assert(key_cache_size == value_cache_size);
      assert(key_cache_size >=
             num_kv_heads * kProjSize *
                 BeamSearchBatchConfig::max_requests_per_batch() *
                 (BatchConfig::max_sequence_length() +
                  BatchConfig::max_spec_tree_token_num()));
    }
    kv_cache_instance_size += (key_cache_size + value_cache_size) * size_of_dt;
    if (enable_peft_finetuning) {
      // add kv cache for single sequence
      peft_key_cache_size = peft_value_cache_size =
          num_kv_heads * kProjSize * BatchConfig::max_sequence_length();
      peft_instance_size +=
          (peft_key_cache_size + peft_value_cache_size) * size_of_dt;
    }

    // 3. buffers for intermediate results
    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int max_tokens_per_batch = (infer_mode == TREE_VERIFY_MODE)
                                   ? BatchConfig::max_verify_tokens_per_batch()
                                   : BatchConfig::max_tokens_per_batch();
    // devQKVProjArray
    qkv_max_proj_size = qProjSize * tot_num_heads * max_tokens_per_batch;
    inf_instance_size += qkv_max_proj_size * size_of_dt;
    if (enable_peft_finetuning) {
      qkv_max_proj_size_bwd =
          qProjSize * tot_num_heads * BatchConfig::max_sequence_length();

      peft_instance_size += qkv_max_proj_size_bwd * size_of_dt;
    }
    // queryTmp and outputTmp: only for paged attention
    if (infer_mode == INC_DECODING_MODE) {
      query_tmp_size = num_q_heads * qProjSize * max_tokens_per_batch;
      inf_instance_size += (query_tmp_size)*size_of_dt;
    }
    // complex_input & complex_input_bwd
    complex_size = max_tokens_per_batch * qProjSize *
                   (num_q_heads + num_kv_heads) /
                   2; // only used for Q and K, not V
    inf_instance_size += complex_size * sizeof(cuFloatComplex);
    if (enable_peft_finetuning) {
      complex_size_bwd = BatchConfig::max_sequence_length() * qProjSize *
                         (num_q_heads + num_kv_heads) /
                         2; // only used for Q and K, not V
      peft_instance_size += complex_size_bwd * sizeof(cuFloatComplex);
    }
    // QK prods and QK prods (softmax)
    if (infer_mode == BEAM_SEARCH_MODE) {
      qk_prod_size = max_tokens_per_batch * BatchConfig::max_sequence_length() *
                     num_q_heads;
      inf_instance_size += 2 * qk_prod_size * size_of_dt;
    } else if (enable_peft_finetuning) {
      // only need one copy as they can be reused by PEFT fwd and PEFT bwd, as
      // they never run concurrently
      qk_prod_size = BatchConfig::max_sequence_length() *
                     BatchConfig::max_sequence_length() * num_q_heads;
      peft_instance_size += qk_prod_size * size_of_dt;
    }
    // PEFT partial results buffers
    if (enable_peft_finetuning) {
      allocated_peft_buffer_size1 = BatchConfig::max_sequence_length() *
                                    num_q_heads * qProjSize * size_of_dt;
      allocated_peft_buffer_size2 = BatchConfig::max_sequence_length() *
                                    BatchConfig::max_sequence_length() *
                                    num_q_heads * size_of_dt;
      peft_token_infos = (BatchConfig::PerTokenInfo *)calloc(
          1,
          sizeof(BatchConfig::PerTokenInfo) *
              BatchConfig::max_sequence_length());
      peft_token_infos_size = sizeof(BatchConfig::PerTokenInfo) *
                              BatchConfig::max_sequence_length();
      peft_instance_size +=
          allocated_peft_buffer_size1 + allocated_peft_buffer_size2;
      peft_instance_size += peft_token_infos_size;
    }

    // 4. offload: TBD
    if (offload) {
      assert(false && "TODO");
    }
  }

  // Allocate chunk of memory
  inf_mem_allocator.create_legion_instance(
      inf_instance, inf_instance_size, "IncMultiHeadSelfAttentionMeta (inf)");
  kv_cache_mem_allocator.create_legion_instance(
      kv_cache_instance, kv_cache_instance_size, "KV Cache");
  peft_mem_allocator.create_legion_instance(
      peft_instance,
      peft_instance_size,
      "IncMultiHeadSelfAttentionMeta (peft)");

  // Assign pointers from chunk of memory
  {
    // gqa pointers
    if (num_q_heads > num_kv_heads) {
      assert(num_q_heads % num_kv_heads == 0 &&
             "Num Q heads must be a multiple of num KV heads");
      d_A_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      d_B_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      d_C_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      if (enable_peft_finetuning) {
        d_A_array2 = (void **)peft_mem_allocator.allocate_instance_untyped(
            gqa_ptr_array_size);
        d_B_array2 = (void **)peft_mem_allocator.allocate_instance_untyped(
            gqa_ptr_array_size);
        d_C_array2 = (void **)peft_mem_allocator.allocate_instance_untyped(
            gqa_ptr_array_size);
      }
    }

    // KV cache
    if (infer_mode == INC_DECODING_MODE) {
      kvCache = kv_cache_mem_allocator.allocate_instance_untyped(
          (key_cache_size + value_cache_size) * size_of_dt);
      keyCache = valueCache = nullptr;
    } else {
      kvCache = nullptr;
      keyCache = kv_cache_mem_allocator.allocate_instance_untyped(
          key_cache_size * size_of_dt);
      valueCache = kv_cache_mem_allocator.allocate_instance_untyped(
          value_cache_size * size_of_dt);
    }
    if (enable_peft_finetuning) {
      assert(infer_mode == INC_DECODING_MODE);
      keyCachePeft = peft_mem_allocator.allocate_instance_untyped(
          peft_key_cache_size * size_of_dt);
      valueCachePeft = peft_mem_allocator.allocate_instance_untyped(
          peft_value_cache_size * size_of_dt);
    } else {
      keyCachePeft = valueCachePeft = nullptr;
    }

    // intermediate buffers
    // devQKVProjArray: used to store QKV proj so that we can modify them (apply
    // rope, etc)
    devQKVProjArray = inf_mem_allocator.allocate_instance_untyped(
        qkv_max_proj_size * size_of_dt);
    // devQKVProjArrayBWD
    if (enable_peft_finetuning) {
      devQKVProjArrayBWD = peft_mem_allocator.allocate_instance_untyped(
          qkv_max_proj_size_bwd * size_of_dt);
    }
    // queryTmp and outputTmp: only for paged attention
    if (infer_mode == INC_DECODING_MODE) {
      queryTmp = inf_mem_allocator.allocate_instance_untyped(query_tmp_size *
                                                             size_of_dt);
    }
    // complex input
    complex_input =
        inf_mem_allocator.allocate_instance<cuFloatComplex>(complex_size);
    complex_input_bwd =
        peft_mem_allocator.allocate_instance<cuFloatComplex>(complex_size_bwd);
    // qk_prods, qk_prods_softmax
    if (infer_mode == BEAM_SEARCH_MODE) {
      qk_prods = inf_mem_allocator.allocate_instance_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = inf_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
    }
    if (enable_peft_finetuning) {
      qk_prods_softmax = peft_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
    }
    // peft partial result buffers
    if (enable_peft_finetuning) {
      query_activation_buffer = peft_mem_allocator.allocate_instance_untyped(
          allocated_peft_buffer_size1);
      softmax_activation_buffer = peft_mem_allocator.allocate_instance_untyped(
          allocated_peft_buffer_size2);
      peft_token_infos_device =
          (BatchConfig::PerTokenInfo *)peft_mem_allocator
              .allocate_instance_untyped(peft_token_infos_size);
    }

    token_infos = static_cast<BatchConfig::PerTokenInfo *>(
        handler.batch_config_metadata->tokens_info);
    request_infos = static_cast<BatchConfig::PerRequestInfo *>(
        handler.batch_config_metadata->requestsInfo);
    request_completed =
        static_cast<bool *>(handler.batch_config_metadata->request_completed);

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
    }
  }

  // set attention constants
  // std::cerr << "Enabling incr attention metadata for handler incr meta: "
  //           << handler.incr_attention_metadata << std::endl;
  handler.incr_attention_metadata->set_enabled(true);
  handler.incr_attention_metadata->set_num_q_heads(num_q_heads);
  handler.incr_attention_metadata->set_num_kv_heads(num_kv_heads);
  handler.incr_attention_metadata->set_head_dim(qProjSize);

  cudaStreamSynchronize(stream);
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (inf_instance != Realm::RegionInstance::NO_INST) {
    inf_instance.destroy();
  }
  if (kv_cache_instance != Realm::RegionInstance::NO_INST) {
    kv_cache_instance.destroy();
  }
  if (peft_instance != Realm::RegionInstance::NO_INST) {
    peft_instance.destroy();
  }
}

template void Kernels::IncMultiHeadAttention::run_batched_matmul<half>(
    IncMultiHeadSelfAttentionMeta const *meta,
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    half const *A,
    cudaDataType Atype,
    int lda,
    long long int strideA,
    half const *B,
    cudaDataType Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    half *C,
    cudaDataType Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo,
    cudaStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::run_batched_matmul<float>(
    IncMultiHeadSelfAttentionMeta const *meta,
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    float const *A,
    cudaDataType Atype,
    int lda,
    long long int strideA,
    float const *B,
    cudaDataType Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    float *C,
    cudaDataType Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo,
    cudaStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    float *output_ptr,
    cudaStream_t inf_stream);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    half *output_ptr,
    cudaStream_t inf_stream);

template void
    Kernels::IncMultiHeadAttention::update_kv_cache_kernel_flashinfer<float>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::update_kv_cache_kernel_flashinfer<half>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        cudaStream_t stream);

template __global__ void
    Kernels::IncMultiHeadAttention::apply_position_bias_qkprd<float>(
        float *input_ptr,
        int num_tokens,
        int num_total_tokens,
        int num_heads,
        int global_num_q_heads,
        int shard_id);

template __global__ void
    Kernels::IncMultiHeadAttention::apply_position_bias_qkprd<half>(
        half *input_ptr,
        int num_tokens,
        int num_total_tokens,
        int num_heads,
        int global_num_q_heads,
        int shard_id);

}; // namespace FlexFlow
