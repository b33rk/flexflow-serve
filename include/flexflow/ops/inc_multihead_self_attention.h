#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/inc_multihead_self_attention_params.h"
#include "flexflow/utils/memory_allocator.h"
#include "math.h"
#include <cfloat>
#include <complex>
#if defined(FF_USE_HIP_ROCM)
#include <hip/hip_complex.h>
#endif

namespace FlexFlow {

class IncMultiHeadSelfAttentionMeta;

class IncMultiHeadSelfAttention : public Op {
public:
  using Params = IncMultiHeadSelfAttentionParams;
  using Input = ParallelTensor;

  IncMultiHeadSelfAttention(FFModel &model,
                            LayerID const &layer_guid,
                            ParallelTensor const _input,
                            int _embed_dim,
                            int _num_q_heads,
                            int _num_kv_heads,
                            int _kdim,
                            int _vdim,
                            float _dropout,
                            bool _add_zero_attn,
                            RotaryEmbeddingMeta _rotary_embedding_meta,
                            bool _scaling_query,
                            float _scaling_factor,
                            bool _qk_prod_scaling,
                            bool _position_bias,
                            DataType _quantization_type,
                            bool _offload,
                            int _tensor_parallelism_degree,
                            int _num_kv_cache_pages,
                            char const *name);
  IncMultiHeadSelfAttention(FFModel &model,
                            ParallelTensor const _input,
                            int _embed_dim,
                            int _num_q_heads,
                            int _num_kv_heads,
                            int _kdim,
                            int _vdim,
                            float _dropout,
                            bool _add_zero_attn,
                            RotaryEmbeddingMeta _rotary_embedding_meta,
                            bool _scaling_query,
                            float _scaling_factor,
                            bool _qk_prod_scaling,
                            bool _position_bias,
                            DataType _quantization_type,
                            bool _offload,
                            int _tensor_parallelism_degree,
                            int _num_kv_cache_pages,
                            char const *name);
  IncMultiHeadSelfAttention(FFModel &model,
                            IncMultiHeadSelfAttention const &other,
                            ParallelTensor const input);
  IncMultiHeadSelfAttention(FFModel &model,
                            Params const &params,
                            Input const &inputs,
                            char const *name = nullptr);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  Legion::FutureMap peft_bwd(FFModel const &,
                             BatchConfigFuture const &,
                             std::vector<ParallelTensor> const &,
                             std::vector<ParallelTensor> const &,
                             MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  bool get_int_parameter(PMParameter, int *) const override;

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static bool peft_bwd_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;
  static void inference_kernel_wrapper(IncMultiHeadSelfAttentionMeta *m,
                                       BatchConfig const *bc,
                                       int shard_id,
                                       GenericTensorAccessorR const &input,
                                       GenericTensorAccessorW const &output);
  static void
      peft_bwd_kernel_wrapper(IncMultiHeadSelfAttentionMeta *m,
                              BatchConfig const *bc,
                              int shard_id,
                              GenericTensorAccessorW const &input_grad,
                              GenericTensorAccessorR const &output_grad);
  Params get_params() const;

public:
  int num_q_heads, num_kv_heads, tensor_parallelism_degree, num_kv_cache_pages;
  float dropout, scaling_factor;
  bool add_zero_attn, scaling_query, qk_prod_scaling, position_bias;
  RotaryEmbeddingMeta rotary_embedding_meta;
  int qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
  DataType quantization_type;
  bool offload;
};

class IncMultiHeadSelfAttentionMeta : public OpMeta {
public:
  IncMultiHeadSelfAttentionMeta(FFHandler handler,
                                IncMultiHeadSelfAttention const *attn,
                                MemoryAllocator &gpu_mem_allocator,
                                int _num_q_heads,
                                int _num_kv_heads);
  IncMultiHeadSelfAttentionMeta(FFHandler handler,
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
                                MemoryAllocator &gpu_mem_allocator,
                                int _global_num_q_heads,
                                int _global_num_kv_heads,
                                int _num_q_heads,
                                int _num_kv_heads,
                                int _num_kv_cache_pages,
                                DataType _quantization_type,
                                bool _offload);
  ~IncMultiHeadSelfAttentionMeta(void);

public:
  Realm::RegionInstance reserveInst;
  size_t reserveSpaceSize;
  int qProjSize, kProjSize, vProjSize, oProjSize;
  int global_num_q_heads, global_num_kv_heads, num_q_heads, num_kv_heads;
  RotaryEmbeddingMeta *rotary_embedding_meta;
  bool *scaling_query;
  bool *qk_prod_scaling;
  bool *position_bias;
  float scaling_factor;
  DataType quantization_type;
  bool offload;
  int num_kv_cache_pages;

  // GPU memory sizes (or num elements)
  size_t gqa_ptr_array_size = 0;
  size_t key_cache_size = 0, value_cache_size = 0;           // numel
  size_t peft_key_cache_size = 0, peft_value_cache_size = 0; // numel
  size_t qkv_max_proj_size, qkv_max_proj_size_bwd = 0;       // numel
  size_t query_tmp_size = 0, output_tmp_size = 0;            // numel
  size_t complex_size = 0, complex_size_bwd = 0;             // numel
  size_t qk_prod_size = 0;                                   // numel
  size_t allocated_peft_buffer_size1 = 0, allocated_peft_buffer_size2 = 0,
         peft_token_infos_size = 0;

  void *devQKVProjArray, *devQKVProjArrayBWD;
  void *kvCache, *keyCache, *valueCache;
  void *keyCachePeft, *valueCachePeft;
  void *qk_prods, *qk_prods_softmax;
  // flashinfer
  void *queryTmp, *outputTmp;

  BatchConfig::PerTokenInfo *token_infos;
  BatchConfig::PerRequestInfo *request_infos;
  bool *request_completed;

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t qk_tensor;
  cuFloatComplex *complex_input, *complex_input_bwd;
#elif defined(FF_USE_HIP_ROCM)
  miopenTensorDescriptor_t qk_tensor;
  hipFloatComplex *complex_input, *complex_input_bwd;
#endif

  // GQA
  void **d_A_array, **d_B_array, **d_C_array;

  // PEFT specific fields
  void **d_A_array2, **d_B_array2, **d_C_array2;
  void *softmax_activation_buffer;
  void *query_activation_buffer;
  BatchConfig::PerTokenInfo *peft_token_infos = nullptr;
  BatchConfig::PerTokenInfo *peft_token_infos_device;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_H
