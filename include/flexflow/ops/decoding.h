#ifndef _FLEXFLOW_DECODING_H
#define _FLEXFLOW_DECODING_H

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/decoding_params.h"
#include "flexflow/utils/memory_allocator.h"
#include "flexflow/fftype.h"
#include "flexflow/device.h"

namespace FlexFlow {

// forward declaration
class DecodingMeta;
struct SoftmaxShardedContext;

class Decoding : public Op {
public:
  using Params = DecodingParams;
  using Input = ParallelTensor;
  Decoding(FFModel &model,
           LayerID const &_layer_guid,
           const ParallelTensor input,
           bool beam_search,
           int tensor_parallelism_degree,
           char const *name);
  Decoding(FFModel &model,
           Params const &params,
           const Input input,
           char const *name = nullptr);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
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
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static BeamInferenceResult
      inference_task_beam(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  static InferenceResult
      inference_task_norm(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  static bool peft_bwd_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  Params get_params() const;
  
  template <typename DT>
  static void inference_kernel(DecodingMeta const *m,
                               BatchConfig const *bc,
                               DT const *input_ptr,
                               DT *softmax_output_ptr,
                               int *argmax_output_ptr,
                               int num_classes,
                               int vocab_offset,
                               float *loss,
                               ffStream_t stream);
  static void inference_kernel_wrapper(DecodingMeta *m,
                                        BatchConfig const *bc,
                                        GenericTensorAccessorR const &input,
                                        GenericTensorAccessorW const &softmax_output,
                                        GenericTensorAccessorW const &argmax_output);
  template <typename DT>
  static void peft_bwd_kernel(DecodingMeta const *m,
                     BatchConfig const *bc,
                     DT *input_grad_ptr,
                     int num_classes,
                     int shard_id,
                     ffStream_t stream);
  static void peft_bwd_kernel_wrapper(DecodingMeta *m,
                                     BatchConfig const *bc,
                                     int shard_id,
                                     GenericTensorAccessorW const &input_grad);

public:
  LayerID layer_guid;
  bool beam_search;
  int tensor_parallelism_degree;
};

class DecodingMeta : public OpMeta {
public:
  DecodingMeta(FFHandler handler,
               Decoding const *decoding,
               Legion::Domain const &input_domain,
               MemoryAllocator &gpu_mem_allocator);
  ~DecodingMeta(void);
  bool beam_search;
  float *probs;
  float *d_loss;
  // Temporary buffers 
  int *parent_output_buffer;
  // Sharded softmax context
  SoftmaxShardedContext *softmax_context;
  // PEFT related fields
  void *output_grad_ptr = nullptr;
  size_t allocated_peft_buffer_size = 0;
  Realm::RegionInstance reserveInst;
  BatchConfig::TokenId peft_token_ids[BatchConfig::MAX_NUM_TOKENS];
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_DECODING_H
