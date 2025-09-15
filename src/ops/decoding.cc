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

#include "flexflow/ops/decoding.h"
#include "flexflow/model.h"
#include "flexflow/ops/fused.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
using PCG::Node;


/* Params */
bool operator==(DecodingParams const &lhs, DecodingParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.beam_search == rhs.beam_search && 
         lhs.tensor_parallelism_degree == rhs.tensor_parallelism_degree;
}

void Decoding::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->beam_search);
  sez.serialize(this->tensor_parallelism_degree);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

/*static*/
Node Decoding::deserialize(FFModel &ff,
                           Legion::Deserializer &dez,
                           ParallelTensor inputs[],
                           int num_inputs) {
  assert(num_inputs == 1);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  bool beam_search;
  dez.deserialize(beam_search);
  int tensor_parallelism_degree;
  dez.deserialize(tensor_parallelism_degree);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

  DecodingParams params;
  params.layer_guid = layer_guid;
  params.beam_search = beam_search;
  params.tensor_parallelism_degree = tensor_parallelism_degree;
  strcpy(params.name, name);
  return ff.get_or_create_node<Decoding>(inputs[0], params);
}

bool DecodingParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

DecodingParams Decoding::get_params() const {
  DecodingParams params;
  params.layer_guid = this->layer_guid;
  params.beam_search = this->beam_search;
  params.tensor_parallelism_degree = this->tensor_parallelism_degree;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }
  return params;
}

Tensor FFModel::decoding(const Tensor input, bool beam_search, char const *name) {
  Layer *li = new Layer(this,
                        OP_DECODING,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        2 /*outputs*/,
                        input);

  printf("Adding decoding layer\n");
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    
    // First output: softmax output (same dimensions as input)
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, input->data_type, li, 0, true /*create_grad*/);
    
    // Second output: argmax output (batch dimensions only, int type)
    for (int i = 1; i < numdims; i++) {
      dims[i-1] = input->dims[i];  // Shift batch dimensions down
    }
    int argmax_numdims = numdims - 1;  // Remove vocab dimension
    li->outputs[1] = create_tensor_legion_ordering(
        argmax_numdims, dims, DT_INT32, li, 1, false /*create_grad*/);
  }
  li->add_int_property("beam_search", beam_search);
  li->add_int_property("tensor_parallelism_degree", config.tensor_parallelism_degree);
  layers.push_back(li);
  return li->outputs[1]; // Return argmax output for compatibility
}

Op *Decoding::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("beam_search", value);
  bool beam_search = (bool)value;
  layer->get_int_property("tensor_parallelism_degree", value);
  int tensor_parallelism_degree = (int)value;
  return new Decoding(model, layer->layer_guid, inputs[0], beam_search, tensor_parallelism_degree, layer->name);
}

static std::string remove_uid(char const *op_name) {
  std::string op_name_without_uid = std::string(op_name);
  size_t last_underscore = op_name_without_uid.length();
  for (int i = op_name_without_uid.length() - 1; i > 0; i--) {
    if (!(std::isdigit(op_name[i]) || op_name[i] == '_')) {
      break;
    } else if (op_name[i] == '_') {
      last_underscore = i;
    }
  }
  if (last_underscore < op_name_without_uid.length()) {
    op_name_without_uid.erase(last_underscore);
  }
  return op_name_without_uid;
}

Decoding::Decoding(FFModel &model,
                   LayerID const &_layer_guid,
                   const ParallelTensor _input,
                   bool _beam_search,
                   int _tensor_parallelism_degree,
                   char const *name)
    : Op(model,
         OP_DECODING,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         2 /*outputs*/,
         _input),
      beam_search(_beam_search), tensor_parallelism_degree(_tensor_parallelism_degree) {
  layer_guid = _layer_guid;
  int numdim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  
  // First output: softmax output (same dimensions as input)
  for (int i = 0; i < numdim; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, _input->data_type, this, 0 /*owner_idx*/);

  // Second output: argmax results (collapse vocab dimension to 1, keep batch dimensions)
  for (int i = 1; i < numdim; i++) {
    dims[i-1] = inputs[0]->dims[i];  // Shift batch dimensions down
  }
  int argmax_numdim = numdim - 1;  // Remove vocab dimension
  dims[argmax_numdim - 1].size = inputs[0]->dims[0].degree;
  dims[argmax_numdim - 1].degree = inputs[0]->dims[0].degree;
  dims[argmax_numdim - 1].parallel_idx = inputs[0]->dims[0].parallel_idx;
  outputs[1] = model.create_parallel_tensor_legion_ordering(
      argmax_numdim, dims, DT_INT32, this, 1 /*owner_idx*/);

  std::string const &input_label = remove_uid(name) + std::string(" input tensor");
  _input->print(input_label);
  std::string const &softmax_label = remove_uid(name) + std::string(" softmax output tensor");
  outputs[0]->print(softmax_label);
  std::string const &argmax_label = remove_uid(name) + std::string(" argmax output tensor");
  outputs[1]->print(argmax_label);
}

Decoding::Decoding(FFModel &model,
                   DecodingParams const &params,
                   const ParallelTensor input,
                   char const *name)
    : Decoding(model, params.layer_guid, input, params.beam_search, params.tensor_parallelism_degree, params.name) {}

struct DecodingInitMeta {
  Decoding *decoding;
};

void Decoding::init_inference(FFModel const &ff,
                              std::vector<ParallelTensor> const &batch_inputs,
                              std::vector<ParallelTensor> const &batch_outputs,
                              MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);

  DecodingInitMeta meta;
  meta.decoding = this;

  IndexLauncher launcher(DECODING_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(&meta, sizeof(DecodingInitMeta)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_DISCARD,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_DISCARD,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void Decoding::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(DECODING_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Decoding)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_DISCARD,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0]: input
  regions[1]: softmax output
  regions[2]: argmax output
 */
OpMeta *Decoding::init_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == regions.size());
  DecodingInitMeta const *meta = (DecodingInitMeta *)task->args;
  Decoding const *decoding = meta->decoding;

  FFHandler handle = *((FFHandler const *)task->local_args);
  Memory gpu_mem = get_proc_mem(Machine::get_machine(), task->target_proc);
  MemoryAllocator gpu_mem_allocator(gpu_mem);

  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain softmax_output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain argmax_output_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  // Note: softmax_output_domain should match input_domain, argmax_output_domain has one fewer dimension
  assert(input_domain == softmax_output_domain);
  int ndims = input_domain.get_dim();
  int output_ndims = ndims - 1; // Argmax output has one fewer dimension
  Domain domain;
  for (int i = 0; i < ndims - 1; i++) {
    assert(!decoding->outputs[0]->dims[i].is_replica_dim);
  }
  // Only the outter-most dim can be a replica_dim
  if (decoding->outputs[0]->dims[ndims - 1].is_replica_dim) {
    int replica_degree = decoding->outputs[0]->dims[ndims - 1].size;
    domain.dim = ndims - 1;
    for (int i = 0; i < ndims - 1; i++) {
      domain.rect_data[i] = input_domain.rect_data[i];
      domain.rect_data[i + ndims - 1] = input_domain.rect_data[i + ndims];
    }
    domain.rect_data[2 * ndims - 3] =
        (domain.rect_data[2 * ndims - 3] + 1) * replica_degree - 1;
    assert(domain.get_volume() == input_domain.get_volume());
  } else {
    domain = input_domain;
  }
  
  DecodingMeta *m =
      new DecodingMeta(handle, decoding, domain, gpu_mem_allocator);
  std::strcpy(m->op_name, decoding->name);
  m->layer_guid = decoding->layer_guid;
  m->beam_search = decoding->beam_search;
  return m;
}

void Decoding::forward(FFModel const &ff) {
  // Decoding does not support forward
  assert(false);
}

FutureMap Decoding::inference(FFModel const &ff,
                              BatchConfigFuture const &bc,
                              std::vector<ParallelTensor> const &batch_inputs,
                              std::vector<ParallelTensor> const &batch_outputs,
                              MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();

  assert(ff.config.computationMode == COMP_MODE_INFERENCE);

  if (beam_search) {
    IndexLauncher launcher(DECODING_BEAM_INF_TASK_ID,
                           parallel_is,
                           TaskArgument(nullptr, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           machine_view_hash);
    launcher.add_future(bc);
    launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      batch_inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_outputs[0]->region));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_outputs[1]->region));
    launcher.add_field(2, FID_DATA);
    return runtime->execute_index_space(ctx, launcher);
  } else {
    IndexLauncher launcher(DECODING_NORM_INF_TASK_ID,
                           parallel_is,
                           TaskArgument(nullptr, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           machine_view_hash);
    launcher.add_future(bc);
    launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      batch_inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_outputs[0]->region));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_outputs[1]->region));
    launcher.add_field(2, FID_DATA);
    return runtime->execute_index_space(ctx, launcher);
  }
}

BeamInferenceResult
    Decoding::inference_task_beam(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    BeamInferenceResult ir;
    return ir;
  }
  DecodingMeta *m = *((DecodingMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW softmax_output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW argmax_output = helperGetGenericTensorAccessorWO(
      m->output_type[1], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int batch_size = bc->num_active_tokens();
  float loss = 0.0f;
  
  inference_kernel_wrapper(m, bc, input, softmax_output, argmax_output);
  
  BeamInferenceResult ir;
  // Copy argmax results from output region
  copy_tensor_dev_to_host<BatchConfig::TokenId>(
      argmax_output.get_int32_ptr(), ir.token_ids, batch_size);
  copy_tensor_dev_to_host(m->probs, ir.probs, batch_size);
  // Copy parent results from temporary buffer in DecodingMeta
  copy_tensor_dev_to_host<int>(
      m->parent_output_buffer, ir.parent_id, batch_size);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    // Save inference tensors to file (implementation needed)
    // Decoding::save_inference_tensors_to_file(
    //     m, shard_id, bc, {}, {}, {input, argmax_output});
  }

  return ir;
}

InferenceResult
    Decoding::inference_task_norm(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  DecodingMeta *m = *((DecodingMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    InferenceResult ir;
    return ir;
  }

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW softmax_output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW argmax_output = helperGetGenericTensorAccessorWO(
      m->output_type[1], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int batch_size = bc->num_active_tokens();
  float loss = 0.0f;

  inference_kernel_wrapper(m, bc, input, softmax_output, argmax_output);

  if (task->index_point.point_data[0] == 0) {
    int in_dim0 = input.domain.hi()[0] - input.domain.lo()[0] + 1;
    int in_dim1 = input.domain.hi()[1] - input.domain.lo()[1] + 1;
    int softmax_out_dim0 = softmax_output.domain.hi()[0] - softmax_output.domain.lo()[0] + 1;
    int softmax_out_dim1 = softmax_output.domain.hi()[2] - softmax_output.domain.lo()[2] + 1;
    int argmax_out_dim0 = argmax_output.domain.hi()[1] - argmax_output.domain.lo()[1] + 1;
    std::string op_name_without_uid = remove_uid(m->op_name);
    printf("Decoding(%s): in=[%i, bz=%i/%i] -> softmax_out=[%i,bz=%i/%i], argmax_out=[bz=%i]\n",
           op_name_without_uid.c_str(),
           in_dim0, bc->num_tokens, in_dim1,
           softmax_out_dim0, bc->num_tokens, softmax_out_dim1,
           argmax_out_dim0);
  }

  InferenceResult ir;
  ir.finetuning_loss = loss;

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    // Save inference tensors to file (implementation needed)
    Decoding::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, {}, {softmax_output, argmax_output});
  } else {
    m->decoding_step++;
  }

  // Copy argmax results from output region
  if (task->index_point.point_data[0] == 0) {
    copy_tensor_dev_to_host<BatchConfig::TokenId>(
        argmax_output.get_int32_ptr(), ir.token_ids, batch_size);
  }

  return ir;
}

void Decoding::backward(FFModel const &ff) {
  // Decoding does not support backward
  assert(false);
}

FutureMap Decoding::peft_bwd(FFModel const &ff,
                             BatchConfigFuture const &bc,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();
  IndexLauncher launcher(DECODING_PEFT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[0]->part_grad,
                        0 /*projection id*/,
                        reset_input_grads[0] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE,
                        batch_inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

bool Decoding::peft_bwd_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(task->regions.size() == regions.size());
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  DecodingMeta *m = *((DecodingMeta **)task->local_args);
  if (!bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id)) {
    return false;
  }
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);

  peft_bwd_kernel_wrapper(m, bc, task->index_point.point_data[0], input_grad);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    // Save inference tensors to file (implementation needed)
    Decoding::save_inference_tensors_to_file(
        m, shard_id, bc, {input_grad}, {}, {}, false);
  }
  return true;
}

bool Decoding::measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const {
  return false;
}

Op *Decoding::materialize(FFModel &ff,
                          ParallelTensor inputs[],
                          int num_inputs) const {
  DecodingParams params = get_params();
  return new Decoding(ff, params, inputs[0], this->name);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::DecodingParams>::operator()(
    FlexFlow::DecodingParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.beam_search);
  hash_combine(key, params.tensor_parallelism_degree);
  return key;
}
}; // namespace std
