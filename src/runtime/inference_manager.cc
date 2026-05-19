/* Copyright 2023 CMU, Stanford, Facebook, LANL
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
#include "flexflow/graph.h"
#include "flexflow/model.h"
#include "flexflow/ops/fused.h"
#include "flexflow/ops/noop.h"
#include "flexflow/parallel_ops/parallel_op.h"
#include "flexflow/request_manager.h"

namespace FlexFlow {

using namespace Legion;

Legion::Logger log_inf_mgr("InferenceManager");
Legion::Logger log_offload("Offloading");

InferenceManager::InferenceManager() {}

InferenceManager *inference_manager_singleton = nullptr;

/*static*/
InferenceManager *InferenceManager::get_inference_manager() {
  if (inference_manager_singleton == nullptr) {
    // FFConfig ffconfig;
    inference_manager_singleton = new InferenceManager();
  }
  return inference_manager_singleton;
}

bool parallel_tensor_list_overlaps(std::vector<ParallelTensor> const &list1,
                                   std::vector<ParallelTensor> const &list2) {
  for (auto const &pt1 : list1) {
    for (auto const &pt2 : list2) {
      if (pt1 == pt2) {
        return true;
      }
    }
  }
  return false;
}

void InferenceManager::compile_model_and_allocate_buffer(FFModel *model) {

  // Check if the model object exists
  if (model == nullptr) {
    std::cout << "###PEFT DEBUGGING### Model object does not exist."
              << std::endl;
    return; // Early return to prevent further operations on a nullptr
  } else {
    std::cout << "###PEFT DEBUGGING### Model object exists." << std::endl;
  }

  // TODO: currently assume there is a single data-parallel pipeline
  // (i.e., data-parallel-degree == 1)
  assert(model->config.data_parallelism_degree == 1);
  model->config.batchSize = BatchConfig::max_tokens_per_batch();

  // Check if the model object exists after importing config
  if (model == nullptr) {
    std::cout << "###PEFT DEBUGGING### Model object does not exist after "
                 "setting config and batch size."
              << std::endl;
    return; // Early return to prevent further operations on a nullptr
  } else {
    std::cout << "###PEFT DEBUGGING### Model object still exists." << std::endl;
  }

  model->compile_inference();
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;

  // std::cout << std::endl << std::endl << "Operators MVs:" << std::endl;
  int num_transformer_layers_per_stage =
      model->current_transformer_layer_id /
          model->config.pipeline_parallelism_degree +
      1;
  int degree = model->config.data_parallelism_degree *
               model->config.tensor_parallelism_degree;

  for (int op_idx = 0; op_idx < model->operators.size(); op_idx++) {
    Op const *op = model->operators[op_idx];
    // Skip weight operators
    if (op->op_type == OP_WEIGHT) {
      continue;
    }
    // Get machine views
    std::vector<MachineView> machine_views;
    for (int j = 0; j < model->config.data_parallelism_degree; j++) {
      MachineView mv;
      mv.device_type = MachineView::GPU;
      mv.ndims = 1;
      // mv.start_device_id = 0;
      mv.stride[0] = 1;
      int parallel_degree = 1;
      for (int k = 0; k < op->outputs[0]->num_dims; k++) {
        parallel_degree *= op->outputs[0]->dims[k].degree;
      }
      mv.dim[0] = parallel_degree;
      LayerID layer_guid = op->layer_guid;
      if (op->op_type == OP_INPUT) {
        // All inputs are assigned to the first stage
        layer_guid.transformer_layer_id = 0;
      } else if (layer_guid == LayerID::NO_ID) {
        Op const *op_with_guid = op;
        // Assert that we only have a single input
        while (op_with_guid->layer_guid == LayerID::NO_ID) {
          assert(op_with_guid->numInputs == 1);
          op_with_guid = op_with_guid->inputs[0]->owner_op;
          assert(op_with_guid != nullptr);
        }
        layer_guid = op_with_guid->layer_guid;
      }
      mv.start_device_id = degree * (layer_guid.transformer_layer_id /
                                     num_transformer_layers_per_stage);
      assert(mv == op->outputs[0]->machine_view);
      machine_views.push_back(mv);
    }
    // std::cout << "operator: " << op->name << std::endl;
    // for (int i = 0; i < op->numInputs; i++) {
    //   op->inputs[i]->print("input pt");
    //   std::cout << "input mv: " << op->inputs[i]->machine_view << std::endl;
    // }
    // std::cout << "Op " << op->name << ": ";
    for (int i = 0; i < op->numOutputs; i++) {
      ParallelTensor pt_base = op->outputs[i];
      assert(tensor_buffer.find(pt_base) == tensor_buffer.end());
      // no need to map inplace tensor
      // A tensor is inplace if it shares the same region as another tensor
      {
        bool inplace = false;
        for (int j = 0; j < op->numInputs; j++) {
          if (op->inputs[j]->region == op->outputs[i]->region) {
            assert(tensor_buffer.find(op->inputs[j]) != tensor_buffer.end());
            tensor_buffer[pt_base] = tensor_buffer[op->inputs[j]];
            inplace = true;
          }
        }
        for (int j = 0; j < i; j++) {
          if (op->outputs[j]->region == op->outputs[i]->region) {
            assert(tensor_buffer.find(op->outputs[j]) != tensor_buffer.end());
            tensor_buffer[pt_base] = tensor_buffer[op->outputs[j]];
            inplace = true;
          }
        }
        if (inplace) {
          continue;
        }
      }
      if (op->op_type == OP_REPLICATE) {
        assert(op->numInputs == 1 && op->numOutputs == 1);
      }
      // pt_base->print("output pt");
      // std::cout << "output mv: " << pt_base->machine_view << std::endl;

      std::vector<ParallelTensor> list;
      bool found_parallel_tensor = false;
      // Always enable memory reuse
      // if (model->cpu_offload) {
      if (true) {
        for (auto const &pre_pt : tensor_buffer) {
          bool used_by_future_operator = false;
          bool used_by_current_operator = false;
          if (pre_pt.first->get_shape() != pt_base->get_shape()) {
            // Continue if shape mismatches
            continue;
          }
          // Skip if pre_pt and pt_base are in different pipeline stages
          // we compare their pipeline stages using the machine views
          // of the first data pipeline
          if (pre_pt.second[0]->machine_view != machine_views[0]) {
            continue;
          }
          // Check that pt cannot be used as an input to the current operator
          for (int j = 0; j < op->numInputs; j++) {
            if (parallel_tensor_list_overlaps(tensor_buffer[op->inputs[j]],
                                              pre_pt.second)) {
              used_by_current_operator = true;
            }
          }
          for (int j = 0; j < i; j++) {
            assert(tensor_buffer.find(op->outputs[j]) != tensor_buffer.end());
            if (parallel_tensor_list_overlaps(tensor_buffer[op->outputs[j]],
                                              pre_pt.second)) {
              used_by_current_operator = true;
            }
          }
          // Check that pt cannot be used by any subsequent operators
          for (int op_idx2 = op_idx; op_idx2 < model->operators.size();
               op_idx2++) {
            Op const *op2 = model->operators[op_idx2];
            for (int j = 0; j < op2->numInputs; j++) {
              if (tensor_buffer.find(op2->inputs[j]) != tensor_buffer.end()) {
                if (parallel_tensor_list_overlaps(tensor_buffer[op2->inputs[j]],
                                                  pre_pt.second)) {
                  used_by_future_operator = true;
                }
              }
            }
          }
          if (!used_by_future_operator && !used_by_current_operator) {
            found_parallel_tensor = true;
            list = pre_pt.second;
          }
        }
        if (!found_parallel_tensor) {
          log_offload.debug(
              "Cannot find a previous tensor for operator(%d) output_idx(%d)",
              op_idx,
              i);
        }
      }
      if (!found_parallel_tensor) {
        for (int j = 0; j < model->config.data_parallelism_degree; j++) {
          // Copy the metadata from pt_base to pt
          ParallelTensor pt = new ParallelTensorBase(*pt_base);
          pt->region =
              runtime->create_logical_region(ctx,
                                             pt_base->region.get_index_space(),
                                             pt_base->region.get_field_space());
          pt->part = runtime->get_logical_partition(
              ctx, pt->region, pt_base->part.get_index_partition());

          pt->region_grad =
              runtime->create_logical_region(ctx,
                                             pt_base->region.get_index_space(),
                                             pt_base->region.get_field_space());
          pt->part_grad = runtime->get_logical_partition(
              ctx, pt->region_grad, pt_base->part.get_index_partition());
          pt->machine_view = machine_views[j];
          // std::cout << "output mv: " << pt->machine_view << std::endl;
          Domain part_domain =
              runtime->get_index_space_domain(ctx, pt_base->parallel_is);
          assert(pt->machine_view.get_domain() == part_domain);
          list.push_back(pt);
        }
      }
      assert(tensor_buffer.find(pt_base) == tensor_buffer.end());
      tensor_buffer[pt_base] = list;
    }
    // std::cout << std::endl;
  }

  // Check whether we need to reset input grads
  // We use a parallel tensor's region as the key
  std::set<LogicalRegion> reset_inputs;
  for (int l = model->operators.size() - 1; l >= 0; l--) {
    Op *op = model->operators[l];
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->region != LogicalRegion::NO_REGION);
      if (reset_inputs.find(op->inputs[i]->region) != reset_inputs.end()) {
        // We should not reset input grads since other operators have already
        // saved gradients into the region
        op->reset_input_grads[i] = false;
      } else if (i == 0 && (op->op_type == OP_RESIDUAL_LAYERNORM ||
                            op->op_type == OP_RESIDUAL_RMS_NORM ||
                            op->op_type == OP_ADD_BIAS_RESIDUAL_LAYERNORM)) {
        if (reset_inputs.find(op->outputs[0]->region) != reset_inputs.end()) {
          op->reset_input_grads[0] = false;
        }
        reset_inputs.insert(op->inputs[i]->region);
      } else {
        if (op->op_type != OP_LORA) {
          reset_inputs.insert(op->inputs[i]->region);
        }
      }
    }
  }

  // Perform fusion optimizations
  if (model->config.perform_fusion) {
    fprintf(stderr, "Applying fusion optimizations during compilation...\n");
    fprintf(
        stderr, "%zu operators before fusion...\n", model->operators.size());
    std::vector<Op *> new_operators;
    std::vector<Op *> old_operators = model->operators;
    while (
        model->apply_fusion(model->operators, new_operators, &tensor_buffer)) {
      for (size_t i = 0; i < new_operators.size(); i++) {
        for (int idx = 0; idx < new_operators[i]->numInputs; idx++) {
          for (size_t j = i + 1; j < new_operators.size(); j++) {
            if (new_operators[i]->inputs[idx]->owner_op == new_operators[j]) {
              assert(false);
            }
          }
        }
      }
      model->operators = new_operators;
    }
    assert(model->check_operators_integrity(old_operators, &tensor_buffer));
    fprintf(stderr, "%zu operators after fusion...\n", model->operators.size());
  }

  build_priority_table(model);

  // print optimized graph
  for (size_t i = 0; i < model->operators.size(); i++) {
    Op *op = model->operators[i];
    if (op->op_type == OP_INPUT || op->op_type == OP_WEIGHT) {
      continue;
    }
    log_inf_mgr.debug(
        "operator[%zu]: type(%s) guid(%lu)\n",
        i,
        get_operator_type_name(model->operators[i]->op_type).c_str(),
        model->operators[i]->op_guid);
    for (int j = 0; j < op->numInputs; j++) {
      assert(tensor_buffer.find(op->inputs[j]) != tensor_buffer.end());
      LogicalRegion handle = tensor_buffer[op->inputs[j]][0]->region;
      log_inf_mgr.debug("\tinputs[%d] mapped_region(%d,%d,%d)\n",
                        j,
                        handle.get_index_space().get_id(),
                        handle.get_field_space().get_id(),
                        handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = tensor_buffer[op->outputs[j]][0]->region;
      log_inf_mgr.debug("\toutputs[%d] mapped_region(%d,%d,%d)\n",
                        j,
                        handle.get_index_space().get_id(),
                        handle.get_field_space().get_id(),
                        handle.get_tree_id());
    }
    for (int j = 0; j < op->numWeights; j++) {
      LogicalRegion handle = op->weights[j]->region;
      log_inf_mgr.debug("\tweights[%d] mapped_region(%d,%d,%d)\n",
                        j,
                        handle.get_index_space().get_id(),
                        handle.get_field_space().get_id(),
                        handle.get_tree_id());
    }
  }
}

void InferenceManager::init_operators_inference(FFModel *model) {
  for (int batch_index = 0; batch_index < model->config.data_parallelism_degree;
       batch_index++) {
    for (size_t o = 0; o < model->operators.size(); o++) {
      Op *op = model->operators[o];
      if (op->op_type == OP_WEIGHT) {
        continue;
      }
      std::vector<ParallelTensor> inputs(op->numInputs);
      std::vector<ParallelTensor> outputs(op->numOutputs);
      for (int i = 0; i < op->numInputs; i++) {
        assert(op->inputs[i] != nullptr);
        assert(op->inputs[i]->parallel_is != IndexSpace::NO_SPACE);
        assert(tensor_buffer[op->inputs[i]].size() > batch_index);
        inputs[i] = tensor_buffer[op->inputs[i]][batch_index];
        assert(inputs[i]->parallel_is != IndexSpace::NO_SPACE);
      }
      assert(op->numOutputs > 0);
      for (int i = 0; i < op->numOutputs; i++) {
        assert(op->outputs[i] != nullptr);
        assert(op->outputs[i]->parallel_is != IndexSpace::NO_SPACE);
        assert(tensor_buffer[op->outputs[i]].size() > batch_index);
        outputs[i] = tensor_buffer[op->outputs[i]][batch_index];
        // if (i > 0) {
        //   assert(outputs[0]->machine_view == outputs[i]->machine_view);
        // }
        assert(outputs[i]->parallel_is != IndexSpace::NO_SPACE);
      }
      if (op->is_parallel_op()) {
        ((ParallelOp *)op)
            ->create_input_partition_inference(*model, inputs, outputs);
      }
      op->init_inference(*model, inputs, outputs);
    }
  }
}

InferenceResultFuture InferenceManager::inference(FFModel *model,
                                                  int index,
                                                  BatchConfig const &bc) {
  if (bc.get_mode() == INC_DECODING_MODE) {
    BatchConfigFuture bcf = Future::from_value<BatchConfig>(bc);
    return inference(model, index, bcf);
  } else if (bc.get_mode() == BEAM_SEARCH_MODE) {
    BatchConfig const *bc_ptr = &bc;
    BeamSearchBatchConfig const *bsbc_ptr =
        static_cast<BeamSearchBatchConfig const *>(bc_ptr);
    BeamSearchBatchConfigFuture bcf =
        Future::from_value<BeamSearchBatchConfig>(*bsbc_ptr);
    return inference(model, index, bcf);
  } else if (bc.get_mode() == TREE_VERIFY_MODE) {
    BatchConfig const *bc_ptr = &bc;
    TreeVerifyBatchConfig const *tvbc_ptr =
        static_cast<TreeVerifyBatchConfig const *>(bc_ptr);
    TreeVerifyBatchConfigFuture bcf =
        Future::from_value<TreeVerifyBatchConfig>(*tvbc_ptr);
    return inference(model, index, bcf);
  } else {
    assert(false && "Unsupported inference mode");
  }
}

InferenceResultFuture InferenceManager::inference(FFModel *model,
                                                  int index,
                                                  BatchConfigFuture const &bc) {
  // log_inf_mgr.print("mode(%d) num_active_infr_tokens(%d)
  // num_active_requests(%d)",
  //                   bc.get_mode(),
  //                   bc.num_active_tokens(),
  //                   bc.num_active_requests());
  //  assert(bc.num_active_tokens() > 0 && bc.num_active_requests() > 0);
  //  We currently assume that the index-th batch will be placed
  //  on the device_index-th device (except for the experts layers)
  int batch_index = index % model->config.data_parallelism_degree;
  FutureMap fm;
  bool found_input_operator = false;
  for (size_t o = 0; o < model->operators.size(); o++) {
    Op *op = model->operators[o];
    if (op->op_type == OP_WEIGHT) {
      continue;
    }
    if (op->op_type == OP_INPUT) {
      // FIXME: this is a hack, should be replace with an input ParallelTensor
      if (found_input_operator) {
        // there is another input for position embedding;
        // now only used in opt model, this input should be init after token
        // input.
        assert(op->numOutputs == 1);
        ParallelTensor pt = tensor_buffer[op->outputs[0]][batch_index];
        load_positions(model, bc, pt, model->position_offset);
      } else {
        found_input_operator = true;
        assert(op->numOutputs == 1);
        ParallelTensor pt = tensor_buffer[op->outputs[0]][batch_index];
        load_input_tokens_from_batch_config(model, bc, pt, model->handlers);
        load_inference_metadata_batch_config(model, bc, model->handlers);
      }
    }

    std::vector<ParallelTensor> inputs(op->numInputs);
    std::vector<ParallelTensor> outputs(op->numOutputs);
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i] != nullptr);
      assert(op->inputs[i]->parallel_is != IndexSpace::NO_SPACE);
      assert(tensor_buffer[op->inputs[i]].size() > batch_index);
      inputs[i] = tensor_buffer[op->inputs[i]][batch_index];
      assert(inputs[i]->parallel_is != IndexSpace::NO_SPACE);
    }
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i] != nullptr);
      assert(op->outputs[i]->parallel_is != IndexSpace::NO_SPACE);
      if (op->op_type == OP_INPUT &&
          tensor_buffer[op->outputs[i]].size() == 0) {
        continue;
      }
      assert(tensor_buffer[op->outputs[i]].size() > batch_index);
      outputs[i] = tensor_buffer[op->outputs[i]][batch_index];
      assert(outputs[i]->parallel_is != IndexSpace::NO_SPACE);
    }
    fm = op->inference(*model, bc, inputs, outputs);
  }
  assert(fm.get_future_map_domain().get_volume() == 1);
  InferenceResultFuture irf = fm.get_future(0);
  return irf;
};

// Per-tensor raw data collected before normalization in build_priority_table.
struct RawEntry {
  ParallelTensor pt;
  int   op_idx;
  float recompute_cost; // C(n): heuristic by op type
  float size_bytes;     // S(t): raw bytes before normalization
  int   transformer_layer_id; // L(t): raw, lower = earlier = more critical
  bool  is_fusion_boundary;   // F(t)
  int   access_freq_val;      // A(t)
  int   lifetime_depth;       // D(t): last_consumer_idx - producer_idx
};

void InferenceManager::build_priority_table(FFModel *model) {
  // ----------------------------------------------------------------
  // Preconditions:
  //   - model->operators[] is final (post-fusion, post-compile_inference)
  //   - tensor_buffer is fully populated for all surviving tensors
  //   - OP_FUSED operators exist and their outputs[i]->owner_op == fused_op
  // This function must be called AFTER apply_fusion() completes.
  // ----------------------------------------------------------------
  priority_table.clear();

  int total_ops = (int)model->operators.size();
  int max_transformer_layer = model->current_transformer_layer_id; // 0-based max

  printf("[FADP] build_priority_table: %d operators, max_transformer_layer=%d\n",
         total_ops, max_transformer_layer);

  // ================================================================
  // Pass 1: Collect A(t) and D(t) for every tensor in tensor_buffer.
  //
  // We walk model->operators[] and for each op record:
  //   - which output tensors it produces (producer index)
  //   - which input tensors it consumes (consumer index)
  //
  // Key: ParallelTensor pointer (same key used in tensor_buffer).
  // Only tensors present in tensor_buffer are tracked, these are the
  // only tensors that actually exist in GPU memory post-fusion.
  // ================================================================
  std::unordered_map<ParallelTensor, int>              producer_op_idx;
  std::unordered_map<ParallelTensor, int>              last_consumer_op_idx;
  std::unordered_map<ParallelTensor, int>              access_freq;

  for (int op_idx = 0; op_idx < total_ops; op_idx++) {
    Op *op = model->operators[op_idx];

    // Skip ops that produce no activations relevant to backward pass.
    // OP_WEIGHT: static parameters.
    // OP_INPUT:  input tokens.
    // OP_ARGMAX / OP_ARG_TOPK / OP_SAMPLING: output ops, no backward.
    if (op->op_type == OP_WEIGHT ||
        op->op_type == OP_INPUT  ||
        op->op_type == OP_ARGMAX ||
        op->op_type == OP_ARG_TOPK ||
        op->op_type == OP_SAMPLING) {
      continue;
    }

    // Record outputs as produced at this op_idx.
    // Only record tensors that are in tensor_buffer, inplace tensors
    // that share a region are aliased in tensor_buffer and should not
    // be counted twice.
    for (int i = 0; i < op->numOutputs; i++) {
      ParallelTensor pt = op->outputs[i];
      if (pt == nullptr) continue;
      if (tensor_buffer.find(pt) == tensor_buffer.end()) continue;

      // Do not overwrite if already registered (inplace alias case:
      // tensor_buffer[pt] == tensor_buffer[other_pt] for same region).
      if (producer_op_idx.find(pt) == producer_op_idx.end()) {
        producer_op_idx[pt] = op_idx;
      }
    }

    // Record each input as consumed at op_idx.
    // Update last_consumer_op_idx to always keep the furthest consumer.
    for (int i = 0; i < op->numInputs; i++) {
      ParallelTensor pt = op->inputs[i];
      if (pt == nullptr) continue;
      if (tensor_buffer.find(pt) == tensor_buffer.end()) continue;

      // access_freq counts distinct op boundaries.
      // We count the op index to avoid
      // double-counting when the same tensor appears in two input slots
      // of the same op (e.g., residual connection in attention).
      if (access_freq.find(pt) == access_freq.end()) {
        access_freq[pt] = 0;
        last_consumer_op_idx[pt] = op_idx;
      }
      // Only increment once per op, even if pt appears in multiple
      // input slots of the same op.
      if (last_consumer_op_idx[pt] < op_idx) {
        access_freq[pt]++;
        last_consumer_op_idx[pt] = op_idx;
      } else if (last_consumer_op_idx[pt] == op_idx &&
                 access_freq[pt] == 0) {
        // First time we see this tensor consumed, count it.
        access_freq[pt]++;
      }
    }
  }

  // ================================================================
  // Pass 2: Compute raw scores for each tensor that is:
  //   (a) in tensor_buffer, has a real GPU memory backing
  //   (b) has a producer in our map, was produced by a compute op
  //   (c) is consumed by at least one op, has a backward dependency
  //
  // Edge cases handled:
  //   - Tensors with no consumers: not relevant to backward pass, skip.
  //   - OP_FUSED outputs: F(t) = 1.0 (fusion boundary, high priority).
  //   - Parallel ops (allreduce, combine, etc.): L(t) inherited from
  //     input since they have LayerID::NO_ID, same as compile logic.
  //   - Inplace tensors: pt->region == input->region, they share a
  //     buffer. We only score the canonical key present in producer_op_idx.
  // ================================================================

  // We need max values for normalization.
  // Collect raw values first, normalize in Pass 3.
  std::vector<RawEntry> raw_entries;
  raw_entries.reserve(producer_op_idx.size());

  for (auto const &[pt, prod_idx] : producer_op_idx) {

    // Skip tensors with no consumers, they are never read in backward.
    if (access_freq.find(pt) == access_freq.end() ||
        access_freq[pt] == 0) {
      continue;
    }

    Op *producer = model->operators[prod_idx];

    // ---- C(n): recomputation cost heuristic by op type ----
    // We cannot use the simulator here (destroyed after graph_optimize_task).
    // These values are grounded in FlexLLM's profiling observations:
    // attention and linear dominate, norms are cheap, parallel ops near-zero.
    float cost = 1.0f; // default
    OperatorType ot = producer->op_type;
    if (ot == OP_FUSED) {
      // FusedOp wraps multiple ops, cost is sum of internals.
      // Use a conservative mid-range value since we cannot enumerate
      // all combinations without the simulator.
      cost = 5.0f;
    } else if (ot == OP_INC_MULTIHEAD_SELF_ATTENTION ||
               ot == OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION ||
               ot == OP_TREE_INC_MULTIHEAD_SELF_ATTENTION) {
      cost = 10.0f;
    } else if (ot == OP_LINEAR) {
      cost = 8.0f;
    } else if (ot == OP_LORA) {
      // LoRA is two small matmuls, cheaper than full linear.
      cost = 3.0f;
    } else if (ot == OP_RESIDUAL_LAYERNORM ||
               ot == OP_ADD_BIAS_RESIDUAL_LAYERNORM ||
               ot == OP_LAYERNORM) {
      cost = 2.0f;
    } else if (ot == OP_RESIDUAL_RMS_NORM ||
               ot == OP_RMS_NORM) {
      cost = 2.0f;
    } else if (ot == OP_SIGMOID_SILU_MULTI ||
               ot == OP_EW_ADD || ot == OP_EW_MUL) {
      cost = 1.5f;
    } else if (ot == OP_ALLREDUCE ||
               ot == OP_PARALLEL_IDENTITY ||
               ot == OP_COMBINE ||
               ot == OP_REPLICATE ||
               ot == OP_REPARTITION ||
               ot == OP_REDUCTION) {
      // Parallel communication ops: not compute-heavy to rerun,
      // but their outputs are needed, assign low cost.
      cost = 0.5f;
    }

    // ---- S(t): tensor size in bytes ----
    // Use get_piece_size() which accounts for tensor parallelism sharding:
    // each GPU only holds size/degree elements, not the full tensor.
    // This is more accurate than get_volume() * dtype_bytes for the
    // actual memory footprint on one device.
    float size_bytes = (float)pt->get_shape().get_piece_size();

    // ---- L(t): transformer layer id ----
    // Resolve NO_ID by walking up inputs, same logic as line 117-125
    // in compile_model_and_allocate_buffer.
    int transformer_layer_id = 0;
    if (producer->layer_guid != LayerID::NO_ID) {
      transformer_layer_id = producer->layer_guid.transformer_layer_id;
    } else {
      // Walk up inputs until we find an op with a valid layer_guid.
      Op const *op_with_guid = producer;
      while (op_with_guid != nullptr &&
             op_with_guid->layer_guid == LayerID::NO_ID) {
        if (op_with_guid->numInputs != 1) break; // safety: multi-input ops
        op_with_guid = op_with_guid->inputs[0]->owner_op;
      }
      if (op_with_guid != nullptr &&
          op_with_guid->layer_guid != LayerID::NO_ID) {
        transformer_layer_id =
            op_with_guid->layer_guid.transformer_layer_id;
      }
    }

    // ---- F(t): fusion boundary ----
    // A tensor is a fusion boundary if its owner_op (post-fusion) is
    // OP_FUSED. These are the ONLY tensors that survived fusion with
    // a real GPU memory backing; all internals are gone.
    bool is_fusion_boundary = (producer->op_type == OP_FUSED);

    // ---- A(t): access frequency ----
    int freq = access_freq.at(pt);

    // ---- D(t): lifetime depth ----
    int last_consumer = last_consumer_op_idx.at(pt);
    int depth = last_consumer - prod_idx;
    // depth >= 0 always: last_consumer >= prod_idx by construction.

    raw_entries.push_back({pt,
                           prod_idx,
                           cost,
                           size_bytes,
                           transformer_layer_id,
                           is_fusion_boundary,
                           freq,
                           depth});
  }

  // ================================================================
  // Pass 3: Normalize each component to [0, 1] and compute P(t).
  //
  // Normalization guard: if max == 0 (all values identical or single
  // entry), set denominator to 1 to avoid division by zero.
  //
  // L(t) is INVERTED: earlier layers -> higher priority, because
  // gradients must flow through all subsequent layers from them.
  // ================================================================
  if (raw_entries.empty()) {
    printf("[FADP] build_priority_table: no eligible tensors found.\n");
    return;
  }

  float max_cost  = 0.0f;
  float max_size  = 0.0f;
  int   max_freq  = 0;
  int   max_depth = 0;
  // max_layer is the global max from model,
  // so L(t) normalization is consistent regardless of which tensors
  // make it into the priority table.
  float max_layer = (max_transformer_layer > 0)
                    ? (float)max_transformer_layer
                    : 1.0f;

  for (auto const &e : raw_entries) {
    if (e.recompute_cost > max_cost) max_cost = e.recompute_cost;
    if (e.size_bytes     > max_size) max_size = e.size_bytes;
    if (e.access_freq_val > max_freq) max_freq = e.access_freq_val;
    if (e.lifetime_depth  > max_depth) max_depth = e.lifetime_depth;
  }
  // Division-by-zero guards
  if (max_cost  < 1e-6f) max_cost  = 1.0f;
  if (max_size  < 1e-6f) max_size  = 1.0f;
  if (max_freq  == 0)    max_freq  = 1;
  if (max_depth == 0)    max_depth = 1;

  for (auto const &e : raw_entries) {
    float cn = e.recompute_cost   / max_cost;
    float st = e.size_bytes       / max_size;
    // Invert: layer 0 (earliest) -> lt = 1.0, last layer -> lt = 0.0
    float lt = 1.0f - ((float)e.transformer_layer_id / max_layer);
    float ft = e.is_fusion_boundary ? 1.0f : 0.0f;
    float at = (float)e.access_freq_val / (float)max_freq;
    float dt = (float)e.lifetime_depth  / (float)max_depth;

    float score = PRUNE_WEIGHT_C * cn
                + PRUNE_WEIGHT_S * st
                + PRUNE_WEIGHT_L * lt
                + PRUNE_WEIGHT_F * ft
                + PRUNE_WEIGHT_A * at
                + PRUNE_WEIGHT_D * dt;

    TensorPriorityInfo info;
    info.priority_score      = score;
    info.recompute_cost_ms   = e.recompute_cost;
    info.size_mb             = e.size_bytes / (1024.0f * 1024.0f);
    info.layer_criticality   = lt;   // already normalized+inverted
    info.is_fusion_boundary  = e.is_fusion_boundary;
    info.access_freq         = e.access_freq_val;
    info.lifetime_depth      = e.lifetime_depth;
    info.op_index            = e.op_idx;

    priority_table[e.pt] = info;

    printf("[FADP][PRIORITY] op_idx=%-3d type=%-30s score=%.4f "
           "C=%.3f S=%.3f L=%.3f F=%d A=%.3f D=%.3f "
           "size_mb=%.3f layer=%d fused=%s\n",
           e.op_idx,
           get_operator_type_name(model->operators[e.op_idx]->op_type).c_str(),
           score, cn, st, lt, (int)ft, at, dt,
           info.size_mb,
           e.transformer_layer_id,
           e.is_fusion_boundary ? "YES" : "no");
  }

  printf("[FADP] build_priority_table: complete. "
         "%zu tensors scored (out of %zu in tensor_buffer).\n",
         priority_table.size(),
         tensor_buffer.size());
}

std::vector<FinetuningBwdFuture> InferenceManager::peft_bwd(
    FFModel *model, int index, BatchConfigFuture const &bc) {
  int batch_index = index % model->config.data_parallelism_degree;
  FutureMap fm;
  bool found_input_operator = false;
  int last_op = model->operators.size() - 1;
  // Assert that the last operator must be argmax or sampling
  assert(model->operators[last_op]->op_type == OP_ARGMAX ||
         model->operators[last_op]->op_type == OP_ARG_TOPK ||
         model->operators[last_op]->op_type == OP_SAMPLING);
  last_op -= 1;
  while (model->operators[last_op]->op_type == OP_WEIGHT && last_op > 0) {
    last_op -= 1;
  }
  for (int o = last_op; o >= 0; o--) {
    Op *op = model->operators[o];
    if (op->op_type == OP_WEIGHT) {
      continue;
    }
    if (op->op_type == OP_INPUT) {
      continue;
    }
    std::vector<ParallelTensor> inputs(op->numInputs);
    std::vector<ParallelTensor> outputs(op->numOutputs);
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i] != nullptr);
      assert(op->inputs[i]->parallel_is != IndexSpace::NO_SPACE);
      assert(tensor_buffer[op->inputs[i]].size() > batch_index);
      inputs[i] = tensor_buffer[op->inputs[i]][batch_index];
      assert(inputs[i]->parallel_is != IndexSpace::NO_SPACE);
    }
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i] != nullptr);
      assert(op->outputs[i]->parallel_is != IndexSpace::NO_SPACE);
      if (op->op_type == OP_INPUT &&
          tensor_buffer[op->outputs[i]].size() == 0) {
        continue;
      }
      assert(tensor_buffer[op->outputs[i]].size() > batch_index);
      outputs[i] = tensor_buffer[op->outputs[i]][batch_index];
      assert(outputs[i]->parallel_is != IndexSpace::NO_SPACE);
    }
    fm = op->peft_bwd(*model, bc, inputs, outputs);
  }
  assert(fm.get_future_map_domain().get_volume() ==
         model->config.tensor_parallelism_degree);
  std::vector<FinetuningBwdFuture> irf;
  for (int i = 0; i < model->config.tensor_parallelism_degree; i++) {
    irf.push_back(fm.get_future(i));
  }
  return irf;
};

void InferenceManager::load_input_tokens_from_batch_config(
    FFModel *model,
    BatchConfigFuture const &bc,
    ParallelTensor const input,
    FFHandler *handlers) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  size_t machine_view_hash = input->machine_view.hash();
  ArgumentMap argmap;
  Domain domain = runtime->get_index_space_domain(ctx, input->parallel_is);

  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    MachineView view = input->machine_view;                                    \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      argmap.set_point(*it,                                                    \
                       TaskArgument(&handlers[view.get_device_id(*it)],        \
                                    sizeof(FFHandler)));                       \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }

  IndexLauncher launcher(RM_LOAD_TOKENS_TASK_ID,
                         input->parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(
      input->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, input->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void InferenceManager::load_inference_metadata_batch_config(
    FFModel *model, BatchConfigFuture const &bc, FFHandler *handlers) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  ArgumentMap argmap;

  Domain domain =
      runtime->get_index_space_domain(ctx, model->config.all_gpu_task_is);
  Rect<1> task_rect = domain;

  int idx = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    FFHandler handler = handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handler, sizeof(FFHandler)));
  }

  IndexLauncher launcher(RM_LOAD_BATCH_CONFIG_TASK_ID,
                         model->config.all_gpu_task_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         FFConfig::DataParallelism_GPU);
  launcher.add_future(bc);
  runtime->execute_index_space(ctx, launcher);
}

void InferenceManager::load_positions(FFModel *model,
                                      BatchConfigFuture const &bc,
                                      ParallelTensor position_input,
                                      int offset) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  size_t machine_view_hash = position_input->machine_view.hash();
  ArgumentMap argmap;
  IndexLauncher launcher(RM_LOAD_POSITION_TASK_ID,
                         position_input->parallel_is,
                         TaskArgument(&offset, sizeof(int)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(position_input->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    position_input->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void InferenceManager::register_model_weights_loader(FFModel *model,
                                                     FileDataLoader *loader) {
  model_weights_loaders[model] = loader;
}

void FFModel::set_transformer_layer_id(int id) {
  // We assume that users call this function with
  // monotonically increasing ids
  assert(id == current_transformer_layer_id + 1 ||
         (id == 0 && current_transformer_layer_id == 0));
  current_transformer_layer_id = id;
  assert(id < MAX_NUM_TRANSFORMER_LAYERS);
}

void FFModel::set_num_kv_cache_pages(int num_kv_cache_pages_) {
  num_kv_cache_pages = num_kv_cache_pages_;
}

int FFModel::get_num_kv_cache_pages() const {
  return num_kv_cache_pages;
}

void FFModel::set_position_offset(int offset) {
  assert(offset == 0 || offset == 2);
  position_offset = offset;
}

void FFModel::compile_inference() {
  std::cout << "###PEFT DEBUGGING### Entering compile_inference." << std::endl;

  // Request at least four CPU processors for inference runs
  assert(
      config.cpusPerNode >= 4 &&
      "FlexFlow Serve requires at least four CPU cores per node, please add "
      "`-ll:cpu 4` in the command line if you are using the C++ interface or "
      "set `num_cpus` in `ff.init` if you are using the Python interface");

  std::cout << "###PEFT DEBUGGING### Configuration check passed: At least four "
               "CPU cores per node."
            << std::endl;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  config.computationMode = COMP_MODE_INFERENCE;
  create_operators_from_layers();

  // Launch the graph optimize task
  std::cout << "###PEFT DEBUGGING### Launching graph optimization task."
            << std::endl;
  {
    FFModel *model = this;
    TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID,
                          TaskArgument(&model, sizeof(FFModel *)));
    Future future = runtime->execute_task(ctx, launcher);

    PCG::GraphOptimalViewSerialized ret =
        future.get_result<PCG::GraphOptimalViewSerialized>();
    Deserializer dez(ret.data, ret.total_bytes);
    // Reconstruct operators
    PCG::Graph *best_graph = new PCG::Graph(this);
    std::unordered_map<PCG::Node, MachineView> optimal_views;
    deserialize_graph_optimal_view(dez, best_graph, optimal_views);
    operators.clear();
    convert_graph_to_operators(best_graph, optimal_views);
    // best_graph->print_dot();
    delete best_graph;
    for (auto const &layer : layers) {
      // map inputs to parallel tensor
      if (layer->op_type == OP_INPUT) {
        Tensor tensor = layer->outputs[0];
        ParallelTensor parallel_tensor = nullptr;
        for (auto const &op : operators) {
          if (op->op_type == OP_INPUT) {
            NoOp *noop = (NoOp *)op;
            if (noop->input_tensor_guid == tensor->tensor_guid) {
              parallel_tensor = op->outputs[0];
            }
          }
        }
        assert(parallel_tensor != nullptr);
        tensor->parallel_tensor = parallel_tensor;
      }
      // map weights to parallel_tensor
      for (int i = 0; i < layer->numWeights; i++) {
        assert(layer->weights[i] != nullptr);
        Tensor weight = layer->weights[i];
        ParallelTensor parallel_weight = nullptr;
        for (auto const &op : operators) {
          if (op->layer_guid == layer->layer_guid) {
            assert(op->op_type == layer->op_type);
            assert(op->numWeights == layer->numWeights);
            parallel_weight = op->weights[i];
          }
        }
        assert(parallel_weight != nullptr);
        weight->parallel_tensor = parallel_weight;
      }
    }
  }

  std::cout
      << "###PEFT DEBUGGING### Operators reconstructed from optimized graph."
      << std::endl;
  // Perform inplace optimizations
  std::cout << "###PEFT DEBUGGING### Starting inplace optimizations."
            << std::endl;

  loss_op = nullptr;
  metrics_op = nullptr;
  // Perform inplace optimizations
  if (config.enable_inplace_optimizations) {
    for (size_t l = 1; l < operators.size(); l++) {
      if (operators[l]->can_inplace_output()) {
        // Assume outputs[0] is inplace with inputs[0]
        assert(operators[l]->numOutputs == 1);
        if (operators[l]->inputs[0]->owner_op != NULL) {
          // int dim1 = operators[l]->outputs[0]->num_dims;
          // int dim2 = operators[l]->inputs[0]->num_dims;
          MachineView view1 = operators[l]->outputs[0]->machine_view;
          MachineView view2 = operators[l]->inputs[0]->machine_view;
          if (view1 == view2) {
            // Check no others also need operators[l]->inputs[0]
            bool found = false;
            for (size_t i = 0; i < operators.size(); i++) {
              if (i == l) {
                continue;
              }
              for (int j = 0; j < operators[i]->numInputs; j++) {
                if ((operators[i]->inputs[j]->owner_op ==
                     operators[l]->inputs[0]->owner_op) &&
                    (operators[i]->inputs[j]->owner_idx ==
                     operators[l]->inputs[0]->owner_idx)) {
                  found = true;
                }
              }
            }
            if (!found) {
              // Perform inplace
              operators[l]->do_inplace_output();
            }
          }
        }
      }
    }
  }

  // Output tensor mapping
  std::cout << "###PEFT DEBUGGING### Mapping output tensors." << std::endl;
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];

    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->owner_op != NULL);
    }
    for (int i = 0; i < op->numWeights; i++) {
      assert(op->weights[i]->owner_op != NULL);
      assert(op->weights[i]->region != LogicalRegion::NO_REGION);
      parameters.push_back(op->weights[i]);
    }
    op->map_output_tensors(*this);
  }

  // Check correctness
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i]->owner_op == op);
      assert(op->outputs[i]->owner_idx == i);
      assert(op->outputs[i]->parallel_tensor_guid != 0);
    }
  }

#ifdef FF_USE_NCCL
  std::cout << "###PEFT DEBUGGING### Setting up NCCL communications."
            << std::endl;
  for (size_t l = 0; l < operators.size(); l++) {
    // Only create nccl for allreduce and fusedop for inference
    // (fusedop may include allreduces)
    if (operators[l]->op_type == OP_ALLREDUCE ||
        operators[l]->op_type == OP_PARALLEL_IDENTITY ||
        operators[l]->op_type == OP_LORA || operators[l]->op_type == OP_FUSED) {
      MachineView view = operators[l]->outputs[0]->machine_view;
      // inference
      if (view_hash_to_nccl_comms.find(view.hash()) ==
          view_hash_to_nccl_comms.end()) {
        TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
        Future future = runtime->execute_task(ctx, launcher);
        ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
        IndexSpace task_is = get_or_create_task_is(view);
        ArgumentMap argmap;
        IndexLauncher index_launcher(
            NCCL_INIT_COMMS_TASK_ID,
            task_is,
            TaskArgument(&ncclId, sizeof(ncclUniqueId)),
            argmap,
            Predicate::TRUE_PRED,
            false /*must*/,
            0 /*mapper_id*/,
            view.hash() /*MappingTagID*/);
        index_launcher.concurrent = true;
        FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
        fm.wait_all_results();
        int idx = 0;
        Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
        ncclComm_t *nccl_comms =
            (ncclComm_t *)malloc(sizeof(ncclComm_t) * task_domain.get_volume());
        for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
          nccl_comms[idx] = fm.get_result<ncclComm_t>(*it);
        }
        view_hash_to_nccl_comms[view.hash()] = nccl_comms;
      }
      // peft
      if (view_hash_to_nccl_comms_peft.find(view.hash()) ==
          view_hash_to_nccl_comms_peft.end()) {
        TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
        Future future = runtime->execute_task(ctx, launcher);
        ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
        IndexSpace task_is = get_or_create_task_is(view);
        ArgumentMap argmap;
        IndexLauncher index_launcher(
            NCCL_INIT_COMMS_TASK_ID,
            task_is,
            TaskArgument(&ncclId, sizeof(ncclUniqueId)),
            argmap,
            Predicate::TRUE_PRED,
            false /*must*/,
            0 /*mapper_id*/,
            view.hash() /*MappingTagID*/);
        index_launcher.concurrent = true;
        FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
        fm.wait_all_results();
        int idx = 0;
        Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
        ncclComm_t *nccl_comms_peft =
            (ncclComm_t *)malloc(sizeof(ncclComm_t) * task_domain.get_volume());
        for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
          nccl_comms_peft[idx] = fm.get_result<ncclComm_t>(*it);
        }
        view_hash_to_nccl_comms_peft[view.hash()] = nccl_comms_peft;
      }
    }
  }
#endif
  std::cout << "###PEFT DEBUGGING### compile_inference completed successfully."
            << std::endl;
}

std::string join_path(std::vector<std::string> const &paths) {
  std::string joined;
  for (auto const &path : paths) {
    if (joined.empty()) {
      joined = path;
    } else {
      if (path[0] == '/') {
        joined = path;
      } else if (joined.back() != '/') {
        joined += '/';
        joined += path;
      } else {
        joined += path;
      }
    }
  }
  return joined;
}

}; // namespace FlexFlow
