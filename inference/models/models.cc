#include "models.h"

using namespace FlexFlow;
using namespace Legion;

FFModel build_model(std::string model_name,
                    std::string cache_folder_path,
                    bool use_full_precision,
                    GenerationConfig &generationConfig,
                    InferenceMode inf_mode) {
    FFConfig ffconfig;
    if (ffconfig.cpu_offload == false && ffconfig.quantization_type != DT_NONE) {
        assert(false && "Doesn't support quantization in non-offload mode");
    }
    if (inf_mode == INC_DECODING_MODE || inf_mode == TREE_VERIFY_MODE) {
      assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree * ffconfig.pipeline_parallelism_degree == ffconfig.numNodes * ffconfig.workersPerNode);
    } else {
      // ssm is not parallelized
      ffconfig.data_parallelism_degree = ffconfig.tensor_parallelism_degree = ffconfig.pipeline_parallelism_degree = 1;
    }

    std::string config_filepath = join_path(
        {cache_folder_path, "configs", model_name, "config.json"});
    std::string tokenizer_filepath =
        join_path({cache_folder_path, "tokenizers", model_name});
    std::string weights_filepath =
        join_path({cache_folder_path,
                   "weights",
                   model_name,
                   use_full_precision ? "full-precision" : "half-precision"});
    std::ifstream config_file_handle(config_filepath);
    if (!config_file_handle.good()) {
      std::cout << "Model config file " << config_filepath << " not found."
                << std::endl;
      assert(false);
    }
    json model_config = json::parse(config_file_handle,
                                    /*parser_callback_t */ nullptr,
                                    /*allow_exceptions */ true,
                                    /*ignore_comments */ true);
    ModelType model_type = ModelType::UNKNOWN;
    auto architectures = model_config["architectures"];
    for (auto const &str : architectures) {
      if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM") {
        model_type = ModelType::LLAMA;
        break;
      } else if (str == "OPTForCausalLM") {
        model_type = ModelType::OPT;
        break;
      } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
        model_type = ModelType::FALCON;
        break;
      } else if (str == "GPTBigCodeForCausalLM") {
        model_type = ModelType::STARCODER;
        break;
      } else if (str == "MPTForCausalLM") {
        model_type = ModelType::MPT;
        break;
      }
    }
    int bos_token_id = model_config.find("bos_token_id") == model_config.end()
                           ? -1
                           : (int)model_config.at("bos_token_id");
    // parse eos token id, which can be either a single integer or an array of
    // integers. Convert to std::vector<int>
    std::vector<int> eos_token_ids;
    if (model_config.find("eos_token_id") != model_config.end()) {
      if (model_config["eos_token_id"].is_array()) {
        for (auto &eos_token_id : model_config["eos_token_id"]) {
          eos_token_ids.push_back(eos_token_id);
        }
      } else {
        eos_token_ids.push_back(model_config["eos_token_id"]);
      }
    } else {
      eos_token_ids.push_back(-1);
    }
  
    assert(model_type != ModelType::UNKNOWN &&
           "Invalid LLM model type passed (or no type was passed).");
  

    if (inf_mode == INC_DECODING_MODE || inf_mode == TREE_VERIFY_MODE) {
      RequestManager *rm = RequestManager::get_request_manager();
      rm->register_tokenizer(model_type, bos_token_id, eos_token_ids, tokenizer_filepath);
    }
  
    FFModel model(ffconfig, ffconfig.cpu_offload);
    if (model_type == ModelType::LLAMA) {
      LLAMA::create_llama_model(model,
                                config_filepath,
                                weights_filepath,
                                inf_mode,
                                generationConfig,
                                use_full_precision);
    } else if (model_type == ModelType::OPT) {
      OPT::create_opt_model(model,
                            config_filepath,
                            weights_filepath,
                            inf_mode,
                            use_full_precision);
    } else if (model_type == ModelType::FALCON) {
      FALCON::create_falcon_model(model,
                                  config_filepath,
                                  weights_filepath,
                                  inf_mode,
                                  use_full_precision);
    } else if (model_type == ModelType::STARCODER) {
      STARCODER::create_starcoder_model(model,
                                        config_filepath,
                                        weights_filepath,
                                        inf_mode,
                                        generationConfig,
                                        use_full_precision);
    } else if (model_type == ModelType::MPT) {
      MPT::create_mpt_model(model,
                            config_filepath,
                            weights_filepath,
                            inf_mode,
                            generationConfig,
                            use_full_precision);
    } else {
      assert(false && "unknow model type");
    }

    return model;

}