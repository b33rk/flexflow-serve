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

#include "models/models.h"

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("incr_dec");

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string output_file_path;
  std::string profiling_folder_path;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &do_sample,
                      float &temperature,
                      float &topp,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      bool &run_warmup) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // cache folder
    if (!strcmp(argv[i], "-cache-folder")) {
      paths.cache_folder_path = std::string(argv[++i]);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-output-file")) {
      paths.output_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-profiling-folder")) {
      paths.profiling_folder_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--use-full-precision")) {
      use_full_precision = true;
      continue;
    }
    // verbose logging to stdout
    if (!strcmp(argv[i], "--verbose")) {
      verbose = true;
      continue;
    }
    if (!strcmp(argv[i], "--warmup")) {
      run_warmup = true;
      continue;
    }
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--temperature")) {
      temperature = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--topp")) {
      topp = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-requests-per-batch")) {
      max_requests_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tokens-per-batch")) {
      max_tokens_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-sequence-length")) {
      max_sequence_length = std::stoi(argv[++i]);
      continue;
    }
  }
  if (paths.cache_folder_path.empty()) {
    char const *ff_cache_path = std::getenv("FF_CACHE_PATH");
    paths.cache_folder_path = ff_cache_path ? std::string(ff_cache_path)
                                            : std::string("~/.cache/flexflow");
  }
  // Expand ~ to the home directory if needed
  wordexp_t p;
  wordexp(paths.cache_folder_path.c_str(), &p, 0);
  paths.cache_folder_path = p.we_wordv[0];
  wordfree(&p);
}

std::vector<Request> make_warmup_requests(int num_requests) {
  std::vector<Request> warmup_requests;
  for (int i = 0; i < num_requests; i++) {
    Request inference_req;
    inference_req.benchmarking_tokens = 512;
    inference_req.max_new_tokens = 30;
    inference_req.warmup = true;
    warmup_requests.push_back(inference_req);
  }
  return warmup_requests;
}

std::vector<Request> load_prompt_list(nlohmann::ordered_json prompt_json) {
  int total_num_requests = 0;
  std::vector<Request> requests;
  for (auto &prompt : prompt_json) {
    std::string text = prompt.get<std::string>();
    printf("Prompt[%d]: %s\n", total_num_requests, text.c_str());
    Request inference_req;
    inference_req.prompt = text;
    inference_req.max_length = 128;
    requests.push_back(inference_req);
    total_num_requests++;
  }
  return requests;
}

std::vector<Request> load_trace(nlohmann::ordered_json prompt_json, bool benchmarking=false) {
  std::vector<Request> requests;
  auto &metadata = prompt_json["metadata"];
  for (auto &entry : prompt_json["entries"]) {
    int prompt_length = entry["prompt_length"];
    int response_length = entry["response_length"];
    std::string text = entry["prompt"];

    Request inference_req;
    if (benchmarking) {
      inference_req.benchmarking_tokens = prompt_length;
      // inference_req.add_special_tokens = false;
    } else {
      inference_req.prompt = text;
    }
    inference_req.max_new_tokens = response_length;
    requests.push_back(inference_req);
  }
  return requests;
}

std::vector<Request> load_requests(std::string prompt_file_path) {
  std::ifstream file_handle(prompt_file_path);
  assert (!file_handle.good() && "Error opening prompt file!");
  nlohmann::ordered_json prompt_json;
  try {
    prompt_json = nlohmann::ordered_json::parse(file_handle,
                                              /*parser_callback_t */ nullptr,
                                              /*allow_exceptions */ true,
                                              /*ignore_comments */ true);
  } catch (const json::parse_error& e) {
    std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
    assert(false);
  }
  file_handle.close();
  if (prompt_json.empty()) {
    std::cerr << "Error: JSON file is empty!" << std::endl;
    assert(false);
  } else if (prompt_json.is_null()) {
    std::cerr << "Error: JSON file is null!" << std::endl;
    assert(false);
  } else if (prompt_json.is_array()) {
    return load_prompt_list(prompt_file_path);
  } else if (prompt_json.is_object()) {
    return load_trace(prompt_file_path);
  } else {
    std::cerr << "JSON is neither an array nor an object!" << std::endl;
    assert(false);
  }
  return {};
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FilePaths file_paths;
  std::string llm_model_name;
  GenerationConfig generationConfig;
  bool use_full_precision = false;
  bool verbose = false;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 256;
  int max_sequence_length = 2048;
  bool run_warmup = false;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   use_full_precision,
                   verbose,
                   generationConfig.do_sample,
                   generationConfig.temperature,
                   generationConfig.topp,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   run_warmup);

  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_verbose(verbose);
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->register_output_filepath(file_paths.output_file_path);
  
  FFModel model = build_model(llm_model_name,
                              file_paths.cache_folder_path,
                              use_full_precision,
                              generationConfig);
  assert(!model.config.enable_peft);

  rm->start_background_server(&model);

  if (run_warmup) {
    std::cout << "----------warmup started--------------" << std::endl;
    std::vector<Request> warmup_requests = make_warmup_requests(10);
    std::vector<GenerationResult> warmup_result =
        model.generate(warmup_requests);
    std::cout << "----------warmup finished--------------" << std::endl;
  }
  std::cout << "----------inference started--------------" << std::endl;
  std::vector<Request> requests = load_requests(file_paths.prompt_file_path);
  std::vector<GenerationResult> result = model.generate(requests);
  std::cout << "----------inference finished--------------" << std::endl;

  rm->terminate_background_server();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  if (!file_paths.profiling_folder_path.empty()) {
    std::cout << "Saving profiling info..." << std::endl;
    std::string dataset_name;
    // set dataset name to "wildchat" if the prompt file path contains "wildchat" 
    if (file_paths.prompt_file_path.find("wildchat") != std::string::npos) {
      dataset_name = "wildchat";
    } else if (file_paths.prompt_file_path.find("sharegpt") != std::string::npos) {
      dataset_name = "sharegpt";
    } else {
      dataset_name = "unknown";
    }
    rm->save_profiling_info_to_csv(file_paths.profiling_folder_path,
                                   dataset_name,
                                   llm_model_name,
                                   model.config.tensor_parallelism_degree,
                                   max_requests_per_batch,
                                   max_tokens_per_batch,
                                   0.0, // arrival rate
                                   10); // num_warmup_requests
  }
  
}

void FlexFlow::register_custom_tasks() {}
