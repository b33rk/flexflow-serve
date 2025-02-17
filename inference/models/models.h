#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include "models/starcoder.h"
#include <wordexp.h>
#include <nlohmann/json.hpp>

FFModel build_model(std::string model_name,
    std::string cache_folder_path,
    bool use_full_precision,
    GenerationConfig &generationConfig,
    InferenceMode inf_mode = INC_DECODING_MODE);