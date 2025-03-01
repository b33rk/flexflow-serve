#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/file_loader.h"
#include <deque>
#include <future>
#include <mutex>
#include <tokenizers_cpp.h>

namespace FlexFlow {

using RequestGuid = BatchConfig::RequestGuid;
using TokenId = BatchConfig::TokenId;
using PageId = size_t;

/*
 * @class PageManager
 * @brief A wrapper class that manages the kv cache allocation status
 * notice that all the layers of model will share the same page manager because
 * the position of kv cache will be the same
 */
class PageManager {
public:
  // Get the singleton instance of the PageManager as it will be shared in
  // multiple places
  static PageManager *get_page_manager();
  static PageManager *get_page_manager(FFModel *ff, size_t total_kv_cache_size);
  
  PageManager(int block_size, int tot_num_pages);
  
  // check if there is enough space for request with given total number of prompt/evicted tokens
  // even if the tokens will be run in multiple steps (chunked prefills)
  bool enough_space_to_add_request(int num_tokens);
  // check if there is enough space to append new tokens to the existing requests
  bool enough_space_to_append_tokens(std::vector<std::pair<RequestGuid, int>> tokens_per_request);
  void add_request(RequestGuid const &guid, int num_tokens);
  void remove_request(RequestGuid const &request_guid);
  RequestGuid evict_request_fifo();
  // add tokens to an existing request
  void append_tokens(RequestGuid const &guid, int num_tokens);
  
private:
  // pages (ordered logically by token depth) assigned to each request
  std::unordered_map<RequestGuid, std::vector<PageId>> req2page_indices;
  // number of pages (from those assigned to the request) that are already filled with tokens. 
  // Of these, only the last one is allowed to be partially filled. The others should be full.
  std::unordered_map<RequestGuid, int> request_num_used_pages;
  // requests ordered by arrival. We use this order for FIFO eviction
  std::deque<RequestGuid> active_requests;

  // slots in use in each last page of each request (all previous pages must be full)
  std::unordered_map<RequestGuid, int> num_tokens_in_last_used_page;
  // queue of available pages
  std::set<PageId> free_pages;
  
  int tot_num_pages;
  int tokens_per_page; // max tokens per page
  
};

}; // namespace FlexFlow
