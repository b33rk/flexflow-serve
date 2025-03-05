#pragma once

#include "flexflow/attention_config.h"
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
  static PageManager *get_page_manager(int num_total_pages);
  PageManager(int tot_num_pages_);

  int get_tot_num_pages() const;
  int get_tokens_per_page() const;

  // returns the number of pages used by the request (excluding those allocated
  // but not used yet)
  int get_num_pages_used_by_req(RequestGuid const &request_guid) const;
  // returns the indices of the pages in use by the request (excluding those
  // allocated but not used yet)
  std::vector<int> get_req_page_indices(RequestGuid const &request_guid) const;
  int get_num_tokens_in_last_used_page(RequestGuid const &request_guid) const;

  // check if there is enough space for request with given total number of
  // prompt/evicted tokens even if the tokens will be run in multiple steps
  // (chunked prefills)
  bool enough_space_to_add_request(int num_prompt_tokens,
                                   int num_prompt_tokens_in_first_batch,
                                   int max_tokens_per_batch) const;
  // check if there is enough space to append new tokens to the existing
  // requests
  bool enough_space_to_append_tokens(
      std::vector<std::pair<RequestGuid, int>> tokens_per_request) const;
  void add_request(RequestGuid const &guid, int num_tokens);
  void remove_request(RequestGuid const &request_guid);
  RequestGuid evict_request_fifo();
  // add tokens to an existing request
  void append_tokens(RequestGuid const &guid, int num_tokens);

  struct PerRequestPageInfo {
    RequestGuid guid;
    // pages (ordered logically by token depth) assigned to each request
    std::vector<int> page_indices;
    // number of pages (from those assigned to the request) that are already
    // filled with tokens. Of these, only the last one is allowed to be
    // partially filled. The others should be full.
    int num_used_pages;
    // slots in use in each last page of each request (all previous pages must
    // be full)
    int num_tokens_in_last_used_page;
  };

  friend std::ostream &operator<<(std::ostream &os, PageManager const &pm);

private:
  // requests ordered by arrival. We use this order for FIFO eviction
  std::deque<RequestGuid> active_requests;
  // request info keyed by guid
  std::unordered_map<RequestGuid, PerRequestPageInfo> requests_info;
  // pool of available pages
  std::set<int> free_pages;

  int tot_num_pages;
};

// returns number of kv cache pages needed to guarantee no evictions if all
// requests are up to max_seq_len if is_spec==true, it also initializes the page
// manager
int compute_num_kv_cache_pages_needed(int max_seq_len,
                                      int batch_size,
                                      bool is_spec);

}; // namespace FlexFlow
