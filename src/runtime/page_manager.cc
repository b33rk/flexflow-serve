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

#include "flexflow/page_manager.h"

namespace FlexFlow {

int ceilDiv(int a, int b) {
  assert(b!=0 && "Attempting to divide by 0");
  assert(a>=0 && b > 0 && "Expected non-negative numbers");
  return (a + b - 1) / b;
}

// For all runtime functions, they share a single page manager for pages
// information
PageManager *page_manager_singleton = nullptr;

PageManager::PageManager(int tokens_per_page_, int tot_num_pages_) : 
              tokens_per_page(tokens_per_page_), tot_num_pages(tot_num_pages_) {}

PageManager *PageManager::get_page_manager() {
  assert(page_manager_singleton != nullptr && "PageManager not initialized");
  return page_manager_singleton;
}

PageManager *PageManager::get_page_manager(FFModel *ff, size_t total_kv_cache_size) {
  int num_kv_heads = ff->num_kv_heads;
  int size_dt = ff->size_dt;
  int qkv_dim = ff->qkv_dim;
  int num_transformer_layers = ff->num_transformer_layers;
  printf("num_kv_heads: %i\n", num_kv_heads);
  printf("size_dt: %i\n", size_dt);
  printf("qkv_dim: %i\n", qkv_dim);
  printf("num_transformer_layers: %i\n", num_transformer_layers);
  printf("total_kv_cache_size: %lu\n", total_kv_cache_size);
  assert(num_kv_heads > 0 && size_dt > 0 && qkv_dim > 0 &&
         num_transformer_layers > 0); // needs to make sure that the model is initialized
  assert(page_manager_singleton == nullptr && "Attempting to initialize PageManager twice");
  size_t num_total_pages = 0;
  if (total_kv_cache_size == 0) {
    // enough pages to fit max seq length in each request
    num_total_pages = ceilDiv(BatchConfig::max_sequence_length() * BatchConfig::max_requests_per_batch(), kPagesize);
  } else {
    assert(total_kv_cache_size > size_dt * qkv_dim * num_kv_heads * num_transformer_layers);
    size_t per_token_size = 2 * size_dt * qkv_dim * num_kv_heads; // 2 factor for K and V
    size_t page_size_bytes = kPagesize * per_token_size; // Each page contains kPagesize tokens 
    num_total_pages = ceilDiv(total_kv_cache_size, page_size_bytes * num_transformer_layers);
  }
  printf("page manager singleton is initialized with %ld pages\n",
          num_total_pages);
  page_manager_singleton = new PageManager(kPagesize, num_total_pages);
  
  return page_manager_singleton;
}

bool PageManager::enough_space_to_add_request(int num_tokens) {
  // there is enough space to add a request if there are enough pages for this request's prompt + the decoding steps
  // assume all existing requests are in decoding mode, as we don't allow multiple partial prompts
  
  // ensure that no other request is an unfinished prompt
  for (int i=0; i<active_requests.size(); i++) {
    RequestGuid const &guid = active_requests[i];
    if (request_num_used_pages[guid] < req2page_indices[guid].size()) {
      // this request is an unfinished prompt
      // we cannot add a new request
      return false;
    }
  }
  // check that there is enough space to add one token to each request (since they are all in decoding mode)
  std::vector<std::pair<RequestGuid, int>> tokens_per_request;
  for (int i=0; i<active_requests.size(); i++) {
    RequestGuid const &guid = active_requests[i];
    tokens_per_request.push_back(std::make_pair(guid, 1));
  }
  if (!enough_space_to_append_tokens(tokens_per_request)) {
    return false;
  }

  int new_pages_needed = ceilDiv(num_tokens, tokens_per_page);
  return free_pages.size() >= new_pages_needed;
}

bool PageManager::enough_space_to_append_tokens(std::vector<std::pair<RequestGuid, int>> tokens_per_request) {
  int new_pages_needed = 0;
  for (auto const &pair : tokens_per_request) {
    RequestGuid const &guid = pair.first;
    int num_tokens = pair.second;
    assert(num_tokens > 0 && "Number of tokens to append must be positive");
    assert(req2page_indices[guid].size() - request_num_used_pages[guid] >= 0 && "Number of used pages must be less than or equal to the number of pages assigned to the request");
    assert(tokens_per_page - num_tokens_in_last_used_page[guid] >= 0 && "Number of tokens in last page must be less than or equal to the number of tokens per page");
    int available_slots = tokens_per_page - num_tokens_in_last_used_page[guid] + 
                          (req2page_indices[guid].size() - request_num_used_pages[guid]) * tokens_per_page;
    if (num_tokens > available_slots) {
      int num_pages_needed = ceilDiv(num_tokens-available_slots, tokens_per_page);
      new_pages_needed += num_pages_needed;
    }
  }
  return free_pages.size() >= new_pages_needed;
}

void PageManager::add_request(RequestGuid const &guid, int num_tokens){
  assert(num_tokens > 0 && "Number of tokens to add must be positive");
  assert(req2page_indices.find(guid) == req2page_indices.end() && "Request already exists");
  assert(enough_space_to_add_request(num_tokens) && "Not enough space to add request");
  // add the request to the active requests
  active_requests.push_back(guid);
  // assign pages to the request
  int num_pages_needed = ceilDiv(num_tokens, tokens_per_page);
  assert(num_pages_needed > 0);
  assert(free_pages.size() >= num_pages_needed && "Not enough free pages");
  std::vector<PageId> pages;
  for (int i=0; i<num_pages_needed; i++) {
    PageId page_id = free_pages.front();
    free_pages.pop_front();
    pages.push_back(page_id);
  }
  req2page_indices[guid] = pages;
  request_num_used_pages[guid] = 0;
  num_tokens_in_last_used_page[guid] = 0;
}

// remove completed request
void PageManager::remove_request(RequestGuid const &request_guid){
  assert(req2page_indices.find(request_guid) != req2page_indices.end() && "Request does not exist");
  // remove the request from the active requests
  auto it = std::find(active_requests.begin(), active_requests.end(), request_guid);
  if (it != active_requests.end()) {
    active_requests.erase(it);
  }
  // free the pages assigned to the request
  std::vector<PageId> pages = req2page_indices[request_guid];
  for (int i=0; i<pages.size(); i++) {
    free_pages.push_back(pages[i]);
  }
  req2page_indices.erase(request_guid);
  request_num_used_pages.erase(request_guid);
  num_tokens_in_last_used_page.erase(request_guid);
  
}

RequestGuid PageManager::evict_request_fifo();
void PageManager::append_tokens(RequestGuid const &guid, int num_tokens);


}; // namespace FlexFlow
