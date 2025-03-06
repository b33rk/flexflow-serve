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

// For all runtime functions, they share a single page manager for pages
// information
PageManager *page_manager_singleton = nullptr;

PageManager::PageManager(int tot_num_pages_) : tot_num_pages(tot_num_pages_) {
  assert(kPagesize > 0 && tot_num_pages >= 0 &&
         "Number of tokens per page must be positive and total number of pages "
         "must be non-negative");
  for (int i = 0; i < tot_num_pages; i++) {
    free_pages.insert(i);
  }
}

PageManager *PageManager::get_page_manager() {
  assert(page_manager_singleton != nullptr && "PageManager not initialized");
  return page_manager_singleton;
}

PageManager *PageManager::get_page_manager(int num_total_pages) {
  assert(num_total_pages > 0 &&
         "attempting to initialize the PageManager with 0 pages");
  assert(page_manager_singleton == nullptr &&
         "Attempting to initialize PageManager twice");
  printf("page manager singleton is initialized with %d pages\n",
         num_total_pages);
  page_manager_singleton = new PageManager(num_total_pages);
  return page_manager_singleton;
}

int PageManager::get_tot_num_pages() const {
  return tot_num_pages;
}

int PageManager::get_tokens_per_page() const {
  return kPagesize;
}

int PageManager::get_num_pages_used_by_req(
    RequestGuid const &request_guid) const {
  assert(requests_info.find(request_guid) != requests_info.end());
  int n = requests_info.at(request_guid).num_used_pages;
  if (!(n >= 0 && n <= requests_info.at(request_guid).page_indices.size())) {
    std::cerr << "Error: requests_info.at(request_guid).num_used_pages is out "
                 "of bounds for request "
              << request_guid << std::endl;
    std::cerr << *this << std::endl;
  }
  assert(n >= 0 && n <= requests_info.at(request_guid).page_indices.size());
  return n;
}

std::vector<int>
    PageManager::get_req_page_indices(RequestGuid const &request_guid) const {
  int n = get_num_pages_used_by_req(request_guid);
  return std::vector<int>(requests_info.at(request_guid).page_indices.begin(),
                          requests_info.at(request_guid).page_indices.begin() +
                              n);
}

int PageManager::get_num_tokens_in_last_used_page(
    RequestGuid const &request_guid) const {
  assert(requests_info.find(request_guid) != requests_info.end());
  int n = requests_info.at(request_guid).num_tokens_in_last_used_page;
  if (!(n >= 0 && n <= kPagesize)) {
    std::cerr
        << "Error: num_tokens_in_last_used_page is out of bounds for request "
        << request_guid << std::endl;
    std::cerr << *this << std::endl;
  }
  assert(n >= 0 && n <= kPagesize);

  return n;
}

bool PageManager::enough_space_to_add_request(
    int num_prompt_tokens,
    int num_prompt_tokens_in_first_batch,
    int max_tokens_per_batch) const {
  // there is enough space to add a request if there are enough pages for this
  // request's prompt + N decoding steps for all existing requests, where
  // N = tot prefilling steps needed to consume the new request's prompt
  assert(num_prompt_tokens > 0 && num_prompt_tokens_in_first_batch > 0);
  assert(num_prompt_tokens_in_first_batch <= num_prompt_tokens);

  // pages needed to process the new request's prompt alone
  int new_pages_needed = round_up_pages(num_prompt_tokens);

  // number of steps to finish prefilling (during which other requests will
  // accrue more tokens)
  int num_expected_prefill_steps =
      ceilDiv(num_prompt_tokens - num_prompt_tokens_in_first_batch,
              max_tokens_per_batch - (int)active_requests.size());

  for (auto req_info_pair : requests_info) {
    RequestGuid const &guid = req_info_pair.first;
    PerRequestPageInfo const &req_info = req_info_pair.second;
    // ensure that no other request is an unfinished prompt
    if (req_info.num_used_pages < req_info.page_indices.size()) {
      // this request is an unfinished prompt
      // we cannot add a new request
      std::cout << *this << std::endl;
      assert(false && "Attempting to add a request with another unfinished "
                      "prefill present in the batch");
    }
    int available_slots =
        kPagesize - req_info.num_tokens_in_last_used_page +
        ((int)req_info.page_indices.size() - req_info.num_used_pages) *
            kPagesize;
    if (num_expected_prefill_steps > available_slots) {
      new_pages_needed +=
          round_up_pages(num_expected_prefill_steps - available_slots);
    }
  }
  // printf("new pages needed to add request with %d prompt tokens, %d "
  //        "tokens in first batch, %d max tokens per batch: %d\n",
  //        num_prompt_tokens,
  //        num_prompt_tokens_in_first_batch,
  //        max_tokens_per_batch,
  //        new_pages_needed);
  // printf("free pages: %ld\n", free_pages.size());
  // printf("total pages: %d\n", tot_num_pages);
  // printf("active requests: %ld\n", active_requests.size());
  return free_pages.size() >= new_pages_needed;
}

bool PageManager::enough_space_to_append_tokens(
    std::vector<std::pair<RequestGuid, int>> new_tokens_per_request) const {

  int new_pages_needed = 0;
  for (auto const &pair : new_tokens_per_request) {
    RequestGuid const &guid = pair.first;
    int num_tokens = pair.second;
    assert(num_tokens > 0 && "Number of tokens to append must be positive");
    assert(requests_info.find(guid) != requests_info.end() &&
           "Request does not exist");
    PerRequestPageInfo const &req_info = requests_info.at(guid);
    assert((int)req_info.page_indices.size() - req_info.num_used_pages >= 0 &&
           "Number of used pages must be less than or equal to the number of "
           "pages assigned to the request");
    assert(kPagesize - req_info.num_tokens_in_last_used_page >= 0 &&
           "Number of tokens in last page must be less than or equal to the "
           "number of tokens per page");
    int available_slots =
        kPagesize - req_info.num_tokens_in_last_used_page +
        ((int)req_info.page_indices.size() - req_info.num_used_pages) *
            kPagesize;
    if (num_tokens > available_slots) {
      int num_pages_needed = round_up_pages(num_tokens - available_slots);
      new_pages_needed += num_pages_needed;
    }
  }
  return free_pages.size() >= new_pages_needed;
}

void PageManager::add_request(RequestGuid const &guid, int num_tokens) {
  assert(num_tokens > 0 && "Number of tokens to add must be positive");
  assert(requests_info.find(guid) == requests_info.end() &&
         "Request already exists");
  // assert(enough_space_to_add_request(num_tokens) &&
  //        "Not enough space to add request");
  active_requests.push_back(guid);
  // assign pages to the request
  assert(!free_pages.empty() && "No free pages available");
  int num_pages_needed = round_up_pages(num_tokens);
  std::vector<int> pages;
  for (int i = 0; i < num_pages_needed; i++) {
    int page = *free_pages.begin();
    free_pages.erase(free_pages.find(page));
    pages.push_back(page);
  }
  // add the request to the requests info
  PerRequestPageInfo req_info;
  req_info.guid = guid;
  req_info.page_indices = pages;
  req_info.num_used_pages = 0;
  req_info.num_tokens_in_last_used_page = 0;
  requests_info[guid] = req_info;
  // printf("adding request %d with %d tokens. It allocated %ld new pages\n",
  //        guid,
  //        num_tokens,
  //        pages.size());
}

// remove completed request
void PageManager::remove_request(RequestGuid const &request_guid) {
  assert(requests_info.find(request_guid) != requests_info.end() &&
         "Request does not exist");
  PerRequestPageInfo const &req_info = requests_info[request_guid];
  // free the pages assigned to the request
  for (auto page : req_info.page_indices) {
    free_pages.insert(page);
  }
  requests_info.erase(request_guid);
  // remove the request from the active requests
  auto it =
      std::find(active_requests.begin(), active_requests.end(), request_guid);
  assert(it != active_requests.end() && "Request does not exist");
  active_requests.erase(it);

  assert(requests_info.find(request_guid) == requests_info.end() &&
         "Removal of request info did not go through");
  assert(std::find(active_requests.begin(),
                   active_requests.end(),
                   request_guid) == active_requests.end() &&
         "Removal of active request did not go through");
}

RequestGuid PageManager::evict_request_fifo() {
  assert(!active_requests.empty() && "No active requests to evict");
  RequestGuid request_guid = active_requests.back();
  remove_request(request_guid);
  return request_guid;
}

void PageManager::append_tokens(RequestGuid const &request_guid,
                                int num_tokens) {
  assert(num_tokens > 0 && "Number of tokens to append must be positive");
  assert(requests_info.find(request_guid) != requests_info.end() &&
         "Request does not exist");
  PerRequestPageInfo &req_info = requests_info[request_guid];

  // std::vector<std::pair<RequestGuid, int>> new_tokens_per_request;
  // for (auto const &pair : requests_info) {
  //   RequestGuid const &guid = pair.first;
  //   if (guid == request_guid) {
  //     new_tokens_per_request.push_back(std::make_pair(guid, num_tokens));
  //   } else {
  //     new_tokens_per_request.push_back(std::make_pair(guid, 1));
  //   }
  // }
  // assert(enough_space_to_append_tokens(new_tokens_per_request) &&
  //        "Not enough space to append tokens");

  int available_slots =
      kPagesize - req_info.num_tokens_in_last_used_page +
      ((int)req_info.page_indices.size() - req_info.num_used_pages) * kPagesize;
  if (num_tokens > available_slots) {
    int num_pages_needed = round_up_pages(num_tokens - available_slots);
    assert(num_pages_needed <= free_pages.size() &&
           "Not enough free pages to append new tokens");
    for (int i = 0; i < num_pages_needed; i++) {
      int page = *free_pages.begin();
      free_pages.erase(free_pages.find(page));
      req_info.page_indices.push_back(page);
    }
  }
  // update the number of used pages and the number of tokens in the last used
  // page
  if (req_info.num_tokens_in_last_used_page == 0 &&
      req_info.num_used_pages == 0) {
    req_info.num_used_pages = 1;
  }

  req_info.num_tokens_in_last_used_page += num_tokens;
  while (req_info.num_tokens_in_last_used_page > kPagesize) {
    req_info.num_used_pages += 1;
    req_info.num_tokens_in_last_used_page -= kPagesize;
  }

  // printf("appending %d tokens to request %d. It now has %d tokens in the last
  // "
  //        "used page and %d used pages\n",
  //        num_tokens,
  //        request_guid,
  //        req_info.num_tokens_in_last_used_page,
  //        req_info.num_used_pages);
}

std::ostream &operator<<(std::ostream &os, PageManager const &pm) {
  os << "PageManager State: {\n";
  os << "\tTotal number of pages: " << pm.tot_num_pages << "\n";
  os << "\tTokens per page: " << kPagesize << "\n";
  os << "\tActive requests: " << pm.active_requests.size() << "\n";
  os << "\tFree pages: " << pm.free_pages.size() << "\n";
  os << "\tRequests info:\n";
  for (auto const &[guid, info] : pm.requests_info) {
    os << "\t  RequestGuid: " << guid << "\n";
    os << "\t    Number of used pages: " << info.num_used_pages << "\n";
    os << "\t    Number of tokens in last used page: "
       << info.num_tokens_in_last_used_page << "\n";
    os << "\t    Page indices: ";
    for (int index : info.page_indices) {
      os << index << " ";
    }
    os << "\n}\n";
  }
  return os;
}

int compute_num_kv_cache_pages_needed(int max_seq_len,
                                      int batch_size,
                                      bool is_spec) {

  int num_pages_needed = round_up_pages(max_seq_len * batch_size);
  if (!is_spec) {
    PageManager *pm = PageManager::get_page_manager(num_pages_needed);
    assert(pm->get_tot_num_pages() == num_pages_needed);
  }
  return num_pages_needed;
}

}; // namespace FlexFlow
