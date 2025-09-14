#ifndef _FLEXFLOW_DECODING_PARAMS_H
#define _FLEXFLOW_DECODING_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct DecodingParams {
  LayerID layer_guid;
  bool beam_search;
  bool is_valid(ParallelTensorShape const &) const;
  char name[MAX_OPNAME];
};
bool operator==(DecodingParams const &, DecodingParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::DecodingParams> {
  size_t operator()(FlexFlow::DecodingParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_DECODING_PARAMS_H
