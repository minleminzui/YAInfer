/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-26 16:57:47
 */
#pragma once
#include <glog/logging.h>

#include <vector>

#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace YAInfer {
// information of computation graph nodes
struct RuntimeAttribute {
  std::vector<char> weight_data;  // weights parameters of nodes
  std::vector<int> shape;         // shapes of nodes
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;  // datatypes of nodes

  /**
   * @description: load weight parameters from nodes
   * @tparam T weight type
   * @return weight paramters array
   */
  template <class T>
  std::vector<T> get();
};
template <typename T>
std ::vector<T> RuntimeAttribute::get() {
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::kTypeUnknown);
  std::vector<T> weights;
  switch (type) {
    case RuntimeDataType::kTypeFloat32: {
      // when the type of loaded data is float
      const bool is_float = std::is_same<T, float>::value;
      CHECK_EQ(is_float, true);
      const uint32_t float_size = sizeof(float);
      CHECK_EQ(weight_data.size() % float_size, 0);
      for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
        float weight = *((float *)weight_data.data() + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unknown weight data type";
    }
  }

  return weights;
}
}  // namespace YAInfer