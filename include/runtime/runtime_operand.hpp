#pragma once
#include <memory>
#include <string>
#include <vector>

#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace YAInfer {
// operands of input and output computation nodes
struct RuntimeOperand {
  std::string name;             // name of the operand node
  std::vector<int32_t> shapes;  // shapes of the operand node
  std::vector<std::shared_ptr<Tensor<float>>> datas;
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;  // usually float32
};
}  // namespace YAInfer