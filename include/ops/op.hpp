/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 14:43:15
 */
#pragma once
namespace YAInfer {
enum class OpType {
  kOperatorUnknown = -1,
  kOperatorRelu = 0,
  kOperatorSigmoid = 1,
};

class Operator {
 public:
  OpType op_type_ = OpType::kOperatorUnknown;

  virtual ~Operator() = default;

  explicit Operator(OpType op_type);
};
}  // namespace YAInfer