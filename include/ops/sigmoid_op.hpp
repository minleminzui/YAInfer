/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 22:42:54
 */
#pragma once
#include "layer/layer.hpp"
#include "op.hpp"

namespace YAInfer {
class SigmoidOperator : public Operator {
 public:
  explicit SigmoidOperator();
};
}  // namespace YAInfer