/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 17:17:36
 */
#pragma once
#include "op.hpp"

namespace YAInfer {
class ReluOperator : public Operator {
 public:
  ~ReluOperator() override = default;

  explicit ReluOperator(float thresh);

  void set_thresh(float thresh);

  float get_thresh() const;

 private:
  float thresh_ = 0.f;
  // Operator plays the role of attribute storage and variable
  // Operator class need to transfer these memeber variables to Layer class
  // Opeartor class is responsible for storage, while Layer class is
  // resiponsible for computing
};
}  // namespace YAInfer
