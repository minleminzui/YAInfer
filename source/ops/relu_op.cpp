/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 17:56:17
 */
#include "ops/relu_op.hpp"

namespace YAInfer {
ReluOperator::ReluOperator(float thresh)
    : thresh_(thresh), Operator(OpType::kOperatorRelu) {}

void ReluOperator::set_thresh(float thresh) { this->thresh_ = thresh; }

float ReluOperator::get_thresh() const { return thresh_; }
}  // namespace YAInfer