/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 17:52:09
 */
#include "ops/op.hpp"

namespace YAInfer {
Operator::Operator(OpType op_type) : op_type_(op_type) {}
}  // namespace YAInfer