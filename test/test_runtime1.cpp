/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-27 18:33:52
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "runtime/runtime_ir.hpp"

TEST(TestRuntime, Runtime) {
  const std::string &param_path = "./tmp/test.pnnx.param";
  const std::string &bin_path = "./tmp/test.pnnx.bin";
  YAInfer::RuntimeGraph graph(param_path, bin_path);

  graph.Init();
  const auto operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
  }
}