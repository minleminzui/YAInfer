/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-25 13:05:16
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "factory/layer_factory.hpp"
#include "layer/relu_layer.hpp"
#include "ops/relu_op.hpp"

TEST(TestLayer, ForwardRelu1) {
  float thresh = 0.f;
  std::shared_ptr<YAInfer::Operator> relu_op =
      std::make_shared<YAInfer::ReluOperator>(thresh);

  std::shared_ptr<YAInfer::Tensor<float>> input =
      std::make_shared<YAInfer::Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;

  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> inputs;
  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> outputs;

  inputs.push_back(input);
  YAInfer::ReluLayer layer(relu_op);
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}

TEST(TestLayer, ForwardRelu2) {
  float thresh = 0.f;
  std::shared_ptr<YAInfer::Operator> relu_op =
      std::make_shared<YAInfer::ReluOperator>(thresh);
  // the operator is registered by the factory method
  std::shared_ptr<YAInfer::Layer> relu_layer =
      YAInfer::LayerRegisterer::CreateLayer(relu_op);

  std::shared_ptr<YAInfer::Tensor<float>> input =
      std::make_shared<YAInfer::Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> inputs;
  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> outputs;

  inputs.push_back(input);
  relu_layer->Forwards(inputs, outputs);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}