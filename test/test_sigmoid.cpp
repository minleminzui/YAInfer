/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-25 10:57:51
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "factory/layer_factory.hpp"
#include "layer/layer.hpp"
#include "layer/sigmoid_layer.hpp"
#include "ops/op.hpp"

TEST(TestLayer, ForwardSigmoid1) {
  std::shared_ptr<YAInfer::Operator> sigmoid_op =
      std::make_shared<YAInfer::SigmoidOperator>();

  std::shared_ptr<YAInfer::Tensor<float>> input =
      std::make_shared<YAInfer::Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(0) = -2.f;
  input->index(0) = 3.f;

  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> inputs;
  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> outputs;

  inputs.push_back(input);
  YAInfer::SigmoidLayer layer(sigmoid_op);
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 1 / (1 + std::exp(-input->index(0))));
    ASSERT_EQ(outputs.at(i)->index(1), 1 / (1 + std::exp(-input->index(1))));
    ASSERT_EQ(outputs.at(i)->index(2), 1 / (1 + std::exp(-input->index(2))));
  }
}

TEST(TestLayer, ForwardSigmoid2) {
  std::shared_ptr<YAInfer::Operator> sigmoid_op =
      std::make_shared<YAInfer::SigmoidOperator>();
  std::shared_ptr<YAInfer::Layer> sigmoid_layer =
      YAInfer::LayerRegisterer::CreateLayer(sigmoid_op);

  std::shared_ptr<YAInfer::Tensor<float>> input =
      std::make_shared<YAInfer::Tensor<float>>(1, 1, 3);

  input->index(0) = -1.f;
  input->index(0) = -2.f;
  input->index(0) = 3.f;

  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> inputs;
  std::vector<std::shared_ptr<YAInfer::Tensor<float>>> outputs;

  inputs.push_back(input);
  sigmoid_layer->Forwards(inputs, outputs);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 1 / (1 + std::exp(-input->index(0))));
    ASSERT_EQ(outputs.at(i)->index(1), 1 / (1 + std::exp(-input->index(1))));
    ASSERT_EQ(outputs.at(i)->index(2), 1 / (1 + std::exp(-input->index(2))));
  }
}