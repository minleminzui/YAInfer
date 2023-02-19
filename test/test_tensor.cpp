/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-19 22:29:36
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <armadillo>

#include "data/tensor.hpp"
TEST(TestTensor, Create) {
  YAInfer::Tensor<float> tensor(3, 31, 32);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 31);
  ASSERT_EQ(tensor.cols(), 32);
}

TEST(TestTensor, Fill) {
  YAInfer::Tensor<float> tensor(3, 3, 3);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 3);
  ASSERT_EQ(tensor.cols(), 3);

  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back((float)i);
  }

  tensor.Fill(values);
  LOG(INFO) << tensor.data();

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        ASSERT_EQ(values.at(index), tensor.at(c, r, c_));
        index += 1;
      }
    }
  }
  LOG(INFO) << "Test1 Passed!!!";
}

TEST(TestTensor, Padding) {
  YAInfer::Tensor<float> tensor(3, 3, 3);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 3);
  ASSERT_EQ(tensor.cols(), 3);

  tensor.Fill(1.f);
  tensor.Padding({1, 1, 1, 1}, 0);
  ASSERT_EQ(tensor.rows(), 5);
  ASSERT_EQ(tensor.cols(), 5);

  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if (r == 0 || c_ == 0) {
          ASSERT_EQ(tensor.at(c, r, c_), 0);
        }
      }
    }
  }

  LOG(INFO) << "Test2 Passed!!!";
}