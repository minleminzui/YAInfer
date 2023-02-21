/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-21 22:30:36
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "data/load_data.hpp"

TEST(TestDataLoad, LoadCSV) {
  const std::string &file_path = "./tmp/data1.csv";
  std::shared_ptr<YAInfer::Tensor<float>> data =
      YAInfer::CSVDataLoader::LoadData(file_path, ',');
  uint32_t index = 1;
  uint32_t rows = data->rows();
  uint32_t cols = data->cols();
  ASSERT_EQ(rows, 3);
  ASSERT_EQ(cols, 6);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      ASSERT_EQ(data->at(0, r, c), index);
      index += 1;
    }
  }
}

TEST(TestDataLoad, LoadCSVWithHead) {
  const std::string &file_path = "./tmp/data2.csv";
  std::vector<std::string> headers;
  std::shared_ptr<YAInfer::Tensor<float>> data =
      YAInfer::CSVDataLoader::LoadDataWithHeader(file_path, headers, ',');

  uint32_t index = 1;
  uint32_t rows = data->rows();
  uint32_t cols = data->cols();
  LOG(INFO) << "\n" << data;
  ASSERT_EQ(rows, 3);
  ASSERT_EQ(cols, 3);
  ASSERT_EQ(headers.size(), 3);

  ASSERT_EQ(headers.at(0), "ROW1");
  ASSERT_EQ(headers.at(1), "ROW2");
  ASSERT_EQ(headers.at(2), "ROW3");

  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      ASSERT_EQ(data->at(0, r, c), index);
      index += 1;
    }
  }
}