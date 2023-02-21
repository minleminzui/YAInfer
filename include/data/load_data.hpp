/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-21 13:33:26
 */
#pragma once

#include <armadillo>

#include "data/tensor.hpp"
namespace YAInfer {

class CSVDataLoader {
 public:
  /**
   * @description: Load CSV data and initialize a tensor using its data
   * @param file
   * @param split_char
   * @return 
   */
  static std::shared_ptr<Tensor<float>> LoadData(const std::string &file_path,
                                                 const char split_char = ',');
  /**
   * @description: Load CSV data with headers and initialize a tensor using its
   * data
   * @param file
   * @param headers
   * @param split_char
   * @return 
   */
  static std::shared_ptr<Tensor<float>> LoadDataWithHeader(
      const std::string &file_path, std::vector<std::string> &header,
      const char split_char = ',');

 private:
  /**
   * @description: get the size of the CSV data
   * @param file  
   * @param split_char
   * @return the size of CSV data
   */
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file,
                                                 char split_char);
};
}  // namespace YAInfer