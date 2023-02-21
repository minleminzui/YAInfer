/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-18 19:45:57
 */
/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-18 13:46:58
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <armadillo>

TEST(TestFirst, Demo1) {
  LOG(INFO) << "First Test";
  arma::fmat in(32, 32, arma::fill::ones);
  ASSERT_EQ(in.n_cols, 32);
  ASSERT_EQ(in.n_rows, 32);
  ASSERT_EQ(in.size(), 32 * 32);
}

TEST(TestFirst, Linear) {
  arma::fmat A =
      "1,2,3;"
      "4,5,6;"
      "7,8,9;";

  arma::fmat X =
      "1,1,1;"
      "1,1,1;"
      "1,1,1;";

  arma::fmat bias =
      "1,1,1;"
      "1,1,1;"
      "1,1,1;";

  arma::fmat output(3, 3);
  output = A * X + bias;

  const uint32_t cols = 3;
  for (uint32_t c = 0; c < cols; ++c) {
    float *col_ptr = output.colptr(c);
    ASSERT_EQ(*(col_ptr + 0), 7);
    ASSERT_EQ(*(col_ptr + 1), 16);
    ASSERT_EQ(*(col_ptr + 2), 25);
  }
  LOG(INFO) << "\n"
            << "Result Passed!";
}