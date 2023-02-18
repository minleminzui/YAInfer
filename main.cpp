/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-18 11:49:11
 */
#include <armadillo>
#include <iostream>

int main() {
  arma::fmat in_1(32, 32, arma::fill::ones);
  arma::fmat in_2(32, 32, arma::fill::ones);

  arma::fmat out = in_1 + in_2;
  std::cout << out.at(0) << std::endl;
  std::cout << out.n_rows << std::endl;
  std::cout << out.n_cols << std::endl;
  return 0;
}