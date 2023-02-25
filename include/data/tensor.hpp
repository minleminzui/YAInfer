/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-19 14:35:55
 */
#pragma once
#include <armadillo>
#include <memory>
#include <vector>

namespace YAInfer {
template <typename T>
class Tensor {};

template <>
class Tensor<uint8_t> {
  // todo, use for quantification
};

template <>
class Tensor<float> {
 public:
  explicit Tensor() = default;

  /**
   * @description: construct a tensor
   * @param {uint32_t} channels
   * @param {uint32_t} rows
   * @param {uint32_t} cols
   * @return {Tensor}
   */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  explicit Tensor(const std::vector<uint32_t> &shapes);

  Tensor(const Tensor<float> &tensor);

  Tensor(Tensor<float> &&tensor) noexcept;

  Tensor<float> &operator=(Tensor<float> &&tensor) noexcept;

  Tensor<float> &operator=(const Tensor<float> &tensor);

  /**
   * @description: return rows of the tensor
   * @return rows of the tensor
   */
  uint32_t rows() const;

  /**
   * @description: return cols of the tensor
   * @return cols of the tensor
   */
  uint32_t cols() const;

  /**
   * @description: return channels of the tensor
   * @return channels of the tensor
   */
  uint32_t channels() const;
  /**
   * @description: return the number of elements in the tensor
   * @return number
   */
  uint32_t size() const;

  /**
   * @description: set the tensor with a concrete data
   * @param {fcube} the concrete data
   */
  void set_data(const arma::fcube &data);
  /**
   * @description: decide whether the tensor empty
   * @return return a bool
   */
  bool empty() const;
  /**
   * @description: return the element of the position by the offset
   * @param {uint32_t} offset
   * @return the element of the position by the offset
   */
  float index(uint32_t offset) const;

  /**
   * @description: return the element of the position by the offset
   * @param {uint32_t} offset
   * @return the element of the position by the offset
   */
  float &index(uint32_t offset);

  /**
   * @description: return the shape of this tensor
   * @return shape
   */
  std::vector<uint32_t> shapes() const;

  /**
   * @description: return the data in the tensor
   * @return return the data
   */
  arma::fcube &data();

  /**
   * @description: return the data in the tensor
   * @return return the data
   */
  const arma::fcube &data() const;

  /**
   * @description: reuturn the specified channel in the tensor
   * @param {uint32_t} channel
   * @return the specified channel in the tensor
   */
  arma::fmat &at(uint32_t channel);

  /**
   * @description: reuturn the specified channel in
   * @param {uint32_t} channel
   * @return the specified channel in the tensor
   */
  const arma::fmat &at(uint32_t channel) const;

  /**
   * @description: return an element at a specific location, according to the
   * channel, row, col
   * @param {uint32_t} channel
   * @param {uint32_t} row
   * @param {uint32_t} col
   * @return elemnt
   */
  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  // const and non-const version?
  /**
   * @description: return an element at a specific lo
   * channel, row, col
   * @param {uint32_t} channel
   * @param {uint32_t} row
   * @param {uint32_t} col
   * @return elemnt
   */
  float &at(uint32_t channel, uint32_t row, uint32_t col);

  /**
   * @description: padding the tensor with specified value
   * @param {vector<uint32_t>} &pads
   * @param {float} padding_value
   * @return
   */
  void Padding(const std::vector<uint32_t> &pads, float padding_value);

  /**
   * @description: Fill the tensor with specifed value
   * @param {float} value
   * @return
   */
  void Fill(float value);

  /**
   * @description: Fill the tensor using a vector
   * @param {vector<float>} &values
   * @return
   */
  void Fill(const std::vector<float> &values);

  /**
   * @description: initialize the tensor with ones
   * @return
   */
  void Ones();

  /**
   * @description: initialize the tenosr with random values
   * @return
   */
  void Rand();

  /**
   * @description:  print the tensor
   * @return
   */
  void Show();

  /**
   * @description: flatten the tensor
   * @return
   */
  void Flatten();

  /**
   * @description: return a deep-copy of the tensor
   * @return the new tensor
   */
  std::shared_ptr<Tensor> Clone();

 private:
  std::vector<uint32_t> raw_shapes_;  // the specific dimension of the tensor
  arma::fcube data_;                  // the data of the tensor
};
}  // namespace YAInfer
