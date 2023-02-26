#pragma once
#include <cstdint>

#include "op.hpp"

namespace YAInfer {
class MaxPoolingOp : public Operator {
 public:
  explicit MaxPoolingOp(uint32_t pooling_h, uint32_t pooling_w,
                        uint32_t stride_h, uint32_t stride_w,
                        uint32_t padding_h, uint32_t paddind_w);

  void set_pooling_w(uint32_t pooling_width);
  void set_pooling_h(uint32_t pooling_height);

  void set_stride_w(uint32_t stride_width);
  void set_stride_h(uint32_t stride_height);

  void set_padding_w(uint32_t padding_width);
  void set_padding_h(uint32_t padding_height);

  uint32_t padding_height() const;
  uint32_t padding_width() const;

  uint32_t pooling_height() const;
  uint32_t pooling_width() const;

  uint32_t stride_height() const;
  uint32_t stride_width() const;

 private:
  uint32_t pooling_h_;
  uint32_t pooling_w_;
  uint32_t stride_h_;
  uint32_t stride_w_;
  uint32_t padding_h_;
  uint32_t padding_w_;
};
}  // namespace YAInfer