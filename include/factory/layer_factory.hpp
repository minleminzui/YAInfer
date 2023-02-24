/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 15:56:44
 */
#pragma once
#include "layer/layer.hpp"
#include "ops/op.hpp"

namespace YAInfer {
class LayerRegisterer {
 public:
  using Creator =
      std::shared_ptr<Layer> (*)(const std::shared_ptr<Operator> &op);
  using CreateRegistry = std::map<OpType, Creator>;

  static void RegisterCreator(OpType op_type, const Creator &creator);

  static std::shared_ptr<Layer> CreateLayer(
      const std::shared_ptr<Operator> &op);

  static CreateRegistry &Registry();
};

class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(OpType op_type,
                         const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(op_type, creator);
  }
};
}  // namespace YAInfer
