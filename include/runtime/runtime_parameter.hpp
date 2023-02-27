/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-27 11:22:30
 */
#pragma once

#include <string>
#include <vector>

#include "status_code.hpp"

namespace YAInfer {
// there are several categories params of computation nodes:
// 1. int
// 2. float
// 3. string
// 4. bool
// 5. int array
// 6. string array
// 7. float array
struct RuntimeParameter {  // parameters of computation nodes
  virtual ~RuntimeParameter() = default;
  explicit RuntimeParameter(
      RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
      : type(type) {}
  RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};

struct RuntimeParameterInt : public RuntimeParameter {
  RuntimeParameterInt()
      : RuntimeParameter(RuntimeParameterType::kParameterInt) {}
  int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
  RuntimeParameterFloat()
      : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}
  float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
  RuntimeParameterString()
      : RuntimeParameter(RuntimeParameterType::kParameterString) {}
  std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
  RuntimeParameterIntArray()
      : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}
  std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
  RuntimeParameterFloatArray()
      : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}
  std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
  RuntimeParameterStringArray()
      : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}
  std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
  RuntimeParameterBool()
      : RuntimeParameter(RuntimeParameterType::kParameterBool) {}
  bool value = false;
};
}  // namespace YAInfer