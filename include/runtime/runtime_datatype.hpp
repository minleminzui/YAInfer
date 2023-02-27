/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-26 17:23:38
 */
#pragma once
enum class RuntimeDataType {
  kTypeUnknown = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUint8 = 8,
};