/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-27 11:38:45
 */
#pragma once

namespace YAInfer {
enum class RuntimeParameterType {
  kParameterUnknown = 0,
  kParameterBool = 1,
  kParameterInt = 2,

  kParameterFloat = 3,
  kParameterString = 4,
  kParameterIntArray = 5,
  kParameterFloatArray = 6,
  kParameterStringArray = 7,
};

enum class InferStatus {
  kInferUnknown = -1,
  kInferFailedInputEmpty = 1,
  kInferFailedWeightParameterError = 2,
  kInferFailedBiasParameterError = 3,
  kInferFailedStrideParameterError = 4,
  kInferFailedDimensionParameterError = 5,
  kInferFailedChannelParameterError = 6,
  kInferFailedInputOutSizeAdaptingError = 7,

  kInferFailedOutputSizeError = 8,
  kInferFailedOperationUnknown = 9,
  kInferFailedYoloStageNumberError = 10,

  kInferSuccess = 0,
};

enum class ParseParametersAttrStatus {
  kParameterMissingUnknown = -1,
  kParameterMissingStride = 1,
  kParameterMissingPadding = 2,
  kParameterMissingKernel = 3,
  kParameterMissingUseBias = 4,
  kParameterMissingInChannel = 5,
  kParameterMissingOutChannel = 6,

  kParameterMissingEps = 7,
  kPara
};
}