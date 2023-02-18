/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-18 14:11:29
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  FLAGS_alsologtostderr = true;
  LOG(INFO) << "Start test..." << std::endl;
  return RUN_ALL_TESTS();
}