#include "parser/parse_expression.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>

namespace YAInfer {
void Polish(const std::shared_ptr<TokenNode> &root_node,
            std::vector<std::shared_ptr<TokenNode>> &polish) {
  if (root_node != nullptr) {
    Polish(root_node->left, polish);
    Polish(root_node->right, polish);
    polish.push_back();
  }
}
}  // namespace YAInfer