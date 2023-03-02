/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-03-02 11:57:11
 */
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace YAInfer {
enum class TokenType {
  TokenUnknown = -1,
  TokenInputNumber = 0,
  TokenComma = 1,
  TokenAdd = 2,
  TokenMul = 3,
  TokenLeftBracket = 4,
  TokenRightBracket = 5,
};

struct Token {
  TokenType token_type = TokenType::TokenUnknown;
  int32_t start_pos = 0;
  int32_t end_pos = 0;
  Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
      : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {}
};

// TokenNode in the abstract syntax tree
struct TokenNode {
  int32_t num_index = -1;
  std::shared_ptr<TokenNode> left = nullptr;
  std::shared_ptr<TokenNode> right = nullptr;
  TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
            std::shared_ptr<TokenNode> right);
  TokenNode() = default;
};

// add(add(add(@0,@1),@1),add(@0,@2))
class ExpressionParser {
 public:
  explicit ExpressionParser(std::string statement)
      : statement_(std::move(statement)) {}

  void Tokenizer(bool need_retoken = false);

  std::shared_ptr<TokenNode> Generate();

  const std::vector<Token> &tokens() const;

  const std::vector<std::string> &token_strs() const;

 private:
  std::shared_ptr<TokenNode> Generate_(int32_t &index);
  std::vector<Token> tokens_;
  std::vector<std::string> token_strs_;
  std::string statement_;
};
}  // namespace YAInfer