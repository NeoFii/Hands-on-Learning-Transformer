from transformers import PreTrainedTokenizerFast
import jieba

def compare_chinese_tokenization(text, tokenizer_path="tokenizer.json"):
    """比较Transformer分词器与jieba分词器的结果差异
    
    Args:
        text: 要分析的中文文本
        tokenizer_path: Transformer分词器文件路径
        
    Returns:
        comparison: 包含两种分词结果的字典
    """
    # 加载Transformer分词器
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    vocab = tokenizer.get_vocab()
    print(f"词汇表大小: {len(vocab)}")
    
    # 获取Transformer分词结果
    transformer_tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(transformer_tokens)
    
    # 解码每个标记查看实际文本
    decoded_tokens = []
    for token_id in token_ids:
        decoded = tokenizer.decode([token_id]).strip()
        decoded_tokens.append(decoded)
    
    # 获取jieba分词结果
    jieba_tokens = list(jieba.cut(text))
    
    # 显示结果
    print("=" * 50)
    print(f"原文: {text}")
    print("=" * 50)
    
    print("\n【Transformer分词结果】")
    print(f"标记数量: {len(transformer_tokens)}")
    
    # 显示标记与其ID及解码后的文本
    print("\n标记详情:")
    for i, (token, token_id, decoded) in enumerate(zip(transformer_tokens, token_ids, decoded_tokens)):
        print(f"  {i+1}. 原始标记: '{token}' | ID: {token_id} | 解码文本: '{decoded}'")
    
    # 带边界标记的文本展示
    marked_text = " | ".join([t for t in decoded_tokens if t])
    print(f"\n边界标记文本:\n{marked_text}")
    
    # 与jieba对比
    print("\n【Jieba分词结果】")
    print(f"分词数量: {len(jieba_tokens)}")
    jieba_text = " | ".join(jieba_tokens)
    print(f"分词结果:\n{jieba_text}")
    
    # 返回对比结果
    return {
        "transformer": {
            "tokens": transformer_tokens,
            "decoded": decoded_tokens,
            "count": len(transformer_tokens)
        },
        "jieba": {
            "tokens": jieba_tokens,
            "count": len(jieba_tokens)
        }
    }


def analyze_tokenization_differences(text, tokenizer_path="tokenizer.json"):
    """深入分析两种分词器的差异
    
    Args:
        text: 要分析的中文文本
        tokenizer_path: Transformer分词器文件路径
    """
    # 获取基本分词结果
    result = compare_chinese_tokenization(text, tokenizer_path)
    
    # Transformer分词特点分析
    transformer_tokens = result["transformer"]["decoded"]
    transformer_lengths = [len(token) for token in transformer_tokens if token]
    
    # jieba分词特点分析
    jieba_tokens = result["jieba"]["tokens"]
    jieba_lengths = [len(token) for token in jieba_tokens]
    
    # 计算统计信息
    print("\n【分词特点对比】")
    print(f"Transformer平均分词长度: {sum(transformer_lengths)/len(transformer_lengths):.2f} 字符")
    print(f"Jieba平均分词长度: {sum(jieba_lengths)/len(jieba_lengths):.2f} 字符")
    
    print(f"\nTransformer最长分词: '{max(transformer_tokens, key=len)}' ({len(max(transformer_tokens, key=len))}字符)")
    print(f"Jieba最长分词: '{max(jieba_tokens, key=len)}' ({len(max(jieba_tokens, key=len))}字符)")
    
    # 找出完全匹配的分词
    transformer_set = set([t for t in transformer_tokens if t])
    jieba_set = set(jieba_tokens)
    common_tokens = transformer_set.intersection(jieba_set)
    
    print(f"\n两种分词器共同识别的词语数量: {len(common_tokens)}")
    if common_tokens:
        print(f"共同词语: {', '.join(common_tokens)}")
    
    # 分析Transformer特有的subword特点
    subword_count = sum(1 for t in transformer_tokens if t.startswith('##'))
    print(f"\nTransformer中的子词(subword)数量: {subword_count}")


# 使用示例
if __name__ == "__main__":
    # 测试文本
    text = "deepseek的母公司幻方量化是一家量化投资的大型企业，在业界享有很高的声誉"
    
    # 运行分词对比
    comparison = compare_chinese_tokenization(text)
    
    # 进一步分析差异（可选）
    # analyze_tokenization_differences(text)