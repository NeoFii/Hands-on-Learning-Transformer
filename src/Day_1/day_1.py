import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
import seaborn as sns

class TransformerInputProcessor:
    """用于处理Transformer输入的类，包含分词、嵌入和位置编码功能"""
    
    def __init__(self, tokenizer_path, embedding_dim=1024, dropout_rate=0.1):
        """初始化处理器
        
        Args:
            tokenizer_path: 分词器文件路径
            embedding_dim: 嵌入向量维度，默认1024
            dropout_rate: Dropout比率，默认0.1
        """
        # 加载分词器
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        
        # 创建嵌入层
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # 创建dropout层
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        
        print(f"初始化完成 - 词汇表大小: {self.vocab_size}, 嵌入维度: {self.embedding_dim}")
    
    def tokenize(self, text):
        """将文本转换为token IDs
        
        Args:
            text: 输入文本
            
        Returns:
            tokens: 分词结果
            token_ids: token ID列表
        """
        # 使用分词器处理文本
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return tokens, token_ids
    
    def get_embeddings(self, token_ids):
        """获取token的嵌入向量
        
        Args:
            token_ids: token ID列表或张量
            
        Returns:
            embeddings: 嵌入向量
        """
        # 如果输入是列表，转换为张量
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)
            
        # 获取嵌入向量
        embeddings = self.embedding_layer(token_ids)
        
        return embeddings
    
    def create_positional_encoding(self, seq_length):
        """创建位置编码矩阵
        
        Args:
            seq_length: 序列长度
            
        Returns:
            positional_encoding: 位置编码矩阵 [seq_length, embedding_dim]
        """
        # 创建位置索引 [seq_length, 1]
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        
        # 创建除数项 [embedding_dim/2]
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2).float() * 
            (-np.log(10000.0) / self.embedding_dim)
        )
        
        # 初始化位置编码矩阵
        pe = torch.zeros(seq_length, self.embedding_dim)
        
        # 计算位置编码：偶数索引使用正弦函数，奇数索引使用余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def create_input_representation(self, text, apply_scaling=False, apply_dropout=True):
        """创建完整的Transformer输入表示
        
        Args:
            text: 输入文本
            apply_scaling: 是否应用sqrt(d_model)缩放，默认False
            apply_dropout: 是否应用dropout，默认True
            
        Returns:
            final_input: 最终输入表示
            tokens: 分词结果
            token_ids: token ID列表
        """
        # 分词
        tokens, token_ids = self.tokenize(text)
        print(f"分词结果 - tokens数量: {len(tokens)}")
        
        # 获取嵌入向量
        embeddings = self.get_embeddings(token_ids)
        
        # 可选：应用缩放因子
        if apply_scaling:
            embeddings = embeddings * (self.embedding_dim ** 0.5)
            print("已应用嵌入向量缩放")
        
        # 创建位置编码
        pos_encoding = self.create_positional_encoding(len(token_ids))
        
        # 将嵌入向量与位置编码相加
        final_input = embeddings + pos_encoding
        
        # 可选：应用dropout
        if apply_dropout:
            final_input = self.dropout(final_input)
        
        print(f"处理完成 - 最终输入表示形状: {final_input.shape}")
        
        return final_input, tokens, token_ids
    
    def visualize_positional_encoding(self, seq_length=50, sample_dims=100):
        """可视化位置编码矩阵
        
        Args:
            seq_length: 要可视化的序列长度
            sample_dims: 要可视化的维度数量
        """
        # 创建位置编码
        pe = self.create_positional_encoding(seq_length)
        
        # 截取要可视化的部分
        pe_sample = pe[:, :sample_dims].numpy()
        
        # 创建热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(pe_sample, cmap='viridis')
        plt.title('位置编码热力图')
        plt.xlabel('嵌入维度')
        plt.ylabel('序列位置')
        plt.show()
        
        # 可视化几个特定维度的变化趋势
        dims_to_show = [0, 10, 30, 60, 99]  # 选择几个不同的维度
        plt.figure(figsize=(12, 6))
        
        for dim in dims_to_show:
            plt.plot(pe[:, dim].numpy(), label=f'维度 {dim}')
            
        plt.title('位置编码在不同维度上的变化')
        plt.xlabel('序列位置')
        plt.ylabel('编码值')
        plt.legend()
        plt.grid(True)
        plt.show()


# 使用示例
def main():
    processor = TransformerInputProcessor(tokenizer_path="tokenizer.json")
    
    # 测试文本
    text = "deepseek的母公司幻方量化是一家量化投资的大企业，在业界享有很高的声誉"
    
    # 创建输入表示
    final_input, tokens, token_ids = processor.create_input_representation(text)
    
    # 打印位置编码样本
    print("\n位置编码样本（前5个位置，前10个维度）:")
    sample_pe = processor.create_positional_encoding(5)[:, :10]
    print(sample_pe)


if __name__ == "__main__":
    main()