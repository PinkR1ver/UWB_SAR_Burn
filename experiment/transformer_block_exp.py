import torch
import torch.nn as nn

# 定义输入维度和输出维度
seq_len = 10
batch_size = 4
input_size = 16
output_size = 8

# 创建随机输入张量
input_tensor = torch.randn(4, seq_len, batch_size, input_size)

# 定义Transformer块
num_heads = 2
hidden_size = 32
num_layers = 3
transformer_block = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
transformer_encoder = nn.TransformerEncoder(transformer_block, num_layers=num_layers)

# 将输入张量进行维度变换以满足Transformer的输入要求
input_tensor_transformed = input_tensor.permute(2, 1, 0, 3).contiguous().view(batch_size, seq_len, -1)

# 输入变换后的张量到Transformer编码器
output_tensor = transformer_encoder(input_tensor_transformed)

# 将输出张量进行维度变换以得到期望的输出维度
output_tensor_transformed = output_tensor.permute(1, 0, 2).contiguous().view(seq_len, batch_size, -1)

# 打印输出张量的大小
print(output_tensor_transformed.size())  # 输出: torch.Size([seq_len, batch_size, output_size])
