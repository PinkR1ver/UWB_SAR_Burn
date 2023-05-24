import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(TransformerModule, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)

    def forward(self, feature, sequence):
        # Embed the feature vector
        embedded_feature = self.embedding(feature)

        # Expand the feature to match the sequence length
        expanded_feature = embedded_feature.unsqueeze(1).expand(-1, sequence.size(1), -1)

        # Concatenate or add the feature with the sequence
        combined_input = torch.cat((expanded_feature, sequence), dim=2)

        # Apply the transformer module
        output = self.transformer(combined_input, combined_input)

        return output


# Example usage
input_size = 32
hidden_size = 64
num_heads = 4
num_layers = 2

# Create an instance of the Transformer module
transformer = TransformerModule(input_size, hidden_size, num_heads, num_layers)

# Generate dummy input
batch_size = 10
feature = torch.randn(batch_size, input_size)
sequence_length = 20
sequence = torch.randn(batch_size, sequence_length, hidden_size)

# Pass the input through the Transformer module
output = transformer(feature, sequence)

print(output.shape)  # Output shape: (batch_size, sequence_length, hidden_size)
