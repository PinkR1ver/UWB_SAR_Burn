import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequences, parameters):
        input_sequences = input_sequences.permute(1, 0, 2)  # [seq_len, batch_size, input_size]
        parameters = parameters.permute(1, 0)  # [seq_len, batch_size]

        encoder_output = self.encoder(input_sequences)  # [seq_len, batch_size, hidden_size]
        decoder_input = parameters.unsqueeze(2)  # [seq_len, batch_size, 1, input_size]
        decoder_output = self.decoder(decoder_input)  # [seq_len, batch_size, 1, output_size]
        decoder_output = decoder_output.squeeze(2)  # [seq_len, batch_size, output_size]

        return decoder_output.permute(1, 0, 2)  # [batch_size, seq_len, output_size]
    
if __name__ == '__main__':
    pass
