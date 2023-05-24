import torch
import torch.nn as nn
import torch.optim as optim

class TFT(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers):
        super(TFT, self).__init__()
        
        # Encoder
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers)
        
        # Decoder
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # Apply encoder to the input sequence
        encoded = self.encoder(x)
        
        # Apply decoder to the encoded sequence
        decoded = self.decoder(encoded)
        
        return decoded

# Define the model hyperparameters
input_dim = 4
output_dim = 1
d_model = 128
nhead = 4
num_layers = 4

# Create an instance of the TFT model
model = TFT(input_dim, output_dim, d_model, nhead, num_layers)

print(model)

'''
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have the time series data stored in `x` and the corresponding targets in `y`

# Convert the data to tensors
x = torch.Tensor(x)  # Shape: (sequence_length, input_dim)
y = torch.Tensor(y)  # Shape: (sequence_length, output_dim)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(x)
    
    # Compute the loss
    loss = criterion(predictions, y)
    
    # Backward pass
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
'''
