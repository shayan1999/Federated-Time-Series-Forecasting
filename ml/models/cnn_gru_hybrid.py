import torch

class CNN_GRU_Hybrid(torch.nn.Module):
    def __init__(self, cnn, gru_with_attention, num_features, num_lags, gru_hidden_size):
        super(CNN_GRU_Hybrid, self).__init__()
        self.cnn = cnn
        self.gru_with_attention = gru_with_attention
        # Assuming the CNN reduces each time lag into a single feature vector
        self.fc = torch.nn.Linear(num_features * num_lags, gru_hidden_size)

    def forward(self, x, exogenous_data=None):
        # Reshape x to fit CNN input if necessary
        # x: (batch_size, num_lags, num_features) -> (batch_size, 1, num_lags, num_features)
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        # Flatten CNN output to match GRU input expectations
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = self.fc(cnn_out)
        # Reshape for GRU: (batch_size, seq_len, features)
        cnn_out = cnn_out.view(cnn_out.size(0), 1, -1)
        if exogenous_data is not None:
            # Concatenate exogenous data if present
            cnn_out = torch.cat((cnn_out, exogenous_data.unsqueeze(1)), dim=2)
        output = self.gru_with_attention(cnn_out)
        return output
