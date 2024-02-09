import torch
import torch.nn.functional as F

class AttentionModule(torch.nn.Module):
    def __init__(self, lstm_hidden_size):
        super(AttentionModule, self).__init__()
        self.attention_weights_layer = torch.nn.Linear(lstm_hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_length, lstm_hidden_size)
        attention_scores = self.attention_weights_layer(lstm_outputs)
        attention_scores = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        weighted_sum = torch.sum(lstm_outputs * attention_scores, dim=1)
        return weighted_sum

# Assume LSTM model from your code
class LSTMWithAttention(torch.nn.Module):
    def __init__(self, lstm_model, lstm_hidden_size):
        super(LSTMWithAttention, self).__init__()
        self.lstm_model = lstm_model
        self.attention = AttentionModule(lstm_hidden_size)

    def forward(self, x, exogenous_data=None, device="cpu"):
        lstm_out = self.lstm_model(x, exogenous_data=exogenous_data, device=device)
        attention_out = self.attention(lstm_out)
        return attention_out
