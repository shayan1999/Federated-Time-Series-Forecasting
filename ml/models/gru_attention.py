import torch
import torch.nn.functional as F

class AttentionModule(torch.nn.Module):
    def __init__(self, gru_hidden_size):
        super(AttentionModule, self).__init__()
        self.attention_weights_layer = torch.nn.Linear(gru_hidden_size, 1)

    def forward(self, gru_outputs):
        # gru_outputs shape: (batch_size, seq_length, gru_hidden_size)
        attention_scores = self.attention_weights_layer(gru_outputs)
        attention_scores = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of GRU outputs based on attention scores
        weighted_sum = torch.sum(gru_outputs * attention_scores, dim=1)
        return weighted_sum

class GRUWithAttention(torch.nn.Module):
    def __init__(self, input_dim: int, gru_hidden_size: int = 128, num_gru_layers: int = 2, gru_dropout: float = 0.2,
                 num_outputs: int = 2, exogenous_dim: int = 0):
        super(GRUWithAttention, self).__init__()
        
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=gru_hidden_size,
                                num_layers=num_gru_layers, batch_first=True, dropout=gru_dropout)
        self.attention = AttentionModule(gru_hidden_size)
        self.fc = torch.nn.Linear(gru_hidden_size + exogenous_dim, num_outputs)

        # Initialize weights
        self.gru.apply(self._init_weights)
        self.fc.apply(self._init_weights)

    def forward(self, x, exogenous_data=None, device="cpu"):
        x = x.to(device)
        if exogenous_data is not None:
            exogenous_data = exogenous_data.to(device)

        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(device)
        gru_out, _ = self.gru(x, h0)
        
        attention_out = self.attention(gru_out)
        
        if exogenous_data is not None:
            attention_out = torch.cat((attention_out, exogenous_data), dim=1)
        
        out = self.fc(attention_out)
        return out

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.GRU):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

