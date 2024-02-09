class HybridLSTMAndCNN(nn.Module):
    def __init__(self, lstm_input_dim, cnn_input_dim, lstm_hidden_size, cnn_out_channels, num_classes):
        super(HybridLSTMAndCNN, self).__init__()
        # Define LSTM component
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, batch_first=True)
        # Define CNN component
        self.cnn = nn.Conv1d(in_channels=cnn_input_dim, out_channels=cnn_out_channels, kernel_size=3)
        # Define fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size + cnn_out_channels, num_classes)

    def forward(self, lstm_input, cnn_input):
        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_input)
        # CNN forward pass
        cnn_output = self.cnn(cnn_input)
        # Flatten CNN output
        cnn_output = cnn_output.view(cnn_output.size(0), -1)
        # Concatenate LSTM and CNN outputs
        combined_output = torch.cat((lstm_output[:, -1, :], cnn_output), dim=1)
        # Fully connected layer
        output = self.fc(combined_output)
        return output
