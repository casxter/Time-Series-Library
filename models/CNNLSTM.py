import torch
from torch import nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(input_size=configs.d_model, hidden_size=configs.cnnlstm_hidden, num_layers=configs.cnnlstm_nl, batch_first=True)
        self.fc1 = nn.Linear(int(configs.seq_len / 2), configs.pred_len)
        self.fc2 = nn.Linear(configs.cnnlstm_hidden, 1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #[b,sl,feature] [b,sl,4] [b,ll+pl,feature] [b,ll+pl,4]
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Permute to match the expected input shape for CNN
        enc_out = enc_out.permute(0, 2, 1)

        cnn_out = self.cnn(enc_out)

        # Permute back to match the expected input shape for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(cnn_out)

        # We are interested in the output of the last time step
        fc1_out = self.fc1(lstm_out.permute(0, 2, 1))
        fc2_out = self.fc2(fc1_out.permute(0, 2, 1))

        # [b, pl, feature]
        return fc2_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def classification(self, x_enc, x_mark_enc):
       pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
