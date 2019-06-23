import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dim:int,
                 rnn_type='GRU'):
        super().__init__()
        if rnn_type == 'GRU':
            layer = nn.GRU
        elif rnn_type == 'LSTM':
            layer = nn.LSTM
        else:
            raise Exception('unsupported type')

        self.rnn = layer(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        '''
        x is embedding vector of shape
        [batch_size, seq_len, input_dim]
        '''
        seq_len = x.shape[1]
        hid = None
        for idx in range(seq_len):
            token = x[:,[idx]]
            _, hid = self.rnn(token, hid)

        if isinstance(self.rnn, nn.LSTM):
            hid =  hid[0]

        return hid.view(len(x), -1)


if __name__ == '__main__':
    import torch

    encoder = Encoder(128, 128)
    x = torch.randn(2, 20, 128)
    y = encoder(x)
    print(y.shape)