import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnLayer(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dim:int,
                 rnn_type:str):
        super().__init__()
        if rnn_type == 'GRU':
            layer = nn.GRU
        elif rnn_type == 'LSTM':
            layer = nn.LSTM
        else:
            raise Exception('unsupported type')

        self.rnn = layer(input_dim, hidden_dim, batch_first=True)


class Encoder(RnnLayer):
    def __init__(self,
                 num_vocab:int,
                 emb_dim:int,
                 hidden_dim:int,
                 drop:float,
                 rnn_type:str):
        super().__init__(emb_dim, hidden_dim, rnn_type)
        self.dropout = nn.Dropout(drop)
        self.embedding = nn.Embedding(num_vocab, emb_dim)

    def forward(self, seq):
        '''
        seq: [batch_size, seq_len]
        '''
        emb = self.dropout(self.embedding(seq)) # [batch_size, seq_len, emb_dim]
        _, hid = self.rnn(emb)
        return hid


class Decoder(RnnLayer):
    def __init__(self,
                 num_vocab:int,
                 emb_dim:int,
                 hidden_dim:int,
                 drop:float,
                 tied:bool,
                 rnn_type:str):
        super().__init__(emb_dim, hidden_dim, rnn_type)
        self.dropout = nn.Dropout(drop)
        self.embedding = nn.Embedding(num_vocab, emb_dim)
        if not tied:
            self.fc = nn.Linear(hidden_dim, num_vocab, bias=False)
        self.tied = tied

    def forward(self, x, hid):
        '''
        call this for each time frame
        x: [batch_size]
        '''
        emb = self.dropout(self.embedding(x.unsqueeze(1))) # [batch_size, 1, emb_dim]
        out, hid = self.rnn(emb, hid) # out: [batch_size, 1, hidden_dim]
        if self.tied:
            out = out.squeeze(1) @ self.embedding.weight.t()
        else:
            out = self.fc(out.squeeze(1)) # [batch_size, num_vocab]
        return out, hid


class Network(nn.Module):
    def __init__(self,
                 num_vocab_in:int,
                 num_vocab_out:int,
                 emb_dim:int,
                 hidden_dim:int,
                 drop:float,
                 rnn_type:str='GRU',
                 tied:bool=False,
                 device:str='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(num_vocab_in, emb_dim, hidden_dim, drop, rnn_type)
        self.decoder = Decoder(num_vocab_out, emb_dim, hidden_dim, drop, tied, rnn_type)

    def forward(self, seq_in, seq_trg, force_prob:float=0.5):
        '''
        seq_in: [batch_size, seq_in_len]
        seq_trg: [batch_size, seq_trg_len]
        '''
        seq_trg_len = seq_trg.shape[1]
        force_teach = torch.rand((seq_trg_len,)) < force_prob
        hid = self.encoder(seq_in)

        trg = seq_trg[:,0]
        result = seq_trg.clone()
        loss = 0
        for idx in range(seq_trg_len - 1):
            out, hid = self.decoder(trg, hid) # out: [batch_size, num_vocab]
            loss += F.cross_entropy(out, seq_trg[:,idx+1])
            top1 = out.argmax(-1)
            result[:,idx+1] = top1
            if force_teach[idx].item():
                trg = seq_trg[:,idx+1]
            else:
                trg = top1

        return loss, result


if __name__ == '__main__':
    network = Network(num_vocab_in=10,
                      num_vocab_out=9,
                      emb_dim=5,
                      hidden_dim=7)
    x = torch.randint(0,10,(2,3))
    y = torch.randint(0,9,(2,4))
    z = network(x, y)