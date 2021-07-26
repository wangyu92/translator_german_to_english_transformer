from typing import no_type_check
import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        # default : hid_dim=512, n_heads=8, dropout=0.1

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_out = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key   = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        # print(q.shape)

        # Reshape to [batch size, len, num_heads, head_dim]
        q = q.view(batch_size, -1, self.n_heads, self.head_dim)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim)
        # print(q.shape)

        # Permutate [batch size, num_heads, len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Calcaulate energy [batch size, num_heads, query_len, key_len]
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(self.getdevice())
        energy = torch.matmul(q, torch.transpose(k, 2, 3)) / self.scale
        # print(energy.shape)

        if mask is not None:
            # print(f'Energy = {energy.shape}')
            # print(f'Mask = {mask.shape}')
            energy = energy.masked_fill(mask==0, -1e10)
        
        # Attention score [batch size, num_heads, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)

        # Scaled dot-product attention [batch size, num_heads, query_len, head_dim]
        x = torch.matmul(self.dropout(attention), v)

        # [batch, query_len, query_len, num_heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # Flatten [batch size, query_len, embeddig dim]
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_out(x)

        return x, attention


    def getdevice(self):
        return torch.device(next(self.parameters()).device if next(self.parameters()).is_cuda else 'cpu')


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.multihead_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_ff = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src [batch size, src len, hid_dim]
        # src mask [batch size, src len]

        # Multi-head [query, key, value, mask=None]
        _src, _ = self.multihead_attention(src, src, src, src_mask)

        # Dropout, residual connection and layer norm [batch size, src len, hid_dim]
        src = self.attention_layer_norm(src + self.dropout(_src))

        # ---

        # Positionwise feedforward
        _src = self.positionwise_ff(src)

        # Dropout, residual connection and layer norm [batch size, src len, hid dim]
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_len):
        super().__init__()

        self.hid_dim = hid_dim

        self.input_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src [batch size, src len]
        # src mask [batch_size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.getdevice())

        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.getdevice())
        src = self.dropout((self.input_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        # [batch size, src_len, hid_dim]
        return src

    def getdevice(self):
        return torch.device(next(self.parameters()).device if next(self.parameters()).is_cuda else 'cpu')

    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        
        self.multihead_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg [batch size, trg len, hid dim]
        # enc_src [batch size, src len, hid dim]
        # trg mask [batch size, trg len]
        # src mask [batch size, src len]

        # Self-attention
        _trg, _ = self.multihead_attention(trg, trg, trg, trg_mask)
        
        # Dropout, residual and layer norm [batch size, trg_len, hid dim]
        trg = self.self_attention_layer_norm(trg + self.dropout(_trg))

        # Encoder attention [batch size, trg_len, hid_dim]
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attention_layer_norm(trg + self.dropout(_trg))

        # Positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg [batch size, trg len, hid dim]
        # attention [batch size, n_heads, trg len, src len]
        return trg, attention


class Decoder(nn.Module):
    def __init__(self, out_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()

        self.hid_dim = hid_dim

        self.tok_embeddig = nn.Embedding(out_dim, hid_dim)
        self.pos_embeddig = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg [batch size, trg len]
        # enc src [batch size, src len, hid dim]
        # trg mask [batch size, trg len]
        # src mask [batch size, src len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos [batch size, trg len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.getdevice())

        # trg [batch size, trg len, hid dim]
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.getdevice())
        trg = self.dropout((self.tok_embeddig(trg) * self.scale) + self.pos_embeddig(pos))

        # trg = [batch size, trg len, hid_dim]
        # attention = [batch size, n_heads, trg_len, src_len]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)

        # output [batch size, trg len, output dim]
        return output, attention

    def getdevice(self):
            return torch.device(next(self.parameters()).device if next(self.parameters()).is_cuda else 'cpu')


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.getdevice())).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

    def getdevice(self):
            return torch.device(next(self.parameters()).device if next(self.parameters()).is_cuda else 'cpu')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batchsize = 32
    input_len = 512
    embed_dim = 512
    num_heads = 8

    embedding_layer = nn.Embedding(input_len, embed_dim)
    embedding_layer.to(device)

    input_tensor = torch.randint(0, input_len - 1, (batchsize, input_len)).long().to(device)
    input_tensor = embedding_layer(input_tensor)
    # print(input_tensor.shape)
    # print(input_tensor)

    mha = MultiHeadAttentionLayer(embed_dim, num_heads, 0.1)
    mha.to(device)
    query = input_tensor
    out, attention = mha(query, query, query)
    print(out, attention)