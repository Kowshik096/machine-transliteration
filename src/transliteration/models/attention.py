"""Seq2Seq with Bahdanau additive attention."""

import random
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Bidirectional RNN encoder (BiRNN)."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 1,
        cell_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self, src: torch.Tensor, src_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs_packed, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        if self.bidirectional:
            outputs = outputs[:, :, : self.hidden_dim] + outputs[:, :, self.hidden_dim :]
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Additive attention: score(h_t, e_i) = v^T tanh(W_enc e_i + U_dec h_t)."""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.U_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        enc_outputs: torch.Tensor,
        dec_hidden: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_outputs: (batch, src_len, enc_dim)
        # dec_hidden: (batch, dec_dim)
        enc_proj = self.W_enc(enc_outputs)
        dec_proj = self.U_dec(dec_hidden).unsqueeze(1)
        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)
        scores = scores.masked_fill(src_mask, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)
        return context.squeeze(1), attn_weights


class Decoder(nn.Module):
    """Unidirectional RNN decoder with Bahdanau additive attention."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        enc_dim: int = 256,
        num_layers: int = 1,
        cell_type: str = "lstm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim + enc_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = BahdanauAttention(enc_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim + enc_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_tok: torch.Tensor,
        hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        dec_hidden = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
        context, attn_weights = self.attention(enc_outputs, dec_hidden, src_mask)
        emb = self.embedding(input_tok).unsqueeze(1)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        logits = self.fc_out(self.dropout(torch.cat([output, context], dim=1)))
        return logits, hidden, attn_weights


class AttentionSeq2Seq(nn.Module):
    """Seq2Seq with Bahdanau attention."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_pad_idx: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def _init_decoder_hidden(
        self, enc_hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        num_layers = self.encoder.num_layers
        if self.encoder.bidirectional:
            if isinstance(enc_hidden, tuple):
                h, c = enc_hidden
                h = h.view(num_layers, 2, h.size(1), -1).sum(dim=1)
                c = c.view(num_layers, 2, c.size(1), -1).sum(dim=1)
                return (h, c)
            h = enc_hidden.view(num_layers, 2, enc_hidden.size(1), -1).sum(dim=1)
            return h
        return enc_hidden

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lens: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        src_mask = src == self.src_pad_idx

        enc_outputs, enc_hidden = self.encoder(src, src_lens)
        dec_hidden = self._init_decoder_hidden(enc_hidden)

        outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc_out.out_features, device=src.device)
        input_tok = tgt[:, 0]  # <SOS>

        for t in range(1, tgt_len):
            logits, dec_hidden, _ = self.decoder(input_tok, dec_hidden, enc_outputs, src_mask)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return outputs