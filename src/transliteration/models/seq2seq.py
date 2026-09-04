"""RNN-based Seq2Seq model without attention (baseline)."""

import random
from typing import Optional, Union

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """Unidirectional or bidirectional RNN encoder."""

    def __init__(
        self,
        input_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        cell_type: str = "gru",
        bidirectional: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        assert not bidirectional or hidden_dim % 2 == 0, "hidden_dim must be even for bidirectional"
        per_dir_hidden = hidden_dim // self.num_directions

        self.embedding = nn.Embedding(input_vocab_size, emb_dim)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim,
            per_dir_hidden,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(
        self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        embedded = self.embedding(src)
        if src_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs_packed, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class DecoderRNN(nn.Module):
    """Unidirectional RNN decoder."""

    def __init__(
        self,
        output_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        cell_type: str = "gru",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(output_vocab_size, emb_dim)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(
        self,
        input_step: torch.Tensor,
        hidden: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        emb = self.embedding(input_step).unsqueeze(1)
        output, hidden = self.rnn(emb, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden


class Seq2Seq(nn.Module):
    """Seq2Seq wrapper connecting encoder and decoder with teacher forcing."""

    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        device: torch.device,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _concat_directions(self, h: torch.Tensor) -> torch.Tensor:
        """Sum bidirectional encoder directions to get decoder initial hidden."""
        nl_times_dir, b, per_h = h.size()
        num_directions = 2 if self.encoder.bidirectional else 1
        num_layers = nl_times_dir // num_directions
        h = h.view(num_layers, num_directions, b, per_h)
        h = h.permute(0, 2, 1, 3).contiguous().view(num_layers, b, per_h * num_directions)
        return h

    def _init_decoder_hidden_from_encoder(
        self, enc_hidden: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(enc_hidden, tuple):  # LSTM: (h, c)
            h, c = enc_hidden
            return (self._concat_directions(h), self._concat_directions(c))
        return self._concat_directions(enc_hidden)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        enc_outputs, enc_hidden = self.encoder(src, src_lens)
        dec_hidden = self._init_decoder_hidden_from_encoder(enc_hidden)

        input_tok = trg[:, 0]  # <SOS>
        for t in range(1, trg_len):
            logits, dec_hidden = self.decoder(input_tok, dec_hidden)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1
        return outputs
