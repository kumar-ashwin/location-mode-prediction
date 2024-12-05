import torch.nn as nn
import numpy as np
import torch, math
from torch import Tensor

import torch.nn.functional as F

from models.embed_geotron import AllEmbeddingGeoTron

class SimpleEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(SimpleEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbeddingGeoTron(self.d_input, config)

        # Simplified encoder: single linear layer with activation
        self.encoder = nn.Sequential(
            nn.Linear(self.d_input, self.d_input),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Output fully connected layer
        self.fc = FullyConnected(self.d_input, config, if_residual_layer=True)

        # Initialize weights
        self._init_weights()

    def forward(self, src, context_dict, device) -> Tensor:
        # Embedding
        emb = self.Embedding(src, context_dict)

        # Pass through the simplified encoder
        encoded_seq = self.encoder(emb)  # Shape: (seq_len, batch_size, feature_dim)

        # Compute valid mask for non-padded values
        src_padding_mask = (src == 0).transpose(0, 1).to(device)  # Shape: (batch_size, seq_len)
        valid_mask = ~src_padding_mask  # Shape: (batch_size, seq_len)

        # Expand valid_mask for feature dimensions
        valid_mask = valid_mask.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

        # Permute encoded_seq to match mask dimensions
        encoded_seq = encoded_seq.permute(1, 0, 2)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute the number of valid timesteps for each sequence
        valid_lengths = valid_mask.sum(dim=1, keepdim=True).float()  # Shape: (batch_size, 1, 1)

        # Apply masking and compute mean pooling
        aggregated_out = (encoded_seq * valid_mask).sum(dim=1) / valid_lengths.squeeze(-1)  # Shape: (batch_size, feature_dim)

        # Pass the aggregated output through the fully connected layer
        fc_output = self.fc(aggregated_out, context_dict)

        return fc_output


    def _init_weights(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LSTMEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(LSTMEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbeddingGeoTron(self.d_input, config)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.d_input,
            hidden_size=config.dim_feedforward,
            num_layers=2,
            batch_first=False,  # expects (seq_len, batch_size, feature_dim)
            bidirectional=False,
            dropout=config.dropout,
        )

        self.fc = FullyConnected(config.dim_feedforward, config, if_residual_layer=True)
        self._init_weights()

    def forward(self, src, context_dict, device) -> torch.Tensor:
        # Embedding
        emb = self.Embedding(src, context_dict)

        # Pass through the LSTM
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            emb, context_dict["len"].cpu(), enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed_seq)  # LSTM outputs and hidden states
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)  # Unpack the sequence

        # Aggregate output (use hidden state of the last timestep for each sequence)
        last_hidden = h_n[-1]

        fc_output = self.fc(last_hidden, context_dict)

        return fc_output

    def _init_weights(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbeddingGeoTron(self.d_input, config)

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_input,
            nhead=config.nhead,
            activation="gelu",
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        encoder_norm = torch.nn.LayerNorm(self.d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
        )

        self.fc = FullyConnected(self.d_input, config, if_residual_layer=True)

        # init parameter
        self._init_weights()

    def forward(self, src, context_dict, device) -> Tensor:
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        fc_output = self.fc(out, context_dict)
        
        return fc_output

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def _init_weights(self):
        """Initiate parameters in the transformer model."""
        # initrange = 0.1
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

class FullyConnected(nn.Module):
    def __init__(self, d_input, config, if_residual_layer=True):
        super(FullyConnected, self).__init__()
        # the last fully connected layer
        #Inputs:
        fc_dim = d_input

        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            print("Embedding user using latlon of home location")
            self.emb_user = nn.Linear(2, config.user_emb_size)
            fc_dim = d_input + config.user_emb_size
            self.user_norm = nn.BatchNorm1d(config.user_emb_size)
        
        # time embedding
        self.time_encode = PeriodicEncoding(d_output=8)
        fc_dim += 8 + 1 # 8 for time_to_next encoding, 1 for time_to_next

        # Dropout and residual
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            # the residual
            self.fc_1 = nn.Linear(fc_dim, fc_dim)
            self.norm_1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout = nn.Dropout(p=config.fc_dropout)
        
        # location output
        self.predict_clusters = config.predict_clusters
        self.predict_intra_cluster = config.predict_intra_cluster
        if config.predict_clusters:
            self.fc_cluster = nn.Linear(fc_dim, config.total_cluster_num)
            if config.predict_intra_cluster:
                # append the cluster output as context
                # self.fc_intra_cluster = nn.Linear(fc_dim, config.max_intra_cluster_num)
                self.fc_intra_cluster = nn.Linear(fc_dim+config.total_cluster_num, config.max_intra_cluster_num)
            # print("Intra cluster ID not implemented yet")
        else:
            self.fc_loc = nn.Linear(fc_dim, config.total_loc_num)

    def forward(self, out, context_dict) -> Tensor:

        # with fc output
        if self.if_embed_user:
            # emb = self.emb_user(user)
            # out = torch.cat([out, emb], -1)
            user1 = context_dict["homelat"].view(-1, 1)
            user2 = context_dict["homelon"].view(-1, 1)
            user = torch.cat([user1, user2], -1)
            emb = self.emb_user(user)
            emb = self.user_norm(emb) # Add normalization for user embedding
            out = torch.cat([out, emb], -1)
        
        # time embedding
        time_to_next = context_dict["time_to_next"] # Should be normalized to [0,1]
        time_to_next_enc = self.time_encode(time_to_next)
        out = torch.cat([out, time_to_next_enc], -1) # Add time_to_next encoding as a feature
        out = torch.cat([out, time_to_next.view(-1, 1)], -1) # Add time_to_next as a feature
        
        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

        # location output
        if self.predict_clusters:
            out_cluster = self.fc_cluster(out)
            if self.predict_intra_cluster:
                out = torch.cat([out, out_cluster], -1)
                out_intra_cluster = self.fc_intra_cluster(out)
                return out_cluster, out_intra_cluster
            else:
                return out_cluster
        else:
            return self.fc_loc(out)


class PeriodicEncoding(torch.nn.Module):
    """
    Creates a periodic time embedding. Expect the input to be normalized so that [0,1] corresponds to a day.
    d_output should be an even number.
    """
    def __init__(self, d_output):
        super(PeriodicEncoding, self).__init__()
        self.d_output = d_output

    def forward(self, x):
        # Create a range of frequencies
        freqs = torch.arange(0, self.d_output // 2, dtype=torch.float32) * (2 * math.pi)
        # Match shape
        freqs = freqs.to(x.device).unsqueeze(0)
        x = x.unsqueeze(-1)
        # Compute sin and cos for each frequency
        encodings = torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], dim=-1)
        return encodings