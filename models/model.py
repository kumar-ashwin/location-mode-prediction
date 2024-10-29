import torch.nn as nn
import numpy as np
import torch, math
from torch import Tensor

import torch.nn.functional as F

from models.embed import AllEmbedding



class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # Clip values to prevent instability before LayerNorm
        src2 = torch.clamp(src2, min=-5, max=5)

        # Log statistics only if NaN or Inf is detected
        self.log_tensor_if_nan_or_inf(src2, "Attention Output After Clipping (src2)")

        # Add & Norm (first layer normalization)
        src = src + self.dropout1(src2)
        
        # Clipping before applying LayerNorm
        src = torch.clamp(src, min=-5, max=5)
        self.log_tensor_if_nan_or_inf(src, "Before LayerNorm1")

        # Apply LayerNorm
        src = self.norm1(src)
        self.log_tensor_if_nan_or_inf(src, "After LayerNorm1")

        # Clipping after applying LayerNorm
        src = torch.clamp(src, min=-5, max=5)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Clip values to prevent instability before second LayerNorm
        src2 = torch.clamp(src2, min=-5, max=5)
        self.log_tensor_if_nan_or_inf(src2, "Feedforward Output After Clipping (src2)")

        # Add & Norm (second layer normalization)
        src = src + self.dropout2(src2)
        self.log_tensor_if_nan_or_inf(src, "After Add & Dropout2")

        # Clipping before applying LayerNorm
        src = torch.clamp(src, min=-5, max=5)

        # Apply LayerNorm
        src = self.norm2(src)
        self.log_tensor_if_nan_or_inf(src, "After LayerNorm2")

        # Clipping after applying LayerNorm
        src = torch.clamp(src, min=-5, max=5)

        return src

    def log_tensor_if_nan_or_inf(self, tensor, tensor_name):
        """Log tensor statistics only if NaN or Inf is detected."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"NaN or Inf detected in {tensor_name}.")
            print(f"{tensor_name} Values: ", tensor)
            print(f"Stats for {tensor_name}:")
            print(f"  Mean: {torch.mean(tensor).item():.4f}")
            print(f"  Std: {torch.std(tensor).item():.4f}")
            print(f"  Min: {torch.min(tensor).item():.4f}")
            print(f"  Max: {torch.max(tensor).item():.4f}")
            print(f"  Shape: {tensor.shape}")
            exit()



class TransEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config)

        # encoder
        # encoder_layer = torch.nn.TransformerEncoderLayer(
        encoder_layer = CustomTransformerEncoderLayer(
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
        self.if_embed_next_mode = config.if_embed_next_mode

        # init parameter
        self._init_weights()

    def forward(self, src, context_dict, device, next_mode=None) -> Tensor:
        emb = self.Embedding(src, context_dict)
        print("src:", src)
        print("context_dict:", context_dict)
        exit()
        # Embedding layer output check
        if torch.isnan(emb).any():
            print("NaN detected after Embedding")
            print("Embedding output:", emb)
            # return emb  # or exit()

        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # Transformer encoder output check
        if torch.isnan(out).any():
            print("NaN detected after Transformer Encoder")
            print("Transformer output:", out)
            # return out
        
        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        if torch.isnan(out).any():
            print("NaN detected after gather operation")
            print("Gather output:", out)
            # return out

        # Fully connected layer output check
        if self.if_embed_next_mode:
            fc_output = self.fc(out, context_dict["user"], mode_emb=self.Embedding.get_modeEmbedding(), next_mode=next_mode)
        else:
            fc_output = self.fc(out, context_dict["user"])
        
        if torch.isnan(fc_output[0]).any():
            print("NaN detected after Fully Connected Layer")
            print("Fully Connected output:", fc_output)
            return fc_output

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
    
    def debug_forward(self, src, context_dict, device, next_mode=None) -> Tensor:
        # To trace where nans appear in the model
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]



        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        #print something to see if there are nans
        print(out)

        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        if self.if_embed_next_mode:
            return self.fc(out, context_dict["user"], mode_emb=self.Embedding.get_modeEmbedding(), next_mode=next_mode), out
        else:
            return self.fc.debug_forward(out, context_dict["user"]), out


class FullyConnected(nn.Module):
    def __init__(self, d_input, config, if_residual_layer=True):
        super(FullyConnected, self).__init__()
        # the last fully connected layer
        fc_dim = d_input

        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, config.user_emb_size)
            fc_dim = d_input + config.user_emb_size

        self.if_embed_next_mode = config.if_embed_next_mode
        if self.if_embed_next_mode:
            # mode number -> user_embed_size (add)
            self.next_mode_fc = nn.Linear(config.base_emb_size, config.user_emb_size)

        self.if_loss_mode = config.if_loss_mode
        if self.if_loss_mode:
            self.fc_mode = nn.Linear(fc_dim, 8)
        self.fc_loc = nn.Linear(fc_dim, config.total_loc_num)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            # the residual
            self.fc_1 = nn.Linear(fc_dim, fc_dim)
            self.norm_1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout = nn.Dropout(p=config.fc_dropout)

    def forward(self, out, user, mode_emb=None, next_mode=None) -> Tensor:

        # with fc output
        if self.if_embed_user:
            emb = self.emb_user(user)

            if self.if_embed_next_mode:
                emb += self.next_mode_fc(mode_emb(next_mode))

            out = torch.cat([out, emb], -1)
        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

        if self.if_loss_mode:
            return self.fc_loc(out), self.fc_mode(out)
        else:
            return self.fc_loc(out)

    def debug_forward(self, out, user, mode_emb=None, next_mode=None) -> Tensor:
        # To trace where nans appear in the model

        # with fc output
        if self.if_embed_user:
            emb = self.emb_user(user)

            if self.if_embed_next_mode:
                emb += self.next_mode_fc(mode_emb(next_mode))

            out = torch.cat([out, emb], -1)
        
        print(out)
        
        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

        print('After residual:', out)
        
        if self.if_loss_mode:
            return self.fc_loc(out), self.fc_mode(out), out
        else:
            return self.fc_loc(out), out