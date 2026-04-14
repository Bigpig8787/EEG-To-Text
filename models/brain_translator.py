"""Baseline BrainTranslator models from Wang and Ji, 2021."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024,
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super().__init__()
        self.pretrained = pretrained_layers
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        encoded = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        return F.relu(self.fc1(encoded))

    @torch.no_grad()
    def generate(self, input_embeddings_batch, input_masks_batch, input_masks_invert,
                 target_ids_batch_converted, **kwargs):
        encoded = self.addin_forward(input_embeddings_batch, input_masks_invert)
        return self.pretrained.generate(
            inputs_embeds=encoded,
            attention_mask=input_masks_batch[:, :encoded.shape[1]],
            labels=target_ids_batch_converted, return_dict=True, **kwargs)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch):
        encoded = self.addin_forward(input_embeddings_batch, input_masks_invert)
        return self.pretrained(inputs_embeds=encoded, attention_mask=input_masks_batch,
                               return_dict=True, labels=target_ids_batch)


class BrainTranslatorNaive(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, **kwargs):
        super().__init__()
        self.pretrained = pretrained_layers
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch):
        encoded = F.relu(self.fc1(input_embeddings_batch))
        return self.pretrained(inputs_embeds=encoded, attention_mask=input_masks_batch,
                               return_dict=True, labels=target_ids_batch)
