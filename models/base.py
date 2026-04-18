from abc import ABC, abstractmethod
import torch.nn as nn


class BaseEEGModel(ABC, nn.Module):
    """Interface for all EEG-to-text models. Plug into Trainer/FinetunePipeline."""

    @property
    @abstractmethod
    def eeg_encoder(self) -> nn.Module:
        """Primary EEG encoder (view_encoders ModuleDict or single encoder)."""

    @property
    @abstractmethod
    def global_transformer(self):
        """Global cross-region transformer. None if not applicable."""

    @property
    @abstractmethod
    def language_model(self) -> nn.Module:
        """Language model backbone (BartForConditionalGeneration etc.)."""

    def freeze_language_model(self):
        for p in self.language_model.parameters():
            p.requires_grad = False

    def unfreeze_language_model(self):
        for p in self.language_model.parameters():
            p.requires_grad = True

    def get_encoder_params(self) -> list:
        """All EEG-side params: eeg_encoder + global_transformer + any fc layers."""
        params = list(self.eeg_encoder.parameters())
        if self.global_transformer is not None:
            params += list(self.global_transformer.parameters())
        return params

    def get_lm_params(self) -> list:
        return list(self.language_model.parameters())

    @abstractmethod
    def get_tuning_targets(self) -> dict:
        """Return {'encoder': [...params], 'lm': [...params]} for optimizer setup."""

    @abstractmethod
    def encode(self, inputs, masks=None, masks_inv=None):
        """EEG -> encoder hidden states ready for LM decoder cross-attention."""

    @abstractmethod
    def forward(self, inputs, masks, masks_inv, target_ids): ...

    @abstractmethod
    def generate(self, inputs, masks, masks_inv, **gen_kwargs): ...
