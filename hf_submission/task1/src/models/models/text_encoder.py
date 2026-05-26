import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


TOKEN_PATTERN = re.compile(r"<image>|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[^\sA-Za-z0-9]", re.UNICODE)


class BasicVQATokenizer:
    """
    Tokenizer tối giản cho prototype VQA classification.

    Không thay thế tokenizer LLM thật. Mục tiêu là smoke test cross-modal pipeline:
        question_text -> input_ids/attention_mask -> SimpleTextEncoder.
    """

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        lowercase: bool = True,
        keep_image_token: bool = True,
    ):
        self.lowercase = lowercase
        self.keep_image_token = keep_image_token

        if vocab is None:
            vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3,
                "<image>": 4,
            }
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    def __len__(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        text = text or ""
        tokens = TOKEN_PATTERN.findall(text)
        processed = []
        for token in tokens:
            if token == "<image>":
                if self.keep_image_token:
                    processed.append(token)
                continue
            processed.append(token.lower() if self.lowercase else token)
        return processed

    def encode(
        self,
        text: str,
        max_length: int = 64,
        add_special_tokens: bool = True,
    ) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids[:max_length]

    def batch_encode(
        self,
        texts: Sequence[str],
        max_length: int = 64,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(text, max_length=max_length, add_special_tokens=add_special_tokens) for text in texts]
        max_len = max((len(ids) for ids in encoded), default=1)
        max_len = min(max_len, max_length)

        input_ids = torch.full((len(texts), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)

        for i, ids in enumerate(encoded):
            ids = ids[:max_len]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        lowercase: bool = True,
        keep_image_token: bool = True,
    ) -> "BasicVQATokenizer":
        tokenizer = cls(lowercase=lowercase, keep_image_token=keep_image_token)
        counter = Counter()
        for text in texts:
            counter.update(tokenizer.tokenize(text))

        reserved = dict(tokenizer.vocab)
        vocab = dict(reserved)
        max_new_tokens = None
        if max_vocab_size is not None:
            max_new_tokens = max(max_vocab_size - len(reserved), 0)

        added = 0
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token in vocab:
                continue
            if max_new_tokens is not None and added >= max_new_tokens:
                break
            vocab[token] = len(vocab)
            added += 1

        return cls(vocab=vocab, lowercase=lowercase, keep_image_token=keep_image_token)


@dataclass
class SimpleTextEncoderConfig:
    vocab_size: int
    d_model: int = 768
    max_length: int = 64
    pad_token_id: int = 0
    num_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.10
    use_positional_embedding: bool = True

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size phải > 0")
        if self.d_model <= 0:
            raise ValueError("d_model phải > 0")
        if self.max_length <= 0:
            raise ValueError("max_length phải > 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model phải chia hết cho num_heads")


class SimpleTextEncoder(nn.Module):
    """
    Text encoder tối giản: Embedding + TransformerEncoder.

    Input:
        input_ids:      [B, Nt]
        attention_mask: [B, Nt], 1=valid, 0=pad

    Output:
        text_tokens: [B, Nt, D]
        pooled_text: [B, D]
    """

    def __init__(self, config: SimpleTextEncoderConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = None
        if config.use_positional_embedding:
            self.position_embedding = nn.Embedding(config.max_length, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids phải có shape [B, Nt], nhận {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_length:
            input_ids = input_ids[:, : self.config.max_length]
            seq_len = self.config.max_length
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.config.max_length]

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        attention_mask = attention_mask.to(device=input_ids.device)

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)

        key_padding_mask = attention_mask == 0
        text_tokens = self.encoder(x, src_key_padding_mask=key_padding_mask)
        text_tokens = self.output_norm(text_tokens)

        pooled_text = self.masked_mean_pool(text_tokens, attention_mask)

        if not return_dict:
            return text_tokens

        return {
            "text_tokens": text_tokens,
            "pooled_text": pooled_text,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def masked_mean_pool(tokens: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        summed = (tokens * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(eps)
        return summed / denom


def build_text_encoder_from_questions(
    questions: Sequence[str],
    d_model: int = 768,
    max_length: int = 64,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = 30000,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.10,
) -> Tuple[BasicVQATokenizer, SimpleTextEncoder]:
    tokenizer = BasicVQATokenizer.build_from_texts(
        questions,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )
    config = SimpleTextEncoderConfig(
        vocab_size=len(tokenizer),
        d_model=d_model,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    return tokenizer, SimpleTextEncoder(config)
