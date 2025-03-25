# This file is copied from https://github.com/CellProfiling/SubCellPortable/blob/main/vit_model.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import (
    BaseModelOutputWithPooling,
    ViTAttention,
    ViTEmbeddings,
    ViTIntermediate,
    ViTOutput,
    ViTPatchEmbeddings,
    ViTPooler,
    ViTPreTrainedModel,
    ViTSdpaAttention,
)
from torch import nn, Tensor


@dataclass
class ViTPoolModelOutput:
    attentions: Tuple[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None

    pool_op: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None

    probabilities: torch.FloatTensor = None


class GatedAttentionPooler(nn.Module):
    def __init__(
        self, dim: int, int_dim: int = 512, num_heads: int = 1, out_dim: int = None
    ):
        super().__init__()

        self.num_heads = num_heads

        self.attention_v = nn.Sequential(nn.Linear(dim, int_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, int_dim), nn.GELU())
        self.attention = nn.Linear(int_dim, num_heads)

        self.softmax = nn.Softmax(dim=-1)

        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)

        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)

        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)

        x = self.out_proj(x)
        return x, attn


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, sdpa_attn=False) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = (
            ViTAttention(config) if not sdpa_attn else ViTSdpaAttention(config)
        )
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        # self.layer = nn.ModuleList(
        #     [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        # )
        layer = []
        for i in range(config.num_hidden_layers):
            if i == config.num_hidden_layers - 1:
                layer.append(ViTLayer(config, sdpa_attn=False))
            else:
                layer.append(ViTLayer(config, sdpa_attn=True))
        self.layer = nn.ModuleList(layer)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTInferenceModel(ViTPreTrainedModel):
    def __init__(
        self,
        config: ViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTPoolClassifier(nn.Module):
    def __init__(self, config: Dict):
        super(ViTPoolClassifier, self).__init__()

        self.vit_config = ViTConfig(**config["vit_model"])

        self.encoder = ViTInferenceModel(self.vit_config, add_pooling_layer=False)

        pool_config = config.get("pool_model")
        self.pool_model = GatedAttentionPooler(**pool_config) if pool_config else None

        self.out_dim = (
            self.pool_model.out_dim if self.pool_model else self.vit_config.hidden_size
        )

        self.num_classes = config["num_classes"]
        self.sigmoid = nn.Sigmoid()

    def make_classifier(self):
        return nn.Sequential(
            nn.Linear(self.out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

    def load_model_dict(
        self,
        encoder_path: str,
        classifier_paths: Union[str, List[str]],
        device="cpu",
    ):
        checkpoint = torch.load(encoder_path, map_location=device)
        encoder_ckpt = {
            k[len("encoder.") :]: v for k, v in checkpoint.items() if "encoder." in k
        }

        status = self.encoder.load_state_dict(encoder_ckpt)
        print(f"Encoder status: {status}")

        pool_ckpt = {
            k.replace("pool_model.", ""): v
            for k, v in checkpoint.items()
            if "pool_model." in k
        }
        pool_ckpt = {k.replace("1.", "0."): v for k, v in pool_ckpt.items()}
        if pool_ckpt and self.pool_model:
            status = self.pool_model.load_state_dict(pool_ckpt)
            print(f"Pool model status: {status}")
        else:
            print("No pool model found in checkpoint")

        if isinstance(classifier_paths, str):
            classifier_paths = [classifier_paths]

        self.classifiers = nn.ModuleList(
            [self.make_classifier() for _ in range(len(classifier_paths))]
        )
        for i, classifier_path in enumerate(classifier_paths):
            classifier_ckpt = torch.load(classifier_path, map_location=device)
            classifier_ckpt = {
                k.replace("3.", "2."): v for k, v in classifier_ckpt.items()
            }
            classifier_ckpt = {
                k.replace("6.", "4."): v for k, v in classifier_ckpt.items()
            }
            status = self.classifiers[i].load_state_dict(classifier_ckpt)
            print(f"Classifier {i+1} status: {status}")

    def forward(self, x: torch.Tensor) -> ViTPoolModelOutput:
        b, c, h, w = x.shape
        outputs = self.encoder(x, output_attentions=True, interpolate_pos_encoding=True)

        if self.pool_model:
            pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        else:
            pool_op = torch.mean(outputs.last_hidden_state, dim=1)
            pool_attn = None

        probs = torch.stack(
            [self.sigmoid(classifier(pool_op)) for classifier in self.classifiers],
            dim=1,
        )
        probs = torch.mean(probs, dim=1)

        h_feat = h // self.vit_config.patch_size
        w_feat = w // self.vit_config.patch_size

        attentions = outputs.attentions[-1][:, :, 0, 1:].reshape(
            b, self.vit_config.num_attention_heads, h_feat, w_feat
        )

        pool_attn = pool_attn[:, :, 1:].reshape(
            b, self.pool_model.num_heads, h_feat, w_feat
        )

        return ViTPoolModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            attentions=attentions,
            pool_op=pool_op,
            pool_attn=pool_attn,
            probabilities=probs,
        )
