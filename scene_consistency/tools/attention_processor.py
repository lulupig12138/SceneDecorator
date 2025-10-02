# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
from typing import Optional

from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import Attention

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        
        return clip_extra_context_tokens
    

class MaskGuidedSceneInjectionProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=4, scene_scale=1.0,
                 use_mask_guided_scene_injection=True, attention_store=None, place_in_unet=None):
        super().__init__()

        self.use_mask_guided_scene_injection = use_mask_guided_scene_injection
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scene_scale = scene_scale
        self.num_tokens = num_tokens
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        record_attention=True,
        batch_cnt=[],
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = hidden_states.shape
            height = width = int(wh ** 0.5)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True
            # get encoder_hidden_states, scene_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, scene_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # record attention map
        if record_attention and self.attnstore is not None:
            self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads, batch_cnt)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.use_mask_guided_scene_injection:
            scene_key = self.to_k_ip(scene_hidden_states)
            scene_value = self.to_v_ip(scene_hidden_states)
            #
            scene_key = attn.head_to_batch_dim(scene_key)
            scene_value = attn.head_to_batch_dim(scene_value)

            # get mask
            attention_mask = self.attnstore.get_extended_attn_mask_instance(width, -1, batch_cnt=batch_cnt)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size // 2, -1).repeat(2,1).unsqueeze(-1).int().to(hidden_states.dtype)

            scene_attention_probs = attn.get_attention_scores(query, scene_key, None)
            self.attn_map = scene_attention_probs
            scene_hidden_states = torch.bmm(scene_attention_probs, scene_value)
            scene_hidden_states = attn.batch_to_head_dim(scene_hidden_states)

            # mask-guided scene injection
            if attention_mask is not None:
                hidden_states = hidden_states + (1 - attention_mask.to(scene_hidden_states.device)) * self.scene_scale * scene_hidden_states
            else:
                hidden_states = hidden_states + self.scene_scale * scene_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SceneSharingAttentionProcessor:
    def __init__(
            self, place_in_unet=None, attnstore=None,
            attn_kwargs=None, use_scene_sharing_attention=True,
        ):
        self.use_scene_sharing_attention = use_scene_sharing_attention
        self.t_range = attn_kwargs.get('t_range', [])
        self.extend_kv_unet_parts = attn_kwargs.get('extend_kv_unet_parts', ['down', 'mid', 'up'])
        self.place_in_unet = place_in_unet
        self.curr_unet_part = self.place_in_unet.split('_')[0]
        self.attnstore = attnstore

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        batch_cnt=[],
    ) -> torch.FloatTensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = hidden_states.shape
            height = width = int(wh ** 0.5)
        is_cross = encoder_hidden_states is not None

        use_scene_sharing_attention = self.use_scene_sharing_attention and (not is_cross) and \
                              any([self.attnstore.curr_iter >= x[0] and self.attnstore.curr_iter <= x[1] for x in self.t_range]) and \
                              self.curr_unet_part in self.extend_kv_unet_parts
        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()

        if use_scene_sharing_attention:
            # Pre-allocate the output tensor
            ex_out = torch.empty_like(query)
            for i in range(batch_size):
                start_idx = i * attn.heads
                end_idx = start_idx + attn.heads
                
                curr_q = query[start_idx:end_idx]
                if i < batch_size//2:
                    curr_k = key[:batch_size//2]
                    curr_v = value[:batch_size//2]
                else:
                    curr_k = key[batch_size//2:]
                    curr_v = value[batch_size//2:]

                # get the attention map of background, shape: [B, h*w]
                attention_mask = self.attnstore.get_extended_attn_mask_instance(width, i%(batch_size//2), mask_background=True, batch_cnt=batch_cnt)
                if attention_mask is not None:
                    curr_k = curr_k.flatten(0,1)[attention_mask].unsqueeze(0)
                    curr_v = curr_v.flatten(0,1)[attention_mask].unsqueeze(0)
                else:
                    curr_k = curr_k.flatten(0,1).unsqueeze(0)
                    curr_v = curr_v.flatten(0,1).unsqueeze(0)
                curr_k = attn.head_to_batch_dim(curr_k).contiguous()
                curr_v = attn.head_to_batch_dim(curr_v).contiguous()

                hidden_states = xformers.ops.memory_efficient_attention(
                    curr_q, curr_k, curr_v, scale=attn.scale
                )
                ex_out[start_idx:end_idx] = hidden_states

            hidden_states = ex_out
        else:
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()
            # attn_masks needs to be of shape [batch_size, query_tokens, key_tokens]
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, scale=attn.scale
            )

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# for naive processor
class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


## for controlnet
class CNAttnProcessor:
    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
