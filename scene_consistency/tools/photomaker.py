from typing import Union, Dict
from safetensors import safe_open

import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers import CLIPImageProcessor

from diffusers.utils import (
    _get_model_file,
)

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}


def load_photomaker_adapter(
    pipe,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    weight_name: str,
    subfolder: str = '',
    trigger_word: str = 'img',
    pm_version: str = 'v1',
    **kwargs,
):
    # Load the main state dict first.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        model_file = _get_model_file(
            pretrained_model_name_or_path_or_dict,
            weights_name=weight_name,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        if weight_name.endswith(".safetensors"):
            state_dict = {"id_encoder": {}, "lora_weights": {}}
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("id_encoder."):
                        state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                    elif key.startswith("lora_weights."):
                        state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    keys = list(state_dict.keys())
    if keys != ["id_encoder", "lora_weights"]:
        raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

    pipe.num_tokens = 2 if pm_version == 'v2' else 1
    pipe.pm_version = pm_version
    pipe.trigger_word = trigger_word
    # load finetuned CLIP image encoder and fuse module here if it has not been registered to the pipeline yet
    print(f"Loading PhotoMaker {pm_version} components [1] id_encoder from [{pretrained_model_name_or_path_or_dict}]...")
    pipe.id_image_processor = CLIPImageProcessor()
    if pm_version == "v1": # PhotoMaker v1 
        id_encoder = PhotoMakerIDEncoder()
    else:
        raise NotImplementedError(f"The PhotoMaker version [{pm_version}] does not support")

    id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)
    id_encoder = id_encoder.to(pipe.device, dtype=pipe.unet.dtype)    
    pipe.id_encoder = id_encoder

    # load lora into models
    print(f"Loading PhotoMaker {pm_version} components [2] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
    pipe.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

    # Add trigger word token
    if pipe.tokenizer is not None: 
        pipe.tokenizer.add_tokens([pipe.trigger_word], special_tokens=True)
    
    pipe.tokenizer_2.add_tokens([pipe.trigger_word], special_tokens=True)

    # fuse_lora
    pipe.fuse_lora()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        # breakpoint()
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        breakpoint()
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs], maybe something wrong!
        valid_id_mask = (
            torch.arange(batch_size * max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds


class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)
        # breakpoint()
        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return updated_prompt_embeds
