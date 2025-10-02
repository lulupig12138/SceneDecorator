import os

from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import torch

from scene_consistency.tools.utils import AttentionStore
from scene_consistency.tools.attention_processor import (
    AttnProcessor,
    MaskGuidedSceneInjectionProcessor,
    CNAttnProcessor,
    ImageProjModel
)


def register_mask_guided_scene_injection(
    pipe, scene_encoder_path, scene_adapter_path,
    num_tokens=4, scene_injection_blocks=["block"],
    use_mask_guided_scene_injection=True,
):
    # init attention score
    pipe.attention_store = AttentionStore()
    # image encoder
    pipe.clip_image_processor = CLIPImageProcessor()
    pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(scene_encoder_path).to(
        pipe.device, dtype=torch.float16
    )
    pipe.image_proj_model = ImageProjModel(
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        clip_embeddings_dim=pipe.image_encoder.config.projection_dim,
        clip_extra_context_tokens=num_tokens,
    ).to(pipe.device, dtype=torch.float16)
    # cross attention
    set_scene_adapter(pipe, scene_injection_blocks, num_tokens, use_mask_guided_scene_injection)
    load_scene_adapter(pipe, scene_adapter_path)


def set_scene_adapter(pipe, scene_injection_blocks, num_tokens=4, use_mask_guided_scene_injection=True):
    unet = pipe.unet
    attn_procs = {}
    for i, name in enumerate(unet.attn_processors.keys()):
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = f"mid_{i}"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = f"up_{i}"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = f"down_{i}"
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            selected = False
            for block_name in scene_injection_blocks:
                if block_name in name:
                    selected = True
                    break
            if selected:
                attn_procs[name] = MaskGuidedSceneInjectionProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=num_tokens,
                    scene_scale=1.0,
                    attention_store=pipe.attention_store,
                    place_in_unet=place_in_unet,
                    use_mask_guided_scene_injection=use_mask_guided_scene_injection,
                ).to(pipe.device, dtype=torch.float16)
            else:
                attn_procs[name] = MaskGuidedSceneInjectionProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=num_tokens,
                    scene_scale=1.0,
                    attention_store=pipe.attention_store,
                    place_in_unet=place_in_unet,
                    use_mask_guided_scene_injection=False # no use the mask guided sceneinjection
                ).to(pipe.device, dtype=torch.float16)
    unet.set_attn_processor(attn_procs)
    # additional for controlnet
    if pipe.controlnet:
        if isinstance(pipe.controlnet, MultiControlNetModel):
            for controlnet in pipe.controlnet.nets:
                controlnet.set_attn_processor(CNAttnProcessor(num_tokens=num_tokens))
        else:
            pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=num_tokens))


def load_scene_adapter(pipe, scene_adapter_path):
    if os.path.splitext(scene_adapter_path)[-1] == ".safetensors":
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(scene_adapter_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(scene_adapter_path, map_location="cpu")
    pipe.image_proj_model.load_state_dict(state_dict["image_proj"])
    ip_layers = torch.nn.ModuleList(pipe.unet.attn_processors.values())
    ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
