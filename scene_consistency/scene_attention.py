import torch
from scene_consistency.tools.attention_processor import SceneSharingAttentionProcessor


def register_scene_sharing_attention(pipe, prompts=None, subjects=None, use_scene_sharing_attention=True):
    # register attention store
    set_attention_store(pipe, prompts, subjects)
    # set the scene sharing attention
    attn_kwargs = {'extend_kv_unet_parts': ['up'], 't_range': [(1, 50)]}
    attn_procs = {}
    for i, name in enumerate(pipe.unet.attn_processors.keys()):
        is_self_attn = (i % 2 == 0)
        #
        if name.startswith("mid_block"):
            place_in_unet = f"mid_{i}"
        elif name.startswith("up_blocks"):
            place_in_unet = f"up_{i}"
        elif name.startswith("down_blocks"):
            place_in_unet = f"down_{i}"
        else:
            continue
        #
        if is_self_attn:
            attn_procs[name] = SceneSharingAttentionProcessor(place_in_unet, pipe.attention_store, attn_kwargs, use_scene_sharing_attention=use_scene_sharing_attention)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)


def set_attention_store(pipe, prompts=None, subjects=None):
    assert len(subjects) == len(prompts)
    batch_size = len(subjects)
    token_indices = create_token_indices(batch_size, pipe.tokenizer, prompts, subjects)
    anchor_mappings = create_anchor_mapping(batch_size, anchor_indices=list(range(2)))
    default_attention_store_kwargs = {
        'batch_size': batch_size,
        'token_indices': token_indices,
        'extended_mapping': anchor_mappings,
    }
    pipe.attention_store.init_set(default_attention_store_kwargs)


def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping


def create_token_indices(batch_size, tokenizer, prompts, subjects):
    subject_token_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in subjects]
    prompt_token_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']
    #
    token_indices = torch.full((1, batch_size), -1, dtype=torch.int64)
    for i, subject_token_id in enumerate(subject_token_ids):
        token_loc = torch.where(prompt_token_ids[i] == subject_token_id)[0]
        token_indices[0, i] = token_loc[0]

    return token_indices
