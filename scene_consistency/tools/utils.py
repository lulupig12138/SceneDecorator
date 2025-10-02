import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from skimage import filters
import matplotlib.pyplot as plt
import textwrap


class AttentionStore:
    def __init__(self):
        pass

    def init_set(self, attention_store_kwargs):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.batch_size = attention_store_kwargs.get('batch_size', 1)
        self.token_indices = attention_store_kwargs['token_indices']
        self.extended_mapping = attention_store_kwargs.get('extended_mapping', torch.ones(self.batch_size, self.batch_size).bool())
        self.mask_dropout = attention_store_kwargs.get('mask_dropout', 0.0)
        self.attn_res = attention_store_kwargs.get('attn_res', (32,32))

        self.curr_iter = 0
        self.ALL_RES = [32, 64]
        self.step_store = [defaultdict(list) for _ in range(self.batch_size)]
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.last_mask_dropout = {res: None for res in self.ALL_RES}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads: int, batch_cnt=[]):
        if is_cross and attn.shape[1] == np.prod(self.attn_res):
            guidance_attention = attn[attn.size(0)//2:]
            batched_guidance_attention = guidance_attention.reshape([guidance_attention.shape[0]//attn_heads, attn_heads, *guidance_attention.shape[1:]])
            batched_guidance_attention = batched_guidance_attention.mean(dim=1)
            #
            for idx, cnt in enumerate(batch_cnt):
                self.step_store[cnt][place_in_unet].append(batched_guidance_attention[idx])

    def reset(self):
        self.step_store = [defaultdict(list) for _ in range(self.batch_size)]
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.last_mask_dropout = {res: None for res in self.ALL_RES}
        #
        torch.cuda.empty_cache()

    def aggregate_last_steps_attention(self) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        attention_maps = []
        for step_store in self.step_store:
            attention_map = torch.cat([torch.stack(x[-20:]) for x in step_store.values()]).mean(dim=0)    
            attention_maps.append(attention_map.unsqueeze(0))
        attention_maps = torch.cat(attention_maps)
        bsz, wh, _ = attention_maps.shape

        # Create attention maps for each concept token, for each batch item
        agg_attn_maps = []
        for i in range(bsz):
            curr_prompt_indices = []
            for concept_token_indices in self.token_indices:
                if concept_token_indices[i] != -1:
                    curr_prompt_indices.append(attention_maps[i, :, concept_token_indices[i]].view(*self.attn_res))

            agg_attn_maps.append(torch.stack(curr_prompt_indices))

        # Upsample the attention maps to the target resolution
        # and create the attention masks, unifying masks across the different concepts
        for tgt_size in self.ALL_RES:
            tgt_agg_attn_maps = [F.interpolate(x.to(torch.float32).unsqueeze(1), size=tgt_size, mode='bilinear').squeeze(1) for x in agg_attn_maps]

            attn_masks = []
            for batch_item_map in tgt_agg_attn_maps:
                concept_attn_masks = []
                for concept_maps in batch_item_map:
                    concept_attn_masks.append(torch.from_numpy(self.attn_map_to_binary(concept_maps, 1.)).to(attention_maps.device).bool().view(-1))
                concept_attn_masks = torch.stack(concept_attn_masks, dim=0).max(dim=0).values
                attn_masks.append(concept_attn_masks)

            attn_masks = torch.stack(attn_masks)
            self.last_mask[tgt_size] = attn_masks.clone()

            # Add mask dropout
            if self.curr_iter < 1000:
                rand_mask = (torch.rand_like(attn_masks.float()) < self.mask_dropout)
                attn_masks[rand_mask] = False

            self.last_mask_dropout[tgt_size] = attn_masks.clone()

    # get the mask
    def get_extended_attn_mask_instance(self, width, i, mask_background=False, batch_cnt=[]):
        attn_mask = self.last_mask_dropout[width]
        if attn_mask is None:
            return None
        attn_mask = attn_mask[batch_cnt]
        n_patches = width**2     
        output_attn_mask = torch.zeros((attn_mask.shape[0] * attn_mask.shape[1],), device=attn_mask.device, dtype=torch.bool)
        for j in range(attn_mask.shape[0]):
            if i==j:
                output_attn_mask[j*n_patches:(j+1)*n_patches] = 1
            else:
                if self.extended_mapping[i,j]:
                    if mask_background: # get mask for background
                        output_attn_mask[j*n_patches:(j+1)*n_patches] = ~attn_mask[j].unsqueeze(0)
                    else: # get mask for foreground
                        output_attn_mask[j*n_patches:(j+1)*n_patches] = attn_mask[j].unsqueeze(0) #.expand(n_patches, -1)

        return output_attn_mask

    def attn_map_to_binary(self, attention_map, scaler=1.):
        attention_map_np = attention_map.cpu().numpy()
        threshold_value = filters.threshold_otsu(attention_map_np) * scaler
        binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

        return binary_mask


def show_results(images, prompts, font_size=14, wrap_width=55, save_path=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(9*n, 10))
    if n == 1:
        axes = [axes]
    for ax, img, text in zip(axes, images, prompts):
        ax.imshow(np.array(img))
        ax.set_xticks([]); ax.set_yticks([])

        wrapped = "\n".join(textwrap.wrap(text, width=wrap_width))
        ax.text(0.5, -0.05, wrapped,
                ha="center", va="top",
                fontsize=font_size, transform=ax.transAxes)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
