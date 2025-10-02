from diffusers import StableDiffusionXLControlNetPipeline
import torch
from typing import Optional, Tuple, Union, List

from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
import inspect
import PIL
import PIL
import gc
from diffusers.loaders import (
    TextualInversionLoaderMixin,
)
from scene_consistency.noise_blending import generate_combinations


class SceneDecoratorPipeline(
    StableDiffusionXLControlNetPipeline
):  
    @torch.no_grad()
    def get_scene_embeds(self, scene_images=None, clip_image_embeds=None, content_prompt_embeds=None):
        if scene_images is not None:
            if isinstance(scene_images, PIL.Image.Image):
                scene_images = [scene_images]
            clip_image = self.clip_image_processor(images=scene_images, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))

        return image_prompt_embeds, uncond_image_prompt_embeds

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        ### Added args
        num_id_images: int = 1,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                # breakpoint()
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids 
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                
                prompt_embeds_all = []
                class_tokens_mask_all = []
                pooled_prompt_embeds_all = []
                # Find out the corresponding class word token based on the newly added trigger word token
                for sub_text_input_ids in text_input_ids.tolist():
                    clean_index = 0
                    clean_input_ids = []
                    class_token_index = []
                    for i, token_id in enumerate(sub_text_input_ids):
                        if token_id == image_token_id:
                            class_token_index.append(clean_index - 1)
                        else:
                            clean_input_ids.append(token_id)
                            clean_index += 1

                    if len(class_token_index) != 1:
                        raise ValueError(
                            f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                                Trigger word: {self.trigger_word}, Prompt: {prompt}."
                        )
                    # breakpoint()
                    class_token_index = class_token_index[0]

                    # Expand the class word token and corresponding mask
                    class_token = clean_input_ids[class_token_index]
                    clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images * self.num_tokens + \
                        clean_input_ids[class_token_index+1:]                
                        
                    # Truncation or padding
                    max_len = tokenizer.model_max_length
                    if len(clean_input_ids) > max_len:
                        clean_input_ids = clean_input_ids[:max_len]
                    else:
                        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                            max_len - len(clean_input_ids)
                        )

                    class_tokens_mask = [True if class_token_index <= i < class_token_index+(num_id_images * self.num_tokens) else False \
                        for i in range(len(clean_input_ids))]

                    clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                    class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)
                    class_tokens_mask_all.append(class_tokens_mask)


                    prompt_embeds = text_encoder(clean_input_ids.to(device), output_hidden_states=True)

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    pooled_prompt_embeds_all.append(pooled_prompt_embeds)
                    if clip_skip is None:
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                    else:
                        # "2" because SDXL always indexes from the penultimate layer.
                        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                    prompt_embeds_all.append(prompt_embeds)
                #
                pooled_prompt_embeds_all = torch.cat(pooled_prompt_embeds_all)
                class_tokens_mask_all = torch.cat(class_tokens_mask_all)    
                #
                prompt_embeds_list.append(torch.cat(prompt_embeds_all))

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        class_tokens_mask_all = class_tokens_mask_all.to(device=device) # TODO: ignoring two-prompt case
        
        
        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        # breakpoint()
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds_all, negative_pooled_prompt_embeds, class_tokens_mask_all

    def set_scene_scale(self, scene_scale):
        for attn_processor in self.unet.attn_processors.values():
            attn_processor.scene_scale = scene_scale

    @torch.no_grad()
    def generate_story(
        self,
        scene_image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 5.0,
        scene_scale: float = 1.0,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        # Additional parameters for ControlNet
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image=None,
        # additional parameters for PhotoMaker
        input_id_images: List[PIL.Image.Image] = None,
        start_merge_step: int = 10,
        id_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # set scene_scale
        self.set_scene_scale(scene_scale)

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]
        

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale


        if input_id_images and not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        

        device = self._execution_device


        # ---------------------- additional for controlnet----------------------------
        if self.controlnet:
            global_pool_conditions = (
                self.controlnet.config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions
            #
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=1,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = control_image.shape[-2:]
        # --------------------------------------------------

        # breakpoint()

        # -----------------------Get the embeddings-----------------------
        if hasattr(self, 'id_encoder'):
            (
                prompt_embeds, 
                _,
                pooled_prompt_embeds,
                _,
                class_tokens_mask,
            ) = self.encode_prompt_with_trigger_word(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=device,
                num_id_images=len(input_id_images),
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            # Encode input prompt without the trigger word for delayed conditioning encode, remove trigger word token, then decode
            tokens_text_only = [self.tokenizer.encode(pro, add_special_tokens=False) for pro in prompt]
            trigger_word_token = self.tokenizer.convert_tokens_to_ids(self.trigger_word)

            [token_text_only.remove(trigger_word_token) for token_text_only in tokens_text_only]
            prompt_text_only = [self.tokenizer.decode(token_text_only, add_special_tokens=False) for token_text_only in tokens_text_only]
            (
                prompt_embeds_text_only,
                negative_prompt_embeds,
                pooled_prompt_embeds_text_only, # TODO: replace the pooled_prompt_embeds with text only prompt
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt_text_only,
                negative_prompt=negative_prompt,
                device=device,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            # additional Prepare the input ID images
            dtype = next(self.id_encoder.parameters()).dtype
            if not isinstance(input_id_images[0], torch.Tensor):
                id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values
            id_pixel_values = id_pixel_values.unsqueeze(0).repeat(prompt_embeds.shape[0], 1, 1, 1, 1).to(device=device, dtype=dtype) # TODO: multiple prompts
            # Get the update text embedding with the stacked ID embedding
            if id_embeds is not None:
                id_embeds = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
                prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds)
            else:
                prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
        else:
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )
        # -------------------------------------------------------------------------



        # -------------------------additional for scene adapter--------------------
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_scene_embeds(scene_image)
        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        if hasattr(self, 'id_encoder'):
            prompt_embeds_text_only = torch.cat([prompt_embeds_text_only, image_prompt_embeds], dim=1)
        # ----------------------------------------------------------------------


        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        
        # ------------------------additional for controlnet----------------------------
        if self.controlnet:
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0])
        # Prepare added time ids & embeddings
        if isinstance(control_image, list):
            original_size = original_size or control_image[0].shape[-2:]
        else:
            original_size = original_size or control_image.shape[-2:]
        target_size = target_size or (height, width)
        # ----------------------------------------------------------------------------
        
        
        # Prepare added time ids & embeddings
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
        #
        negative_prompt_embeds, prompt_embeds = negative_prompt_embeds.to(device), prompt_embeds.to(device)
        negative_pooled_prompt_embeds, pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device), pooled_prompt_embeds.to(device)
        negative_add_time_ids, add_time_ids = negative_add_time_ids.to(device).repeat(batch_size, 1), add_time_ids.to(device).repeat(batch_size, 1)

        # Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        self._num_timesteps = len(timesteps)



        # for denoising
        use_extrapolable_noise_blending = self.extrapolable_noise_blending_kwargs['use_extrapolable_noise_blending']
        t_1 = self.extrapolable_noise_blending_kwargs['t_1']
        t_2 = self.extrapolable_noise_blending_kwargs['t_2']
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.attention_store.curr_iter = i
                # enumerate the patch
                if use_extrapolable_noise_blending:
                    if i >= t_1 and i <= t_2:
                        K = batch_size - 1
                        # patch_list = generate_combinations(list(range(batch_size)), partition=True, K=K)
                        patch_list = generate_combinations(list(range(batch_size)))
                    else:
                        K = 1
                        patch_list = [[i] for i in range(batch_size)]
                else:
                    K = 1
                    patch_list = [[i for i in range(batch_size)]]
                # breakpoint()
                noise_pred_all = torch.zeros_like(latents)
                for idx, patch in enumerate(patch_list):
                    # breakpoint()
                    # patch = [index + batch_size for index in uncond_patch]


                    # -----------------------------------additional controlnet--------------------------------------------
                    if self.controlnet:
                        if guess_mode and self.do_classifier_free_guidance:
                            control_model_input_patch = latents[patch]
                            control_model_input_patch = self.scheduler.scale_model_input(control_model_input_patch, t)
                            controlnet_prompt_embeds = prompt_embeds[patch]
                            #
                            controlnet_added_cond_kwargs = {
                                "text_embeds": pooled_prompt_embeds[patch],
                                "time_ids": add_time_ids[patch],
                            }
                            control_image_input = control_image[patch]
                        else:
                            control_model_input_patch = torch.cat([latents[patch]] * 2) if self.do_classifier_free_guidance else latents[patch]
                            control_model_input_patch = self.scheduler.scale_model_input(control_model_input_patch, t)
                            controlnet_prompt_embeds = torch.cat([negative_prompt_embeds[patch], prompt_embeds[patch]], dim=0) \
                                                        if self.do_classifier_free_guidance else prompt_embeds[patch]
                            #
                            controlnet_added_cond_kwargs = {
                                "text_embeds": torch.cat([negative_pooled_prompt_embeds[patch], pooled_prompt_embeds[patch]], dim=0)
                                                if self.do_classifier_free_guidance else pooled_prompt_embeds[patch],
                                "time_ids": torch.cat([negative_add_time_ids[patch], add_time_ids[patch]], dim=0)
                                                if self.do_classifier_free_guidance else add_time_ids[patch]
                            }
                            control_image_input = torch.cat([control_image[[index + batch_size for index in patch]], control_image[patch]], dim=0) \
                                                        if self.do_classifier_free_guidance else control_image[patch]
                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]
                        #
                        down_block_res_samples_patch, mid_block_res_sample_patch = self.controlnet(
                            control_model_input_patch,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=control_image_input,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            added_cond_kwargs=controlnet_added_cond_kwargs,
                            return_dict=False,
                        )
                        if guess_mode and self.do_classifier_free_guidance:
                            # Infered ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples_patch = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples_patch]
                            mid_block_res_sample_patch = torch.cat([torch.zeros_like(mid_block_res_sample_patch), mid_block_res_sample_patch])
                    else:
                        down_block_res_samples_patch = None
                        mid_block_res_sample_patch = None
                    # ---------------------------------------------------------------------------------------------------------



                    # ----------------------------------additional photomaker----------------------------------
                    if hasattr(self, 'id_encoder') and i <= start_merge_step: # photomaker
                        prompt_embeds_patch = torch.cat([
                            negative_prompt_embeds[patch], prompt_embeds_text_only[patch]
                        ], dim=0) if self.do_classifier_free_guidance else prompt_embeds_text_only[patch]
                        #
                        added_cond_kwargs_patch = {
                            'text_embeds': torch.cat([negative_pooled_prompt_embeds[patch], pooled_prompt_embeds_text_only[patch]], dim=0) 
                                            if self.do_classifier_free_guidance else pooled_prompt_embeds_text_only[patch],
                            'time_ids': torch.cat([negative_add_time_ids[patch], add_time_ids[patch]], dim=0) 
                                            if self.do_classifier_free_guidance else add_time_ids[patch]
                        }
                    else: # no photomaker
                        prompt_embeds_patch = torch.cat([
                            negative_prompt_embeds[patch], prompt_embeds[patch]
                        ], dim=0) if self.do_classifier_free_guidance else prompt_embeds[patch]
                        #
                        added_cond_kwargs_patch = {
                            "text_embeds": torch.cat([negative_pooled_prompt_embeds[patch], pooled_prompt_embeds[patch]], dim=0)
                                            if self.do_classifier_free_guidance else pooled_prompt_embeds[patch],
                            "time_ids": torch.cat([negative_add_time_ids[patch], add_time_ids[patch]], dim=0)
                                            if self.do_classifier_free_guidance else add_time_ids[patch]
                        }


                    # -----------------------------------------------------------------------------------------
                    

                    latent_model_input_patch = torch.cat([latents[patch]] * 2) if self.do_classifier_free_guidance else latents[patch]
                    latent_model_input_patch = self.scheduler.scale_model_input(latent_model_input_patch, t)
                    # call unet
                    noise_pred = self.unet(
                        latent_model_input_patch,
                        t,
                        encoder_hidden_states=prompt_embeds_patch,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs={'batch_cnt': patch},
                        added_cond_kwargs=added_cond_kwargs_patch,
                        # for controlnet
                        down_block_additional_residuals=down_block_res_samples_patch,
                        mid_block_additional_residual=mid_block_res_sample_patch,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                    # merge latents
                    noise_pred_all[patch] += noise_pred
                # post prosessing
                if use_extrapolable_noise_blending and i >= t_1 and i <= t_2:
                    noise_pred = noise_pred_all / K    
                else:
                    noise_pred = noise_pred_all
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]                
                del noise_pred_all
               

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                # Update attention store mask
                self.attention_store.aggregate_last_steps_attention()


        # additional for clean cache
        self.attention_store.reset()
        gc.collect()
        torch.cuda.empty_cache()


        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            img_list = []
            for i in range(batch_size):
                image= self.vae.decode(latents[i].unsqueeze(0) / self.vae.config.scaling_factor, return_dict=False)[0].cpu()
                img_list.append(image)
            image = torch.cat(img_list, dim=0)

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
