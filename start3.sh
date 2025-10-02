#! /bin/bash

CUDA_VISIBLE_DEVICES=3 python generate_story.py  \
                                                --t_1 0 \
                                                --t_2 25 \
                                                --output_path /data/lxy/sqj/code/SceneDecorator/generated_stories \
                                                --use_scene_sharing_attention \
                                                --use_mask_guided_scene_injection \
                                                --lora_path "lora_ckpts/SDXL_illustrious_mochimochi_artstyle_.safetensors"
                                                # --photomaker_path 'lora_ckpts/photomaker-v1.bin' \
                                                # --controlnet_path "/data/lxy/sqj/base_models/controlnet-canny-sdxl-1.0" \
                                                # --lora_path "lora_ckpts/SDXL_illustrious_mochimochi_artstyle_.safetensors" \
                                                # --use_extrapolable_noise_blending \
                                                