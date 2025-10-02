
CUDA_VISIBLE_DEVICES=1 python infer_style_with_controlnet.py  --t_1 0 \
                                            --t_2 25 \
                                            --output_path /data/lxy/sqj/code/SceneDecorator/generated_stories \
                                            --use_mask_guided_shared_kv \
                                            --enable_mask_with_shared_kv \
                                            --use_mask_guided_scene_injection \
                                            --enable_mask_with_injection \