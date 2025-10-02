import os
import torch

from PIL import Image
import argparse
from scene_decorator.scene_injection import register_mask_guided_scene_injection
from scene_decorator.scene_attention import register_scene_sharing_attention
from scene_decorator.pipeline import SceneDecoratorXLPipeline
from diffusers import ControlNetModel


def main(
    base_model_path: str,
    scene_encoder_path: str, 
    scene_adapter_path: str,
    controlnet_path: str,
    lora_path: str,
    output_path: str,
    #
    t_1: int,
    t_2: int,
    scene_scale: int,
    use_mask_guided_scene_injection: bool,
    use_scene_sharing_attention: bool,
    use_mask_guided_shared_kv: bool,
    STORY_NUMS=5,
    seed: int = 42,
    device: str = 'cuda',
    **kwargs,
):
    # get the controlnet if use
    if controlnet_path:
        controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16)
    else:
        controlnet = None
    # get the pipeline
    pipe = SceneDecoratorXLPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)
    pipe.enable_vae_tiling()
    # set lora for custom results if use
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=0.8)
    # enable for free-u if use
    # pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    # register mask-guided scene injection
    register_mask_guided_scene_injection(
        pipe, scene_encoder_path, scene_adapter_path, num_tokens=4,
        scene_scale=scene_scale, scene_injection_blocks=["up_blocks.0.attentions.1"],
        use_mask_guided_scene_injection=use_mask_guided_scene_injection,
    )
    # ****************************************Towards Scene-Oriented Story Generation***************************************

    local_scene_dir = 'test_data/local_scenes'
    local_story_dir = 'test_data/local_stories'
    local_subject_dir = 'test_data/local_subjects'
    #
    local_scene = sorted(os.listdir(local_scene_dir))[40]
    local_story = sorted(os.listdir(local_story_dir))[40]

    final_image_path = os.path.join(final_scene_dir, image_path)
    # for story prompts
    prompts = ['A scholar leans against an desk, smiling softly.'] \
                + ['A scholar stands near an table, smiling softly.'] \
                + ['The scholar discovers a hidden section, where time seems to stand still.'] \
                + ['The scholar reads a letter in an ancient tome, revealing a truth about his past.'] \
                + ['The scholar hears the books whisper, urging him to stay—or to leave forever.']
    negative_prompts = len(prompts) * ["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"]
    subjects = ["scholar", "scholar", "scholar", "scholar", "scholar"]
    # for scene image
    scene_image_path = "test_data/00090/scene02.png"
    scene_images = [Image.open(scene_image_path).resize((1024, 1.24))] * len(prompts)
    # additional for contrlnet
    canny_maps = [Image.open("examples/canny_map.jpg")] * len(prompts)


    # register the scene sharing attention
    register_scene_sharing_attention(pipe, prompts, subjects, use_scene_sharing_attention)


    


    # generate story images
    images = pipe.generate_story(
        scene_image=scene_images,
        prompt=prompts,
        negative_prompt=negative_prompts,
        num_inference_steps=50, # 30
        generator=torch.Generator(device).manual_seed(seed),
        # SceneDecorator parameters
        guidance_scale=15, # 15
        t_1=t_1,
        t_2=t_2,
        use_mask_guided_shared_kv=use_mask_guided_shared_kv,
        # ControlNet parameters
        control_image=canny_maps,
        controlnet_conditioning_scale=0.7,
    ).images


    # save results
    output_path = os.path.join(output_dir, local_scene_list[idx], image_path.split('.png')[0])
    os.makedirs(output_path, exist_ok=True)
    print(f"save path at: {output_path}")
    for index, image in enumerate(images):
        if prompts[index].endswith('.'):
            image.save(os.path.join(output_path, f"{prompts[index]}png"))
        else:
            image.save(os.path.join(output_path, f"{prompts[index]}.png"))


    # for loop
    for idx, local_story_path in enumerate(local_story_list):
        final_story_path = os.path.join(local_story_dir, local_story_path)
        final_subject_path = os.path.join(local_subject_dir, local_story_path)
        # 读取每张大图对应的所有故事
        with open(final_story_path) as files:
            prompts_list = [line.strip() for line in files.readlines()]
        # 读取每个故事的主体
        with open(final_subject_path) as files:
            subjects_list = [line.strip() for line in files.readlines()]
        # 读取大图对应的每一张小图
        final_scene_dir = os.path.join(local_scene_dir, local_scene_list[idx])
        # breakpoint()
        image_list = sorted(os.listdir(final_scene_dir))
        for jdx, image_path in enumerate(image_list):
            final_image_path = os.path.join(final_scene_dir, image_path)
            # breakpoint()
            images = [Image.open(final_image_path).resize((1024, 1024))] * STORY_NUMS
            prompts = prompts_list[STORY_NUMS * jdx: STORY_NUMS * (jdx + 1)]
            subjects = subjects_list[STORY_NUMS * jdx: STORY_NUMS * (jdx + 1)]
            negative_prompts = len(prompts) * ["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"]
            # modify attn
            breakpoint()

            # register the scene sharing attention
            register_scene_sharing_attention(pipe, prompts, subjects, use_scene_sharing_attention)


            # canny for contrlnet
            canny_map = Image.open("examples/canny_map.jpg")
            canny_maps = [canny_map] * 5


            # generate story images
            images = pipe.generate_story(
                scene_image=images,
                prompt=prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=50, # 30
                generator=torch.Generator(device).manual_seed(seed),
                # SceneDecorator parameters
                guidance_scale=15, # 15
                t_1=t_1,
                t_2=t_2,
                use_mask_guided_shared_kv=use_mask_guided_shared_kv,
                # ControlNet parameters
                control_image=canny_maps,
                controlnet_conditioning_scale=0.7,
            ).images


            # save results
            output_path = os.path.join(output_dir, local_scene_list[idx], image_path.split('.png')[0])
            os.makedirs(output_path, exist_ok=True)
            print(f"save path at: {output_path}")
            for index, image in enumerate(images):
                if prompts[index].endswith('.'):
                    image.save(os.path.join(output_path, f"{prompts[index]}png"))
                else:
                    image.save(os.path.join(output_path, f"{prompts[index]}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--scene_encoder_path", type=str, default="/data/lxy/sqj/code/SceneDecorator/sdxl_ckpts/image_encoder")
    parser.add_argument("--scene_adapter_path", type=str, default="/data/lxy/sqj/code/SceneDecorator/sdxl_ckpts/ip-adapter_sdxl.bin")
    parser.add_argument("--controlnet_path", type=str, default="/data/lxy/sqj/base_models/controlnet-canny-sdxl-1.0")
    parser.add_argument("--lora_path", type=str, default="lora_ckpts/SDXL_illustrious_mochimochi_artstyle_.safetensors")
    parser.add_argument("--output_path", type=str, default='/data/lxy/sqj/code/SceneDecorator/final_main_experiments')
    parser.add_argument("--t_1", type=int, default=0)
    parser.add_argument("--t_2", type=int, default=15)
    parser.add_argument("--scene_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default='cuda')
    # additional components
    parser.add_argument("--use_mask_guided_scene_injection", action='store_true',
                        help='whether to use the mask guided scene injection')
    parser.add_argument("--use_scene_sharing_attention", action='store_true', 
                        help='whether to use the mask guided shared kv')
    parser.add_argument("--use_mask_guided_shared_kv", action='store_true',
                        help='whether to use the mask with shared kv')
    #
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
