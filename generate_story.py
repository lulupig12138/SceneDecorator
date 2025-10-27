import os
import torch

from PIL import Image
import argparse
import cv2

from scene_consistency.scene_injection import register_mask_guided_scene_injection
from scene_consistency.scene_attention import register_scene_sharing_attention
from scene_consistency.noise_blending import register_extrapolable_noise_blending
from scene_consistency.tools.photomaker import load_photomaker_adapter
from scene_consistency.tools.pipeline import SceneDecoratorPipeline
from scene_consistency.tools.utils import show_results

from diffusers import ControlNetModel
from diffusers.utils import load_image


def main(
    base_model_path: str,
    scene_encoder_path: str, 
    scene_adapter_path: str,
    controlnet_path: str,
    photomaker_path: str,
    lora_path: str,
    output_path: str,
    #
    scene_scale: float = 1.0,
    controlnet_conditioning_scale: float = 0.8,
    num_inference_steps: int = 50,
    seed: int = 42,
    device: str = 'cuda',
    #
    use_mask_guided_scene_injection: bool = True,
    use_scene_sharing_attention: bool = True,
    use_extrapolable_noise_blending: bool = False,
    t_1: int = 0,
    t_2: int = 15,
    **kwargs,
):
    # get the controlnet if use
    if controlnet_path:
        controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16)
    else:
        controlnet = None
    # get the pipeline
    pipe = SceneDecoratorPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)
    # set PhotoMaker if use
    if photomaker_path:
        load_photomaker_adapter(
            pipe,
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"
        )
    # set lora if use
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=0.8)

    # enable for free-u if use
    # pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    # register mask-guided scene injection
    register_mask_guided_scene_injection(
        pipe, scene_encoder_path, scene_adapter_path, num_tokens=4,
        scene_injection_blocks=["up_blocks.0.attentions.1"],
        use_mask_guided_scene_injection=use_mask_guided_scene_injection,
    )
    # register extrapolable noise blending
    register_extrapolable_noise_blending(
        pipe, t_1, t_2, use_extrapolable_noise_blending=use_extrapolable_noise_blending
    )


    # ****************************************Towards Scene-Oriented Story Generation***************************************

    # for story prompts
    # prompts = ['A scholar img leans against an desk, smiling softly.'] \
    #             + ['A scholar img stands near an table, smiling softly.'] \
    #             + ['The scholar img discovers a hidden section, where time seems to stand still.'] \
    #             + ['The scholar img reads a letter in an ancient tome, revealing a truth about his past.'] \
    #             + ['The scholar img hears the books whisper, urging him to stay—or to leave forever.']
    # negative_prompts = len(prompts) * ["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"]
    # subjects = ["scholar", "scholar", "scholar", "scholar", "scholar"]
    # # for scene image
    # scene_image_path = "test_data/local_scenes/00090/scene02.png"
    # scene_images = [Image.open(scene_image_path).resize((1024, 1024))] * len(prompts)




    prompts = ['A woman in a straw hat stands in a meadow of wildflowers, looking at the distant hills glowing under the summer sun.'] \
                + ['A child in an orange sweater stands in an open autumn meadow, surrounded by golden grass.'] \
                + ['A cat runs across untouched snow under a clear blue sky, leaving playful trails behind, bright winter sunlight, cinematic, highly detailed.']
    
    negative_prompts = len(prompts) * ["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"]
    subjects = ["girl", "woman", "child", "cat"]
    # for scene image
    scene_image_path2 = "examples/involved_scenes/case2/summer.png"
    scene_image_path3 = "examples/involved_scenes/case2/autumn.png"
    scene_image_path4 = "examples/involved_scenes/case2/winter.png"
    scene_images = [Image.open(scene_image_path2).resize((1024, 1024)),
                    Image.open(scene_image_path3).resize((1024, 1024)),
                    Image.open(scene_image_path4).resize((1024, 1024))]




    # prompts = ['sunrise, misty green hills; a lone fox standing alert in golden light; photorealistic.',] \
    #             + ['sunrise, misty green hills; young botanist collecting a rare dawn-bloom; golden light; photorealistic'] \
    #             + ['sunset meadow; a majestic deer resting in the soft glow; photorealistic.']


    # negative_prompts = len(prompts) * ["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"]
    # subjects = ["fox", "botanist", "deer"]
    # # for scene image
    # scene_image_path1 = "examples/cropped_image_1.png"
    # scene_image_path2 = "examples/cropped_image_2.png"
    # scene_image_path3 = "examples/cropped_image_3.png"
    # scene_images = [Image.open(scene_image_path1).resize((1024, 1024)), Image.open(scene_image_path2).resize((1024, 1024)), Image.open(scene_image_path3).resize((1024, 1024))]

    
    
    # ---------------------------------------additional for contrlnet---------------------------------------
    if controlnet_path:
        control_image = cv2.imread("examples/yann-lecun.jpg")
        canny_map = cv2.Canny(control_image, 50, 200)
        canny_map = cv2.resize(canny_map, (1024, 1024))
        canny_map = Image.fromarray(cv2.cvtColor(canny_map, cv2.COLOR_BGR2RGB))
        canny_maps = [canny_map] * len(prompts)
    else:
        canny_maps = None


    # ---------------------------------------additional for photomaker---------------------------------------
    if photomaker_path:
        input_folder_name = 'examples/newton_man'
        image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in os.listdir(input_folder_name)])
        input_id_images = [load_image(image_path) for image_path in image_path_list]
        #
        style_strength_ratio = 20
        start_merge_step = int(float(style_strength_ratio) / 100 * num_inference_steps)
        start_merge_step = min(start_merge_step, 30)
    else:
        input_id_images = None
        start_merge_step = -1

    # register the scene sharing attention
    register_scene_sharing_attention(pipe, prompts, subjects, use_scene_sharing_attention)



    # ************************************************************************************************************************



    # generate story images
    images = pipe.generate_story(
        scene_image=scene_images,
        prompt=prompts,
        negative_prompt=negative_prompts,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device).manual_seed(seed),
        guidance_scale=15, # 15
        scene_scale=scene_scale,
        # ControlNet parameters
        control_image=canny_maps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        # PhotoMaker parameters
        input_id_images=input_id_images,
        start_merge_step=start_merge_step,
    ).images

    # save results
    os.makedirs(output_path, exist_ok=True)
    print(f"save path at: {output_path}")
 

    # images.insert(0, scene_images[0])
    # prompts.insert(0, "Scene Image")
    # show_results(images, prompts, font_size=20, save_path=os.path.join(output_path, 'generated_stories.png'))


    for index, image in enumerate(images):
        if prompts[index].endswith('.'):
            image.save(os.path.join(output_path, f"{prompts[index]}png"))
        else:
            image.save(os.path.join(output_path, f"{prompts[index]}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--scene_encoder_path", type=str, default="sdxl_ckpts/image_encoder")
    parser.add_argument("--scene_adapter_path", type=str, default="sdxl_ckpts/ip-adapter_sdxl.bin")
    parser.add_argument("--controlnet_path", type=str, default=None)
    parser.add_argument("--photomaker_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default='./generated_stories')
    #
    parser.add_argument("--scene_scale", type=float, default=1.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=777277) # 42
    #
    parser.add_argument("--use_mask_guided_scene_injection", action='store_true',
                        help='whether to use the mask guided scene injection.')
    parser.add_argument("--use_scene_sharing_attention", action='store_true', 
                        help='whether to use the scene sharing attention.')
    parser.add_argument("--use_extrapolable_noise_blending", action='store_true',
                        help='whether to use the extrapolable noise blending. When there are many stories, enable this option to save VRAM, but it will increase inference time.')
    parser.add_argument("--t_1", type=int, default=0)
    parser.add_argument("--t_2", type=int, default=15)
    #
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
