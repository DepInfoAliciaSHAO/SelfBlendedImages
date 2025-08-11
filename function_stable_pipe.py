from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from tqdm import tqdm

def stable_diffusion_inference(image_path, output_path, prompt="A realistic photo of a person"):
    strength = 0.01
    guidance_scale = 1.0

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")

    os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(image_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for image in tqdm(images, desc="Processing images"):
        img = Image.open(os.path.join(image_path, image)).convert("RGB")
        img = img.resize((512, 512))

        img = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=50,
            strength=strength,
            guidance_scale=guidance_scale
        ).images[0]

        img.save(os.path.join(output_path, image))

if __name__ == "__main__":
    image_path = "../../extracted_frames"
    output_path = "../../stable_frames_2"
    stable_diffusion_inference(image_path, output_path)