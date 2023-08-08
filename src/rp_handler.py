'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

# Setup the models
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
)
refiner.to("cuda")
refiner.enable_xformers_memory_efficient_attention()


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        image_url = rp_upload.upload_image(job_id, image_path)
        image_urls.append(image_url)
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    # Extracting the new parameters
    prompt = validated_input['validated_input'].get('prompt')
    negative_prompt = validated_input['validated_input'].get('negative_prompt')
    height = validated_input['validated_input'].get('height')
    width = validated_input['validated_input'].get('width')
    num_inference_steps = validated_input['validated_input'].get(
        'num_inference_steps', 25)  # Default value if not provided
    refiner_inference_steps = validated_input['validated_input'].get(
        'refiner_inference_steps', 50)  # Default value
    guidance_scale = validated_input['validated_input'].get('guidance_scale', 7.5)  # Default value
    image_url = validated_input['validated_input'].get('image_url')
    strength = validated_input['validated_input'].get('strength', 0.3)

    if image_url:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(image_url).convert("RGB")
        output = refiner(
            prompt=prompt,
            num_inference_steps=refiner_inference_steps,  # Using refiner_inference_steps for refiner
            strength=strength,
            image=init_image
        ).images[0]
    else:
        # Generate latent image using pipe
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,  # Using num_inference_steps for pipe
            guidance_scale=guidance_scale,
            output_type="latent"
        ).images[0]

        # Refine the image using refiner with refiner_inference_steps
        output = refiner(
            prompt=prompt,
            num_inference_steps=refiner_inference_steps,  # Using refiner_inference_steps for refiner
            strength=strength,
            image=image[None, :]
        ).images[0]

    image_urls = _save_and_upload_images([output], job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}


runpod.serverless.start({"handler": generate_image})
