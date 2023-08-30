'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
import concurrent.futures
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA


# -------------------------------- Load Models ------------------------------- #
def load_base():
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
    ).to("cuda")
    base_pipe.enable_xformers_memory_efficient_attention()
    return base_pipe


def load_refiner():
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
    ).to("cuda")
    refiner_pipe.enable_xformers_memory_efficient_attention()
    return refiner_pipe


with concurrent.futures.ThreadPoolExecutor() as executor:
    future_base = executor.submit(load_base)
    future_refiner = executor.submit(load_refiner)

    base = future_base.result()
    refiner = future_refiner.result()


# ---------------------------------- Helper ---------------------------------- #
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
    job_input = validated_input['validated_input']

    image_url = job_input['image_url']

    if image_url:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(image_url).convert("RGB")
        output = refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image
        ).images[0]
    else:
        # Generate latent image using pipe
        image = base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images']
        ).images

        # Refine the image using refiner with refiner_inference_steps
        output = refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=image
        ).images[0]

    image_urls = _save_and_upload_images([output], job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}


runpod.serverless.start({"handler": generate_image})
