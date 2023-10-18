'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA


# ------------------------------- Model Handler ------------------------------ #
class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        ).to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe


    def load_refiner(self):
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        ).to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)

            self.base = future_base.result()
            self.refiner = future_refiner.result()

MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
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

    starting_image = job_input['image_url']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    MODELS.base.scheduler = make_scheduler(job_input['scheduler'], MODELS.base.scheduler.config)

    if starting_image:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator
        ).images
    else:
        # Generate latent image using pipe
        image = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        # Refine the image using refiner with refiner_inference_steps
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=image,
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

    image_urls = _save_and_upload_images(output, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results


runpod.serverless.start({"handler": generate_image})
