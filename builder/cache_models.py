# builder/model_fetcher.py

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline


def get_diffusion_pipelines():
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                     torch_dtype=torch.float16,
                                                     variant="fp16",
                                                     use_safetensors=True)

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                               torch_dtype=torch.float16,
                                                               variant="fp16",
                                                               use_safetensors=True)

    return pipe, refiner


if __name__ == "__main__":
    get_diffusion_pipelines()
