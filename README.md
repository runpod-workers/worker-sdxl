# SDXL Worker
[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl)

Run [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) as a serverless endpoint.

This worker provides a serverless API for generating images using Stable Diffusion XL. It supports various parameters for customizing the generation process, including prompt, negative prompt, dimensions, inference steps, guidance scale, and more.

## API Reference

The worker accepts the following input parameters:

```json
{
  "prompt": "A majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
  "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature",
  "height": 1024,
  "width": 1024,
  "num_inference_steps": 25,
  "refiner_inference_steps": 50,
  "guidance_scale": 7.5,
  "strength": 0.3,
  "high_noise_frac": 0.8,
  "seed": 42,
  "scheduler": "K_EULER",
  "num_images": 1,
  "image_url": null
}
```