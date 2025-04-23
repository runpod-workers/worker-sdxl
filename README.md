![SDXL Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl_banner-c7nsJLBOGHnmsxcshN7kSgALHYawnW.jpeg)

---

Run [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) as a serverless endpoint to generate images.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl)

---

## Usage

The worker accepts the following input parameters:

| Parameter                 | Type    | Default  | Required  | Description                                                                                                         |
| :------------------------ | :------ | :------- | :-------- | :------------------------------------------------------------------------------------------------------------------ |
| `prompt`                  | `str`   | `None`   | **Yes\*** | The main text prompt describing the desired image.                                                                  |
| `negative_prompt`         | `str`   | `None`   | No        | Text prompt specifying concepts to exclude from the image                                                           |
| `height`                  | `int`   | `1024`   | No        | The height of the generated image in pixels                                                                         |
| `width`                   | `int`   | `1024`   | No        | The width of the generated image in pixels                                                                          |
| `seed`                    | `int`   | `None`   | No        | Random seed for reproducibility. If `None`, a random seed is generated                                              |
| `scheduler`               | `str`   | `'DDIM'` | No        | The noise scheduler to use. Options include `PNDM`, `KLMS`, `DDIM`, `K_EULER`, `DPMSolverMultistep`                 |
| `num_inference_steps`     | `int`   | `25`     | No        | Number of denoising steps for the base model                                                                        |
| `refiner_inference_steps` | `int`   | `50`     | No        | Number of denoising steps for the refiner model                                                                     |
| `guidance_scale`          | `float` | `7.5`    | No        | Classifier-Free Guidance scale. Higher values lead to images closer to the prompt, lower values more creative       |
| `strength`                | `float` | `0.3`    | No        | The strength of the noise added when using an `image_url` for image-to-image or refinement                          |
| `image_url`               | `str`   | `None`   | No        | URL of an initial image to use for image-to-image generation (runs only refiner). If `None`, performs text-to-image |
| `num_images`              | `int`   | `1`      | No        | Number of images to generate per prompt (Constraint: must be 1 or 2)                                                |
| `high_noise_frac`         | `float` | `None`   | No        | Fraction of denoising steps performed by the base model (e.g., 0.8 for 80%). `denoising_end` for base               |

> [!NOTE]  
> `prompt` is required unless `image_url` is provided

### Example Request

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
