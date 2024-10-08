<div align="center">

<h1>Stable Diffusion XL | Worker</h1>

[![CI | Test Worker](https://github.com/runpod-workers/worker-sdxl/actions/workflows/CI-test_worker.yml/badge.svg)](https://github.com/runpod-workers/worker-sdxl/actions/workflows/CI-test_worker.yml)
&nbsp;
[![CD | Dev Docker Image](https://github.com/runpod-workers/worker-sdxl/actions/workflows/CD-docker_dev.yml/badge.svg)](https://github.com/runpod-workers/worker-sdxl/actions/workflows/CD-docker_dev.yml)

🚀 | RunPod implementation of SDXL for serverless deployment.
</div>

## 📖 | Getting Started

1. Clone this repository.
2. (Optional) Add DockerHub credentials to GitHub Secrets.
3. Add your code to the `src` directory.
4. Update the `rp_handler.py` file to load models and process requests.
5. Add any dependencies to the `requirements.txt` file.
6. Add any other build time scripts to the`builder` directory, for example, downloading models.
7. Update the `Dockerfile` to include any additional dependencies.

### CI/CD

This repository is setup to automatically build and push a docker image to the GitHub Container Registry. You will need to add the following to the GitHub Secrets for this repository to enable this functionality:

- `DOCKERHUB_USERNAME` | Your DockerHub username for logging in.
- `DOCKERHUB_TOKEN` | Your DockerHub token for logging in.
- `DOCKERHUB_REPO` | The name of the repository you want to push to.
- `DOCKERHUB_IMG` | The name of the image you want to push to.

The `CD-docker_dev.yml` file will build the image and push it to the `dev` tag, while the `CD-docker_release.yml` file will build the image on releases and tag it with the release version.

The `CI-test_worker.yml` file will test the worker using the input provided by the `--test_input` argument when calling the file containing your handler. Be sure to update this workflow to install any dependencies you need to run your tests.

## 🔗 | Links

🐳 [Docker Container](https://hub.docker.com/r/runpod/ai-api-sdxl)
