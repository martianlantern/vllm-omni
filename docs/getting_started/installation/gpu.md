# GPU

vLLM-Omni is a Python library that supports the following GPU variants. The library itself mainly contains python implementations for framework and models.

## Requirements

- OS: Linux
- Python: 3.12

!!! note
    vLLM-Omni is currently not natively supported on Windows.

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:requirements"

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### Pre-built wheels

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-wheels"

[](){ #build-from-source }

### Build wheel from source

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images
vLLM-Omni offers an official docker image for deployment. These images are built on top of vLLM docker images
and available on Docker Hub as [vllm/vllm-omni](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni
indicates which release of vLLM it is based on.

!!! tip
    You can use this docker image to serve models the same way you would do in vLLM! To do so, make sure you overwrite the default entrypoint (`vllm serve --omni`) which works only for models supported in the vLLM-Omni project.
