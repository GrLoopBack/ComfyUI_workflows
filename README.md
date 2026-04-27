# ComfyUI workflows

ComfyUI Workflows with Flux1 and Flux2, dev, klein. 

## Flux2 workflows 

In flux2 workflows, do not forget to change the QwenVL

1. flux-2-klein-9b-fp8  uses qwen_3_8b_fp8
2. flux-2-klein-4b uses qwen 4b 
3. flux2 dev uses mistral 3 small.

## LTX 2.3 image/text workflow

LTX 2.3 image/text to vid with 16GB VRAM.

Runs on NVIDIA 4080. 
I start comfyui with:

```
python3.13 -m venv comfy-env && source comfy-env/bin/activate && comfy launch -- --disable-cuda-malloc --use-sage-attention --reserve-vram 4 --cache-none

```
You find the models for 16 GB from https://ltxworkflow.com/models and https://huggingface.co/Kijai/LTX2.3_comfy 

