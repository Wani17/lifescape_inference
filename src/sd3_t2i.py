import torch
from diffusers import StableDiffusion3Pipeline
import os

pipe = None

def sd3_t2i_setPipe(fmodel = None):
    global pipe
    pipe = StableDiffusion3Pipeline.from_pretrained(
        fmodel, torch_dtype=torch.float16
    ).to("cuda")


def sd3_t2i(
  fmodel=None, 
  fpos_prompt="thvk, best quality", 
  fnegative_prompt="",  
  fbatch=1,
  fsteps=28,
  ):

    image = pipe(
            fpos_prompt,
            negative_prompt=fnegative_prompt,
            num_inference_steps=fsteps,
            guidance_scale=7.0,
            height=1024,
            width=1024,
            num_images_per_prompt=fbatch
    ).images
    
    return [fbaseSeed, image]
