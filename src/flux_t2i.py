#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

from src.pbroad import *

prompt_text = """
{
  "5": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "6": {
    "inputs": {
      "text": [
        "28",
        0
      ],
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "MarkuryFLUX",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp16.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-schnell.safetensors",
      "weight_dtype": "fp8_e4m3fn"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "simple",
      "steps": 25,
      "denoise": 1,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "12",
        0
      ],
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": 282420476834494
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "28": {
    "inputs": {
      "string": "Living room, contemporary style, sky blue velvet sectional sofa, white color wallpaper, with smooth surface texture, beige color flooring, with matte ceramic pattern, round white ottoman coffee table, cream woven rug, white pendant lights"
    },
    "class_type": "String Literal",
    "_meta": {
      "title": "String Literal"
    }
  }
}
"""

prompt = json.loads(prompt_text)

saveAddress = "/home/work/lifescape/ComfyUI/tmp" #사진 저장할 곳. 알아서 바꾸

# Initialize WebSocket connection
ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

def flux_t2i(
  # ws=None,
  fmodel=None, 
  fpos_prompt="thvk, best quality", 
  fbatch=1,
  fbaseSeed=-1, 
  ):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]

    prompt["12"]["inputs"]["unet_name"] = fmodel

    prompt["28"]["inputs"]["string"] = fpos_prompt

    fbaseSeed = random.randint(1, 100000000) if fbaseSeed == -1 else fbaseSeed

    prompt["25"]["inputs"]["noise_seed"] = fbaseSeed
    #prompt["80"]["inputs"]["noise_seed"] = frefinerSeed

    #print("base seed:" + str(prompt["79"]["inputs"]["noise_seed"]))
    #print("refiner seed:" + str(prompt["80"]["inputs"]["noise_seed"]))

    prompt["5"]["inputs"]["batch_size"]=fbatch

    images = get_images(ws, prompt)

    ret_image = []

    for node_id in images:
        for image_data in images[node_id]:
            ret_image.append(Image.open(io.BytesIO(image_data)))
    
    return [fbaseSeed, ret_image]

#ws = websocket.WebSocket()
#ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

"""
image = Image.open('a2.jpeg')
mask = Image.open('a1.png')

getI = getImage(
  fmodel="1021img_base_model-000080.safetensors",
  fpos_prompt="thvk, Living room, modern style, light taupe color wallpaper, with vertical ribbed texture, soft beige color flooring, with smooth wood pattern, round white wool rug, white sculptural side table, minimalist wall art, semi-circular wall shelf, soft light, diffuse light, open shade",
  finputImage=image,  
  fmaskImage=mask,
  fbaseSeed = 456919324482822,
  fbatch=2
  ) 


current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]

for idx, img in enumerate(getI):
    img.save(str(idx) + "f.png")
"""
