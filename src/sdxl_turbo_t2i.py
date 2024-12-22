#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

from src.pbroad import *

prompt_text = """
{
  "4": {
    "inputs": {
      "ckpt_name": "dreamshaperXL_v21TurboDPMSDE.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
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
      "text": "Living room, contemporary style, sky blue velvet sectional sofa, white color wallpaper, with smooth surface texture, beige color flooring, with matte ceramic pattern, round white ottoman coffee table, cream woven rug, white pendant lights",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds",
      "clip": [
        "4",
        1
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
        "14",
        1
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
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
      "steps": 10,
      "denoise": 1,
      "model": [
        "4",
        0
      ]
    },
    "class_type": "SDTurboScheduler",
    "_meta": {
      "title": "SDTurboScheduler"
    }
  },
  "13": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "14": {
    "inputs": {
      "add_noise": true,
      "noise_seed": 79151145580663,
      "cfg": 8,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "sampler": [
        "13",
        0
      ],
      "sigmas": [
        "10",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "SamplerCustom",
    "_meta": {
      "title": "SamplerCustom"
    }
  }
}
"""

prompt = json.loads(prompt_text)

saveAddress = "/home/work/lifescape/ComfyUI/tmp" #사진 저장할 곳. 알아서 바꾸

# Initialize WebSocket connection
ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

def sdxl_turbo_t2i(
  # ws=None,
  fmodel=None, 
  fpos_prompt="thvk, best quality", 
  fnegative_prompt="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds",  
  fbatch=1,
  fbaseSeed=-1, 
  fsteps=10,
  ):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]

    prompt["4"]["inputs"]["ckpt_name"] = fmodel

    prompt["6"]["inputs"]["text"] = fpos_prompt
    prompt["7"]["inputs"]["text"] = fnegative_prompt
    prompt["10"]["inputs"]["steps"] = fsteps

    fbaseSeed = random.randint(1, 100000000) if fbaseSeed == -1 else fbaseSeed

    prompt["14"]["inputs"]["noise_seed"] = fbaseSeed
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
