#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint
from src.pbroad import *

prompt_text = """
{
  "9": {
    "inputs": {
      "text": "thvk, Living room, Scandinavian style, gray upholstered sectional sofa, white color wallpaper, with smooth matte finish, beige color flooring, with smooth wood pattern, natural wood side table, textured jute round rug, minimalistic artwork with natural elements, white sheer curtains",
      "clip": [
        "19",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "10": {
    "inputs": {
      "text": "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds",
      "clip": [
        "19",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "19": {
    "inputs": {
      "ckpt_name": "1021img_base_model-000080.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "79": {
    "inputs": {
      "add_noise": "enable",
      "noise_seed": 211058108376247,
      "steps": 60,
      "cfg": 8,
      "sampler_name": "dpmpp_3m_sde_gpu",
      "scheduler": "karras",
      "start_at_step": 0,
      "end_at_step": 48,
      "return_with_leftover_noise": "disable",
      "model": [
        "19",
        0
      ],
      "positive": [
        "9",
        0
      ],
      "negative": [
        "10",
        0
      ],
      "latent_image": [
        "89",
        0
      ]
    },
    "class_type": "KSamplerAdvanced",
    "_meta": {
      "title": "Base-KSampler (Advanced)"
    }
  },
  "80": {
    "inputs": {
      "add_noise": "enable",
      "noise_seed": 102199419495701,
      "steps": 60,
      "cfg": 8,
      "sampler_name": "dpmpp_3m_sde",
      "scheduler": "karras",
      "start_at_step": 48,
      "end_at_step": 60,
      "return_with_leftover_noise": "disable",
      "model": [
        "81",
        0
      ],
      "positive": [
        "82",
        0
      ],
      "negative": [
        "83",
        0
      ],
      "latent_image": [
        "79",
        0
      ]
    },
    "class_type": "KSamplerAdvanced",
    "_meta": {
      "title": "Refiner-KSampler (Advanced)"
    }
  },
  "81": {
    "inputs": {
      "ckpt_name": "sd_xl_refiner_1.0.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "82": {
    "inputs": {
      "ascore": 6,
      "width": 1024,
      "height": 1024,
      "text": "",
      "clip": [
        "81",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXLRefiner",
    "_meta": {
      "title": "Positive-CLIPTextEncodeSDXLRefiner"
    }
  },
  "83": {
    "inputs": {
      "ascore": 6,
      "width": 1024,
      "height": 1024,
      "text": "",
      "clip": [
        "81",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXLRefiner",
    "_meta": {
      "title": "Negative-CLIPTextEncodeSDXLRefiner"
    }
  },
  "84": {
    "inputs": {
      "samples": [
        "80",
        0
      ],
      "vae": [
        "81",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "87": {
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
  "89": {
    "inputs": {
      "amount": 1,
      "samples": [
        "87",
        0
      ]
    },
    "class_type": "RepeatLatentBatch",
    "_meta": {
      "title": "Repeat Latent Batch"
    }
  },
  "90": {
    "inputs": {
      "images": [
        "84",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  }
}
"""

prompt = json.loads(prompt_text)

saveAddress = "/home/work/lifescape/ComfyUI/tmp" #사진 저장할 곳. 알아서 바꾸

# Initialize WebSocket connection
ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

def sdxl_t2i(
  # ws=None,
  fmodel=None, 
  fpos_prompt="thvk, best quality", 
  fnegative_prompt="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds",  
  floramodel="", 
  fbatch=1,
  fbaseSeed=-1, 
  frefinerSeed=-1):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]

    prompt["19"]["inputs"]["ckpt_name"] = fmodel

    prompt["9"]["inputs"]["text"] = fpos_prompt
    prompt["10"]["inputs"]["text"] = fnegative_prompt


    fbaseSeed = random.randint(1, 100000000) if fbaseSeed == -1 else fbaseSeed
    frefinerSeed = fbaseSeed if frefinerSeed == -1 else frefinerSeed

    prompt["79"]["inputs"]["noise_seed"] = fbaseSeed
    prompt["80"]["inputs"]["noise_seed"] = frefinerSeed

    #print("base seed:" + str(prompt["79"]["inputs"]["noise_seed"]))
    #print("refiner seed:" + str(prompt["80"]["inputs"]["noise_seed"]))

    prompt["89"]["inputs"]["amount"]=fbatch

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
