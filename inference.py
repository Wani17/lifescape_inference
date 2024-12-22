import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import random
from datetime import datetime
import os
import sys
from datetime import datetime
import time
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

def parse_arguments():
    """
    Parse command-line arguments and return the parsed object.
    """
    parser = argparse.ArgumentParser(description='Inference script for model evaluation.')
    
    # Required arguments
    parser.add_argument('--model', type=str, help='Model Name/Directory to load')
    parser.add_argument('--test_set', type=str, help='Test Set directory containing prompt .txt files')
    parser.add_argument('--model_class', type=str, choices=["sdxl", "sdxl_turbo", "sd3", "flux"], help='Define Model Category. [sdxl, sdxl_turbo, sd3, flux]')

    
    # Optional arguments
    parser.add_argument('--instance_token', type=str, help='set Instance token', default=None)
    parser.add_argument('--fix_positive', type=str, help='Positive prompt for quality', default=None)
    parser.add_argument('--fix_negative', type=str, help='Negative prompt for quality', default=None)
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--baseseed', type=int, help='Fix base seed value (optional)', default=-1)
    parser.add_argument('--refineseed', type=int, help='Fix refiner seed value (optinal)', default=-1)

    return parser.parse_args()


def main(args):
    """
    Main logic for loading models and performing inference.
    """

    model_class = args.model_class
    if model_class is None:
        print("--model_class [sdxl, sdxl_turbo, sd3, flux] is required. Define Model Category.")
        exit(0)

    model_dir = args.model

    if model_dir == None:
        print("--model [MODEL / MODEL_DIR] is required for test")
        exit(0)

    base_dir = "../models/checkpoints/" if model_class != "flux" else "../models/unet/"
    
    abs_file_path = os.path.abspath(model_dir)
    abs_base_dir = os.path.abspath(base_dir)

    if os.path.commonpath([abs_file_path, abs_base_dir]) != abs_base_dir and model_class != "sd3":
        print(f"--model directory inside {base_dir}")
        exit(0)
    
    model_rel = os.path.relpath(abs_file_path, abs_base_dir)

    print("Load checkpoints...")
    if not os.path.exists(model_dir):
        print("Model or directory not found: " + model_rel)
        exit(0)

    models = []

    if os.path.isdir(model_dir):
        for root, dirs, files in os.walk(model_dir):
            has_text_encoder = any("text_encoder" in d for d in dirs)

            if has_text_encoder:
                models.append(root[:-1])
                dirs[:] = []
            else:
                for file in files:
                    if file.endswith(".safetensors"):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, start=model_dir)
                        models.append(relative_path)

    else:
        models.append(model_rel)

    models.sort()

    print("Found " + str(len(models)) + " models to test: ")
    for model in models:
        print("    " + model)



    test_dir = args.test_set

    if test_dir == None:
        print("--test_set TEST_DIR is required for test")
        exit(0)

    print("\nLoad Testset...")
    if not os.path.exists(test_dir):
        print("Test set directory not found: " + args.test_set)
        exit(0)

    prompts = []

    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".txt"):
                prompts.append(file)

    prompts.sort()

    print("Found " + str(len(prompts)) + " prompts to test.\n")
    
    instance_token = args.instance_token
    if instance_token is None:
        logging.warning(f"Instance Token Is None.")
        logging.warning(f"If trying trained model with dreambooth, please Set Instance Token with --instance_token TOK")
        #print("E")
    
    fix_positive = args.fix_positive
    if fix_positive:
        print(f"Fix Positive Prompt: {fix_positive}")

    fix_negative = args.fix_negative
    if fix_negative == None and model_class != "sd3":
        fix_negative = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds"
    print(f"Fix Negative Prompt: {fix_negative}")
    
    batch = args.batch
    print(f"Batch set to {batch}")

    baseseed = args.baseseed
    if baseseed != -1:
        print(f"Base Seed Fixed to {baseseed}")
    
    refineseed = args.refineseed
    if refineseed != -1:
        print(f"Refiner Seed Fixed to {baseseed}")

    # Prepare output directory
    output_directory = "outputs"
    if os.path.exists(output_directory):
        if not os.path.isdir(output_directory):
            print("Output directory cannot be created due to name conflict.")
            exit(0)
    else:
        os.mkdir(output_directory)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]
    os.mkdir(os.path.join(output_directory, current_time))
    output_directory = os.path.join(output_directory, current_time)

    if model_class == "sd3":
        from src.sd3_t2i import sd3_t2i_setPipe
        from src.sd3_t2i import sd3_t2i

    # Iterate through models and prompts
    for idx, model in enumerate(models):
        model_name = os.path.splitext(os.path.basename(model))[0]
        print(f"Test model {idx+1}/{len(models)}: {model_name}")
        
        model_directory = os.path.join(output_directory, model_name)
        os.mkdir(model_directory)

        if model_class == "sd3":
            sd3_t2i_setPipe(
                fmodel = model
            )

        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            prompt_name = os.path.splitext(os.path.basename(prompt))[0]
            prompt_directory = os.path.join(model_directory, prompt_name)
            os.mkdir(prompt_directory)

            # Read content from the prompt file
            try:
                txt_file = os.path.join(args.test_set, prompt)
                with open(txt_file, 'r', encoding='utf-8') as file:
                    content = file.read()
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
                continue

            if instance_token:
                content= instance_token + ", " + content

            if fix_positive:
                content = content + ", " + fix_positive

            print(content)
            
            # Perform inference using the model and WebSocket
            output = []
            if model_class == "sdxl":
                from src.sdxl_t2i import sdxl_t2i
                output = sdxl_t2i(
                    # ws=ws,
                    fmodel=model,
                    fpos_prompt=content,
                    fnegative_prompt=fix_negative,
                    fbatch=batch,
                    fbaseSeed=baseseed, 
                    frefinerSeed=refineseed
                )
            elif model_class == "sdxl_turbo":
                from src.sdxl_turbo_t2i import sdxl_turbo_t2i
                output = sdxl_turbo_t2i(
                    # ws=ws,
                    fmodel=model,
                    fpos_prompt=content,
                    fnegative_prompt=fix_negative,
                    fbatch=batch,
                    fbaseSeed=baseseed, 
                )
            elif model_class == "flux":
                from src.flux_t2i import flux_t2i
                output = flux_t2i(
                    # ws=ws,
                    fmodel=model,
                    fpos_prompt=content,
                    fbatch=batch,
                    fbaseSeed=baseseed, 
                )
            elif model_class == "sd3":
                from src.sd3_t2i import sd3_t2i
                output = sd3_t2i(
                    fmodel=model, 
                    fpos_prompt=content, 
                    fnegative_prompt=fix_negative,  
                    fbatch=batch,
                )

            # Save the output seed
            with open(os.path.join(prompt_directory, f"{model_name}_{prompt_name}_seed.txt"), "w") as file:
                file.write(str(output[0]))

            # Save the generated images
            for idx, image in enumerate(output[1]):
                image.save(os.path.join(prompt_directory, f"{model_name}_{prompt_name}_{idx+1}.png"))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
