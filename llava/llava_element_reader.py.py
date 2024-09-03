import os
import time
import base64
import requests
import PIL.Image

import json
import yaml
import torch
import numpy as np
import pandas as pd
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoProcessor, AutoTokenizer

from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def process_folder(prompt, folder_path, output_file):

    file_number = 0
    with open(output_file, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")

                chat_prompt = f"[INST] <image>\n{prompt}[/INST]"

                image = PIL.Image.open(image_path)
                inputs = processor(
                    chat_prompt, 
                    image, 
                    return_tensors="pt"
                ).to("cuda")

                # autoregressively complete prompt
                output = model.generate(
                    **inputs, 
                    max_new_tokens=500,
                    temperature=0.0,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

                response_text = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[1].replace("\n", " ").replace("*", "")
                outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                outfile.write(f"#################################################################################\n")
                file_number += 1
               
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='VLM')
    parser.add_argument(
        '--model', 
        # default="llava-hf/llava-v1.6-vicuna-7b-hf",
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        # default="microsoft/Phi-3-vision-128k-instruct",
        # default="openbmb/MiniCPM-Llama3-V-2_5",
        type=str, 
        help='model name'
    )
    
    args = parser.parse_args()

    processor = LlavaNextProcessor.from_pretrained(
        args.model,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        cache_dir="/project/shrikann_35/tiantiaf/llm",
    ) 
    model.to("cuda")

    folder_path = "/scratch1/tiantiaf/pun-rebus/Pun_Chinese_Painting"
    model_type = args.model.replace("/", "_")
    output_name = f'{model_type}_word_results.txt'
    os.makedirs("element_answers", exist_ok=True)
    output_path = os.path.join("element_answers", output_name)
    
    prompt = "Please analyze the provided image carefully to identify key visual elements. Focus on components that traditionally have symbolic meaning in the cultural context from which the artwork originates.\
    Look for elements that might represent ideas, virtues, or wishes, especially those commonly found in nature or historical motifs.\
    For instance, in Chinese culture, certain animals and plants are known to symbolize specific messages when depicted in art. \
    Based on these principles, identify the primary visual elements in the image that are likely used to convey a message or a wish.\
    Do not explain, please list the discernible elements present in the image, excluding any assumptions about elements not clearly visible.\
    Please answer the question in one line with the following format strictly: name of element A, name of element B, etc"
    process_folder(prompt, folder_path, output_path)