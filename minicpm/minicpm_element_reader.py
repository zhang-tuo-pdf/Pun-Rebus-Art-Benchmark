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



def process_folder(prompt, folder_path, output_file):

    file_number = 0
    with open(output_file, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")

                image = PIL.Image.open(image_path).convert('RGB')
                msgs = [{'role': 'user', 'content': [image, prompt]}]

                answer = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer
                )

                outfile.write(f"Image: {filename}\n\n{answer}\n\n")
                outfile.write(f"#################################################################################\n")
                file_number += 1
               
if __name__ == "__main__":

    # Argument parser
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16, cache_dir="/vault/ultraz") # sdpa or flash_attention_2, no eager
    device = 'cuda:5'
    model.to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    
    folder_path = "/home/ultraz/Project/Pun-Rebus-Art-Benchmark/Pun Chinese Painting"

    # output_name = f'MiniCPM-V-2_6_word_results.txt'
    # os.makedirs("element_answers", exist_ok=True)
    # output_path = os.path.join("element_answers", output_name)
    
    # prompt = "Please analyze the provided image carefully to identify key visual elements. Focus on components that traditionally have symbolic meaning in the cultural context from which the artwork originates.\
    # Look for elements that might represent ideas, virtues, or wishes, especially those commonly found in nature or historical motifs.\
    # For instance, in Chinese culture, certain animals and plants are known to symbolize specific messages when depicted in art. \
    # Based on these principles, identify the primary visual elements in the image that are likely used to convey a message or a wish.\
    # Do not explain, please list the discernible elements present in the image, excluding any assumptions about elements not clearly visible.\
    # Please answer the question in one line with the following format strictly: name of element A, name of element B, etc"

    output_name = f'MiniCPM-V-2_6_mc_results.txt'
    os.makedirs("mc_answers", exist_ok=True)
    output_path = os.path.join("mc_answers", output_name)

    prompt = "This is a traditional Chinese artwork that likely conveys its ideas, thoughts, or wishes through symbolic, punning, shape, color, figure, numeral, verb, preposition, character, loanword or alias through the artwork. \
        Carefully analyze the visual elements present in the artwork and select the option from the list below that best aligns with its conveyed meaning: \n \
        A. Longevity and Good Health \n \
        B. Happiness,Joy, Good Luck \n \
        C. Prestige, Promotion, and Good Exam Results \n \
        D. Fecundity, Harmonious Relationship and Family \n \
        E. Wealth or Prosperity \n \
        F. Moral Integrity, Eremitism \n \
        G. Peace and Protection from Evil, Societal Harmony \n \
        You must make a selection using the option above in your response. Your response should start with the chosen letter that best matches the word's meaning based on a precise and sound justification for your selection. Please do not include your justification in your response."

    process_folder(prompt, folder_path, output_path)