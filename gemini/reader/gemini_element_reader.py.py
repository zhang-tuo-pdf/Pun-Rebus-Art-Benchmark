import os
import time
import base64
import requests
import PIL.Image

import google.generativeai as genai

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gemini_read_image(prompt, image_path_4):
    img = PIL.Image.open(image_path_4)
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    response_text = response.text
    response_text = response_text.replace("*", "")
    response_text = response_text.strip()
    return response_text

def process_folder(prompt, folder_path, output_file):

    file_number = 0
    with open(output_file, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")
                try:
                    response_text = gemini_read_image(prompt, image_path)
                    outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                    outfile.write(f"#################################################################################\n")
                    file_number += 1
                    time.sleep(5)
                    print(file_number)
                    print(response_text)
                    # import pdb; pdb.set_trace()
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":

    genai.configure(api_key=GOOGLE_API_KEY)
    folder_path = "../../Pun Chinese Painting"
    model_type = 'gemini-pro-vision'
    output_name = model_type + '_word_results_may_12.txt'
    output_path = '../element_answers/' + output_name

    model = genai.GenerativeModel('gemini-pro-vision')
    generation_config = genai.GenerationConfig(
        temperature=0.01
    )

    prompt = "Please analyze the provided image carefully to identify key visual elements. Focus on components that traditionally have symbolic meaning in the cultural context from which the artwork originates.\
    Look for elements that might represent ideas, virtues, or wishes, especially those commonly found in nature or historical motifs.\
    For instance, in Chinese culture, certain animals and plants are known to symbolize specific messages when depicted in art. \
    Based on these principles, identify the primary visual elements in the image that are likely used to convey a message or a wish.\
    Please list the discernible elements present in the image, excluding any assumptions about elements not clearly visible.\
    Pleas answer the question in one line with the following format strictly: name of element A, name of element B, etc"
    process_folder(prompt, folder_path, output_path)