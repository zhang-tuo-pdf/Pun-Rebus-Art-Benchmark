from openai import OpenAI
import os
import base64
import time
import requests


os.environ["OPENAI_API_KEY"] = "please enter you api keys"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gp4v_read_image(prompt, image_path_4):
    # Getting the base64 string
    question_image = encode_image(image_path_4)

    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", 
            "text": prompt},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{question_image}",
                "detail": "high"
            },
            },
        ],
        }
    ],
    max_tokens=2000,
    )
    response_text = response.choices[0].message.content

    return response_text

def process_folder(prompt, folder_path, output_file):
    with open(output_file, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")
                try:
                    response_text = gp4v_read_image(prompt, image_path)
                    outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                    outfile.write(f"#################################################################################\n")
                    time.sleep(40)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    folder_path = "../../Pun Chinese Painting"
    output_file = "../element_answers/gpt4o_element_results_may_14.txt"
    prompt = "Please analyze the provided image carefully to identify key visual elements. Focus on components that traditionally have symbolic meaning in the cultural context from which the artwork originates.\
    Look for elements that might represent ideas, virtues, or wishes, especially those commonly found in nature or historical motifs.\
    For instance, in Chinese culture, certain animals and plants are known to symbolize specific messages when depicted in art. \
    Based on these principles, identify the primary visual elements in the image that are likely used to convey a message or a wish.\
    Please list the discernible elements present in the image, excluding any assumptions about elements not clearly visible.\
    Pleas answer the question in one line with the following format strictly: name of element A, name of element B, etc"
    process_folder(prompt, folder_path, output_file)