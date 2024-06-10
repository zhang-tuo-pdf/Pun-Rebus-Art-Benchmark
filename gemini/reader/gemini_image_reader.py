import os
import pdb
import time
import base64
import mimetypes
import PIL.Image

import google.generativeai as genai

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_mime_type(image_path):
    # Extract the file extension from the path
    _, file_extension = os.path.splitext(image_path)
    
    # Get the MIME type for the given file extension
    mime_type, _ = mimetypes.guess_type(image_path)
    
    # Handle cases where MIME type is not found
    if mime_type is None:
        return f"Unknown MIME type for extension: {file_extension}"
    return mime_type


def gemini_image_understanding(model, prompt, image_path):
    
    img = PIL.Image.open(image_path)
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    response_text = response.text
    response_text = response_text.replace("*", "")
    response_text = response_text.strip()
    return response_text

def process_folder(prompt, folder_path, output_file_path):
    
    file_number = 0
    damage_file = []
    with open(output_file_path, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")
            try:
                response_text = gemini_image_understanding(model, prompt, image_path)
                outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                outfile.write(f"#################################################################################\n")
                file_number += 1
                time.sleep(10)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                damage_file.append(filename)
                
            print(file_number)
            print(response_text)
    file = open('damage_items.txt','w')
    for image_path in damage_file:
        file.write(image_path+"\n")
    file.close()

if __name__ == "__main__":
    genai.configure(api_key=GOOGLE_API_KEY)
    # import pdb; pdb.set_trace()
    folder_path = "../../Pun Chinese Painting"
    model_type = 'gemini-pro-vision'
    output_name = model_type + '_word_results_apr_14.txt'
    
    model = genai.GenerativeModel('gemini-pro-vision')
    generation_config = genai.GenerationConfig(
        temperature=0.01
    )
    
    output_path = '../multiple_choice_answers/' + output_name
    prompt = "This is a traditional Chinese artwork that likely conveys its ideas, thoughts, or wishes through symbolic, punning, shape, color, figure, numeral, verb, preposition, character, loanword or alias through the artwork. \
            Carefully analyze the visual elements present in the artwork and select the option from the list below that best aligns with its conveyed meaning: \n \
            A. Longevity and Good Health \n \
            B. Happiness,Joy, Good Luck \n \
            C. Prestige, Promotion, and Good Exam Results \n \
            D. Fecundity, Harmonious Relationship and Family \n \
            E. Wealth or Prosperity \n \
            F. Moral Integrity, Eremitism \n \
            G. Peace and Protection from Evil, Societal Harmony \n \
            You must make a selection using the option above in your response. Your response should start with the chosen option that best matches the word's meaning based on a precise and sound justification for your selection. Please do not include your reasoning in your response."
    
    process_folder(
        prompt, 
        folder_path, 
        output_path
    )
