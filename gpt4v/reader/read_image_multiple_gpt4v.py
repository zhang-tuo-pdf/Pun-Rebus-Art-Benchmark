from openai import OpenAI
import os
import base64
import time
import requests

os.environ["OPENAI_API_KEY"] = "please enter you api keys"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gp4v_read_image(prompt, image_path_1):
    # Getting the base64 string
    example_1_image = encode_image(image_path_1)

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
                "url": f"data:image/jpeg;base64,{example_1_image}",
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
    file_number = 0
    with open(output_file, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")
                print(file_number)
                try:
                    response_text = gp4v_read_image(prompt, image_path)
                    outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                    outfile.write(f"#################################################################################\n")
                    file_number += 1
                    time.sleep(30)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
            print(file_number)


if __name__ == "__main__":
    folder_path = "../../Pun Chinese Painting"
    output_file = "../multiple_choice_answers/gpt4o_mc_results_full_may22.txt"
    prompt = "This is a traditional Chinese artwork that likely conveys its ideas, thoughts, or wishes through symbolic, punning, shape, color, figure, numeral, verb, preposition, character, loanword or alias through the artwork. \
            Carefully analyze the visual elements present in the artwork and select the option from the list below that best aligns with its conveyed meaning: \n \
            A. Longevity and Good Health \n \
            B. Happiness,Joy, Good Luck \n \
            C. Prestige, Promotion, and Good Exam Results \n \
            D. Fecundity, Harmonious Relationship and Family \n \
            E. Wealth or Prosperity \n \
            F. Moral Integrity, Eremitism \n \
            G. Peace and Protection from Evil, Societal Harmony \n \
            You must make a selection using the option above in your response. Your response should start with the chosen letter that best matches the word's meaning, followed by a precise and sound justification for your selection."
    process_folder(prompt, folder_path, output_file)