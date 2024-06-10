from http import HTTPStatus
import dashscope
import pandas
import os 
import time


def qwen_api_read_image(model_name, prompt, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {'image': image_path}, 
                {"text": prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model=model_name,
                                                     messages=messages)

    return response.output.choices[0].message.content[0]["text"]

def process_folder(model_name, prompt, folder_path, output_file):
    file_number = 0
    with open(output_file, "w") as outfile:
        outfile.write(f"Prompt: {prompt}\n")
        outfile.write(f"#################################################################################\n")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {image_path}...")
                try:
                    response_text = qwen_api_read_image(model_name, prompt, image_path)
                    outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                    outfile.write(f"#################################################################################\n")
                    file_number += 1
                    time.sleep(10)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
            print(file_number)


if __name__ == '__main__':
    model = 'qwen-vl-max' # 'qwen-vl-plus' or 'qwen-vl-max'
    dashscope.api_key = "please enter you api keys"
    folder_path = "../../Pun Chinese Painting"
    output_name = model + '_mc_results_may_30.txt'
    output_file = "../multiple_choice_answers/" + output_name
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
    process_folder(model, prompt, folder_path, output_file)