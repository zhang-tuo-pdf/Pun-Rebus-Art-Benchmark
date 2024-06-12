import os
import pdb
import time
import pandas as pd

import google.generativeai as genai

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_word_list(file_path):
    df = pd.read_csv(file_path, header='infer')
    df.dropna()
    word_list = df['Chinese Name'].tolist()
    return word_list

def create_prompt(chinese_word):
    prompt = f"What does the word \"{chinese_word}\" want to represent in Chinese culture? Please select the option from the list below that best aligns with its conveyed meaning: \n \
            A. Longevity and Good Health \n \
            B. Happiness,Joy, Good Luck \n \
            C. Prestige, Promotion, and Good Exam Results \n \
            D. Fecundity, Harmonious Relationship and Family \n \
            E. Wealth or Prosperity \n \
            F. Moral Integrity, Eremitism \n \
            G. Peace and Protection from Evil, Societal Harmony \n \
    You must make a selection using the option above in your response. Your response should start with the chosen letter that best matches the word's meaning, followed by a precise and sound justification for your selection."
    return prompt

def gemini_text_understanding(prompt):

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    generation_config = genai.GenerationConfig(
        temperature=0.01
    )
    while True:
        try:
            response = model.generate_content(prompt, stream=True)
            response.resolve()
            response_text = response.text
            response_text = response_text.replace("*", "")
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            pass
        time.sleep(3)
        break
    return response_text

def process_list(word_list, output_file_path):
    with open(output_file_path, "w") as outfile:
        prompt_format = create_prompt("xxxx")
        outfile.write(f"Prompt: {prompt_format}\n")
        outfile.write(f"#################################################################################\n")
        for chinese_word in word_list:
            prompt = create_prompt(chinese_word)
            print(f"Processing {chinese_word}...")
            try:
                response_text = gemini_text_understanding(prompt)
                outfile.write(f"Word: {chinese_word}\n\n{response_text}\n\n")
                outfile.write(f"#################################################################################\n")
                # time.sleep(40)
            except Exception as e:
                print(f"Failed to process {chinese_word}: {e}")


if __name__ == "__main__":
    file_path = '../../scorer/answer_sheet_w_element.csv'
    word_list = get_word_list(file_path)
    output_path = '../text_answers/word_results_apr_8_2.txt'
    process_list(word_list, output_path)