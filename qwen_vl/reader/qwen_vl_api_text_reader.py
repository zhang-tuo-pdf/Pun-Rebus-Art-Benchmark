from http import HTTPStatus
import dashscope
import pandas
import os 

def get_word_list(file_path):
    df = pandas.read_csv(file_path, header='infer')
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
    You must make a selection using the option above in your response. In you response, you should directly start with writing down the chosen captial letter that best matches the word's meaning, such as A. Longevity and Good Health or B. Happiness,Joy, Good Luck. \
    Please do not write any additional text in front of your chosen option. \
    After that, you could write a precise and sound justification for your selection in a new line. \
    Your answer should strictly follow this format"
    return prompt

def text_qwen_reading(model_name, prompt):
    """Simple single round multimodal conversation call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model=model_name,
                                                     messages=messages)

    return response.output.choices[0].message.content[0]["text"]

def process_list(model_name, word_list, output_file_path):
    with open(output_file_path, "w") as outfile:
        prompt_format = create_prompt("xxxx")
        outfile.write(f"Prompt: {prompt_format}\n")
        outfile.write(f"#################################################################################\n")
        for chinese_word in word_list:
            prompt = create_prompt(chinese_word)
            print(f"Processing {chinese_word}...")
            try:
                response_text = text_qwen_reading(model_name, prompt)
                outfile.write(f"Word: {chinese_word}\n\n{response_text}\n\n")
                outfile.write(f"#################################################################################\n")
                # time.sleep(40)
            except Exception as e:
                print(f"Failed to process {chinese_word}: {e}")

if __name__ == '__main__':
    model = 'qwen-vl-max' # 'qwen-vl-plus' or 'qwen-vl-max'
    dashscope.api_key = "please enter you api keys"
    file_path = '../../scorer/text_ground_answer.csv'
    word_list = get_word_list(file_path)
    output_name = model + '_word_results_apr_8.txt'
    output_path = '../text_answers/' + output_name
    process_list(model, word_list, output_path)