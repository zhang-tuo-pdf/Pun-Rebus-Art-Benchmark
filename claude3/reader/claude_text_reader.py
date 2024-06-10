import anthropic
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
    You must make a selection using the option above in your response. Your response should start with the chosen letter that best matches the word's meaning, followed by a precise and sound justification for your selection."
    return prompt

def claude_text_understanding(client, model, prompt):
    message = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response_text = message.content[0].text
    return response_text

def process_list(model, word_list, output_file_path):
    client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="please enter you api keys",
    )
    with open(output_file_path, "w") as outfile:
        prompt_format = create_prompt("xxxx")
        outfile.write(f"Prompt: {prompt_format}\n")
        outfile.write(f"#################################################################################\n")
        for chinese_word in word_list:
            prompt = create_prompt(chinese_word)
            print(f"Processing {chinese_word}...")
            try:
                response_text = claude_text_understanding(client, model, prompt)
                outfile.write(f"Word: {chinese_word}\n\n{response_text}\n\n")
                outfile.write(f"#################################################################################\n")
                # time.sleep(40)
            except Exception as e:
                print(f"Failed to process {chinese_word}: {e}")

if __name__ == "__main__":
    file_path = '../../scorer/text_ground_answer.csv'
    word_list = get_word_list(file_path)
    model = 'claude-3-haiku-20240307' # claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    output_name = model + '_word_results_apr_9.txt'
    output_path = '../text_answers/' + output_name
    process_list(model, word_list, output_path)