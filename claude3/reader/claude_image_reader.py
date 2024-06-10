import base64
import anthropic
import os
import mimetypes
import time

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


def claude_image_understanding(client, model, prompt, image_path):
    image_sr = encode_image(image_path)
    image_type = get_mime_type(image_path)
    message_list = [
        {
            "role": 'user',
            "content": [
                # {"type": "image", "source": {"type": "base64", "media_type": image_type, "data": image_sr}},
                {"type": "image", "source": {"type": "base64", "media_type": image_type, "data": image_sr}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=message_list
    )
    response_text = message.content[0].text
    return response_text

def process_folder(model, prompt, folder_path, output_file_path):
    client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="please enter you api keys",
    )
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
                response_text = claude_image_understanding(client, model, prompt, image_path)
                outfile.write(f"Image: {filename}\n\n{response_text}\n\n")
                outfile.write(f"#################################################################################\n")
                file_number += 1
                time.sleep(20)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                damage_file.append(filename)
            print(file_number)
    file = open('damage_items.txt','w')
    for image_path in damage_file:
        file.write(image_path+"\n")
    file.close()

if __name__ == "__main__":
    folder_path = "../../Pun Chinese Painting"
    model = 'claude-3-opus-20240229' # claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    output_name = model + '_word_results_apr_14.txt'
    output_path = '../text_answers/' + output_name
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
    process_folder(model, prompt, folder_path, output_path)