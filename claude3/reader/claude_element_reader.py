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
    api_key="please enter your api keys",
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
    model = 'claude-3-haiku-20240307' # claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    output_name = model + '_element_results_may_9.txt'
    output_path = '../text_answers/' + output_name
    prompt = "Please analyze the provided image carefully to identify key visual elements. Focus on components that traditionally have symbolic meaning in the cultural context from which the artwork originates.\
    Look for elements that might represent ideas, virtues, or wishes, especially those commonly found in nature or historical motifs.\
    For instance, in Chinese culture, certain animals and plants are known to symbolize specific messages when depicted in art. \
    Based on these principles, identify the primary visual elements in the image that are likely used to convey a message or a wish.\
    Please list the discernible elements present in the image, excluding any assumptions about elements not clearly visible.\
    Pleas answer the question in one line with the following format strictly: name of element A, name of element B, etc"
    process_folder(model, prompt, folder_path, output_path)
