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
    output_name = model + '_element_results_may_9.txt'
    output_file = "../element_answers/" + output_name
    prompt = "Please analyze the provided image carefully to identify key visual elements. Focus on components that traditionally have symbolic meaning in the cultural context from which the artwork originates.\
    Look for elements that might represent ideas, virtues, or wishes, especially those commonly found in nature or historical motifs.\
    For instance, in Chinese culture, certain animals and plants are known to symbolize specific messages when depicted in art. \
    Based on these principles, identify the primary visual elements in the image that are likely used to convey a message or a wish.\
    Please list the discernible elements present in the image, excluding any assumptions about elements not clearly visible.\
    Pleas answer the question in one line with the following format strictly: name of element A, name of element B, etc"
    process_folder(model, prompt, folder_path, output_file)