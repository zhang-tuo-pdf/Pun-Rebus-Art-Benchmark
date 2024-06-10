import re
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_ground_truth(file_path):
    # Initialize an empty dictionary to store your data
    chinese_name_to_category = {}

    # Open the CSV file and read its contents
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        reader = csv.DictReader(csvfile)
        
        # Iterate over each row in the CSV
        for row in reader:
            # Use 'Chinese Name' as the key and 'Category' as the value
            if row['Chinese Name'] in chinese_name_to_category.keys():
                chinese_name_to_category[row['Chinese Name']].append(row['Category'])
            else:
                chinese_name_to_category[row['Chinese Name']] = [row['Category']]
    
    return chinese_name_to_category

def read_doc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_info(document):
    pattern = r"Word: (.+?)\n\n([A-G])"
    matches = re.findall(pattern, document)
    extracted_info = {}
    for match in matches:
        image_name, option = match
        # if image_name in extracted_info.keys():
        #     extracted_info[image_name].append(option)
        # else:
        #     extracted_info[image_name] = [option]
        extracted_info[image_name] = option
    return extracted_info

def grade_answers(llm_answer, ground_truth):
    total = len(ground_truth)
    correct = 0
    wrong_dict = {}
    for word, option in llm_answer.items():
        if option in ground_truth[word]:
            correct = correct + 1
        else:
            if word in wrong_dict.keys():
                wrong_dict[word][1].append(option)
            else:
                wrong_dict[word] = [ground_truth[word],[option]]
    acc = correct / total
    return acc, wrong_dict

def option_acc_report(wrong_dict, ground_truth):
    element_total = {}
    element_wrong = {}
    element_accuracy = {}
    for word, option in ground_truth.items():
        for choice in option:
            if choice in element_total:
                element_total[choice] += 1
            else:
                element_total[choice] = 1
    for word, options in wrong_dict.items():
        for choice in options[0]:
            if choice in element_wrong:
                element_wrong[choice] += 1
            else:
                element_wrong[choice] = 1
    for option, numbers in element_wrong.items():
        total_number = element_total[option]
        element_accuracy[option] = total_number - element_wrong[option]
        element_accuracy[option] = element_accuracy[option] / total_number
    return element_total, element_wrong, element_accuracy

def vis_confusion_matrix(wrong_dict):
    options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    confusion_matrix = pd.DataFrame(np.zeros((len(options), len(options))), index=options, columns=options)
    for true, responded in wrong_dict.values():
        for t in true:
            for r in responded:
                confusion_matrix.at[t, r] += 1
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt=".0f")
    plt.title("Confusion Matrix of Wrong Answers")
    plt.xlabel("Responded Option")
    plt.ylabel("True Option")
    plt.savefig('heatmap_2.png')
    return confusion_matrix



if __name__ == "__main__":
    file_path = "answer_sheet_w_element.csv"
    answer_dict = get_ground_truth(file_path)
    answer_file = "../gpt4/text_answers/gpt4o_word_results_may_13.txt"
    llm_answer = read_doc(answer_file)
    extract_answer = extract_info(llm_answer)
    acc, wrong_dict = grade_answers(extract_answer, answer_dict)
    print(wrong_dict)
    element_total, element_wrong, element_accuracy = option_acc_report(wrong_dict, answer_dict)
    print(element_total, element_wrong, element_accuracy)
    print(wrong_dict)
    conf = vis_confusion_matrix(wrong_dict)