import re
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# get groundtruth answer
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
            if row['Pinyin Name'] in chinese_name_to_category.keys():
                chinese_name_to_category[row['Pinyin Name']].append(row['Category'])
            else:
                chinese_name_to_category[row['Pinyin Name']] = [row['Category']]
    
    return chinese_name_to_category


def read_doc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def extract_info(document):
    # This pattern matches both "A." and "Option B:" formats for the answers
    pattern = r"Image: (\S+_\d+\.(jpg|png|jpeg))\n\n([A-G]|Option\s[A-G])"
    matches = re.findall(pattern, document)
    extracted_info = {}
    for match in matches:
        image_name, _, option = match
        single_answer = re.search(r'Option (\w)', option)
        if single_answer:
            option = single_answer.group(1)
        extracted_info[image_name] = option
    return extracted_info


# read llm ouputs
def grade_answers(extracted_info, ground_truth):
    total = 0
    correct = 0
    distribution_dict = {}
    correct_dict = {}
    for image_name, chosen_option in extracted_info.items():
        plot_name = image_name.split("_")[0]
        if plot_name in ground_truth:
            correct_answer = []
            for idx in range(len(ground_truth[plot_name])):
                correct_answer.append(ground_truth[plot_name][idx].split(".")[0])
            if chosen_option in correct_answer:
                correct += 1
                if chosen_option in correct_dict.keys():
                    correct_dict[chosen_option] += 1
                else:
                    correct_dict[chosen_option] = 1
        for gt in correct_answer:
            if gt in distribution_dict.keys():
                distribution_dict[gt] += 1
            else:
                distribution_dict[gt] = 1
        total += 1
    acc = correct / total
    print(distribution_dict)
    for option in correct_dict.keys():
        correct_dict[option] = correct_dict[option] / distribution_dict[option]
    print(correct_dict)
    return correct, acc 

def grade_answer_confusion(llm_answer, ground_truth):
    wrong_dict = {}
    correct = 0
    for image_name, chosen_option in llm_answer.items():
        plot_name = image_name.split("_")[0]
        if plot_name in ground_truth:
            correct_answer = []
            for idx in range(len(ground_truth[plot_name])):
                correct_answer.append(ground_truth[plot_name][idx].split(".")[0])
            if chosen_option in correct_answer:
                correct += 1
            else:
                if plot_name in wrong_dict.keys():
                    wrong_dict[plot_name][1].append(chosen_option)
                else:
                    wrong_dict[plot_name] = [correct_answer,[chosen_option]]
    print(wrong_dict)
    return correct, wrong_dict

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
    plt.title("Confusion Matrix of Wrong Symbolic Matching Answers", fontsize=20, fontweight='bold')
    plt.xlabel("Responded Option", fontsize=14, fontweight='bold')
    plt.ylabel("True Option", fontsize=14, fontweight='bold')
    plt.savefig('heatmap_mc_gpt4o.png')
    return confusion_matrix

if __name__ == "__main__":
    answer_path = 'answer_sheet_w_element.csv'
    ground_truth = get_ground_truth(answer_path)
    doc_path = '../gpt4/multiple_choice_answers/gpt4o_mc_results_may14.txt'
    llm_answer = read_doc(doc_path)
    extract_answer = extract_info(llm_answer)
    correct, accuracy = grade_answers(extract_answer, ground_truth)
    print(accuracy)