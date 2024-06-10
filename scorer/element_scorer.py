import re
import pandas
import nltk
from nltk.stem import WordNetLemmatizer
from pattern.en import singularize
from sentence_transformers import SentenceTransformer, util

# get groundtruth answer
def get_ground_truth(answer_path):
    df = pandas.read_csv(answer_path, header=None)
    df.dropna()
    answer_dict = {}
    for idx in range(len(df[0])):
        answer_idx = []
        for j in range(2, len(df.iloc[idx].dropna())-1):
            answer_idx.append(df.iloc[idx][j+1].lower().rstrip())
        answer_dict[df[1][idx]] = answer_idx
    del answer_dict['Pinyin Name']
    return answer_dict

def read_doc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def to_singular(words):
    # Convert to lowercase
    words = words.lower()
    
    # Singularize each word if necessary and rebuild the phrase
    # Note: This simplistic approach singularizes each word, which might not always be contextually accurate
    # for complex phrases. Adjustments may be needed for specific cases.
    lemmatizer = WordNetLemmatizer()
    singular_words = [lemmatizer.lemmatize(word, pos='n') for word in words.split()]
    
    # Rejoin the words into a phrase
    return ' '.join(singular_words)
    
def extract_info(document):
    pattern = r"Image: (\S+_\d+\.(jpg|png|jpeg))\n\n([^\n]+)"
    matches = re.findall(pattern, document)
    extracted_info = {}
    for match in matches:
        image_name, _, elements = match
        elements = elements.split(", ")
        elements = [element.replace('.', '') for element in elements]
        elements = [element.replace('(', '') for element in elements]
        elements = [element.replace(')', '') for element in elements]
        singular_nouns = [to_singular(noun) for noun in elements]
        extracted_info[image_name] = singular_nouns
    return extracted_info


def similarity_score_answers(extracted_info, ground_truth):
    total = len(extracted_info)
    score = []
    wrong_zero_dict = {}
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cuda")
    for image_name, result_elements in extracted_info.items():
        plot_name = image_name.split("_")[0]
        if plot_name in ground_truth:
            sim_score = []
            correct_elements = ground_truth[plot_name]
            for true_element in correct_elements:
                embedding_1= model.encode(true_element, convert_to_tensor=True)
                buffer_score = 0
                for answer_element in result_elements:
                    embedding_2 = model.encode(answer_element, convert_to_tensor=True)
                    similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2)
                    if similarity_score > buffer_score:
                        buffer_score = similarity_score
                sim_score.append(buffer_score)
            score.append(sum(sim_score) / len(sim_score))
            if sum(sim_score) / len(sim_score) < 0.5:
                if plot_name in wrong_zero_dict.keys():
                    wrong_zero_dict[plot_name] += 1
                else:
                    wrong_zero_dict[plot_name] = 1
    avg_score = sum(score) / len(score)
    # print(wrong_zero_dict)
    return avg_score



# read llm ouputs
def grade_exact_answers(extracted_info, ground_truth):
    score = 0
    count = 0
    wrong_zero_dict = {}
    for image_name, elements in extracted_info.items():
        correct = 0
        plot_name = image_name.split("_")[0]
        if plot_name in ground_truth:
            count += 1
            correct_elements = ground_truth[plot_name]
            answer_number = len(correct_elements)
            for true_element in correct_elements:
                for answer_element in elements:
                    if true_element in answer_element:
                        correct = correct + 1
        if correct == 0:
            if plot_name in wrong_zero_dict.keys():
                wrong_zero_dict[plot_name] += 1
            else:
                wrong_zero_dict[plot_name] = 1
        score += correct / answer_number
    avg_score = score/count
    print(count)
    # print(wrong_zero_dict)
    return avg_score

def answer_stat(extracted_info, ground_truth):
    score = 0
    count = 0
    wrong_zero_dict = {}
    appear_dict = {}
    for image_name, elements in extracted_info.items():
        correct = 0
        plot_name = image_name.split("_")[0]
        if plot_name in ground_truth:
            count += 1
            correct_elements = ground_truth[plot_name]
            answer_number = len(correct_elements)
            for true_element in correct_elements:
                for answer_element in elements:
                    if true_element in answer_element:
                        correct = correct + 1
                    else:
                        if answer_element in appear_dict.keys():
                            appear_dict[answer_element] += 1
                        else:
                            appear_dict[answer_element] = 1
        if correct == 0:
            if plot_name in wrong_zero_dict.keys():
                wrong_zero_dict[plot_name] += 1
            else:
                wrong_zero_dict[plot_name] = 1
        score += correct / answer_number
    avg_score = score/count
    # Get the top 20 value-key pairs
    top_20_pairs = sorted(appear_dict.items(), key=lambda item: item[1], reverse=True)[:20]
    
    # Print the top 20 pairs
    for pair in top_20_pairs:
        print(pair)
    # print(wrong_zero_dict)
    return avg_score



if __name__ == "__main__":
    answer_path = 'answer_sheet_w_element.csv'
    ground_truth = get_ground_truth(answer_path)
    doc_path = '../gpt4/element_answers/gpt4o_element_results_may_14.txt'
    llm_answer = read_doc(doc_path)
    extract_answer = extract_info(llm_answer)
    score = similarity_score_answers(extract_answer, ground_truth)
    print(score)
    absolute_score = answer_stat(extract_answer, ground_truth)
    print(absolute_score)