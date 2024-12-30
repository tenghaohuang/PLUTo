import pickle
import numpy as np
from IPython import embed
from dspy.primitives import Example

import unicodedata
import re
import string
import os
import pandas as pd

def get_outfile(config):
    outfile = ""
    if config.MODEL_NAME == 'meta-llama/Llama-2-7b-chat-hf':
        outfile += '7b-'
    elif config.MODEL_NAME == 'meta-llama/Llama-2-13b-chat-hf':
        outfile += '13b-'
    elif config.MODEL_NAME == 'gpt-3.5-turbo':
        outfile += 'gpt-'
    outfile += config.DESCRIPTION_FILE.split('.')[0] + '-'
    if config.DPR_PATH.split('/')[0] == 'finetune':
        outfile += config.DPR_PATH.removeprefix('finetune/saved_models/').removesuffix('/') + '-'
    else:
        outfile += "default_dpr-"
    if config.RETRIEVE_WITH_USECASE == 'True':
        outfile += 'usecase-'
    if config.PROCEDURAL_REASONING == 'True':
        outfile += "pr"
    else:
        outfile += "simple"
    outfile += ".csv"
    return outfile

def compute_num_answers(outfile):
    count = 0
    with open(outfile,'r') as f:
        results = pd.read_csv(f)
        for answers in results['pred_answer']:
            if isinstance(answers, str):
                count += len(answers.split(','))
        count /= len(results)
    return count


from config import Config
# from modules import ProceduralReasoning

def load_tool_desc(path: str):
    data = pickle.load(open(path, 'rb'))
    # TODO: Remove hard coded path below
    if path == "descriptions/original_description.p":
        processed = {}
        for key, value in data.items():
            key = key.rstrip(' [SEP] ')
            description = value['description'].lstrip('> ').rstrip(' \n')
            try:
                processed[key] = {'description':description, 'usecase':value['usecase']}
            except:
                embed()
        data = processed
    return data

def load_data(path: str):
    files = os.listdir(path)
    data = []
    for file in files:
        data.extend(pickle.load(open(path+"/"+file, 'rb')))
    return data

def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))  

def preprocess_for_dspy(config, data):
    dataset = []
    for data in data:
        query = data['query']
        # TODO: Remove hard coded path below
        if config.TRAIN_DATA_PATH.split('/')[0] == "toolbench_v2":
            answer = list(set([gt[0] for gt in data['gts']]))
        else:
            answer = data['gts']
        dataset.append(Example({'user_request':query, 'answer':answer}))
    dataset = [x.with_inputs('user_request') for x in dataset]
    return dataset

def preprocess(config, data):
    dataset = []
    for data in data:
        query = data['query']
        # TODO: Remove hard coded path below
        if config.TRAIN_DATA_PATH.split('/')[0] == "toolbench_v2":
            answer = list(set([gt[0] for gt in data['gts']]))
        else:
            answer = data['gts']
        dataset.append({'query':query, 'gts':answer})
    return dataset

# define validation metric
def recall_metric(example, pred, trace=None):
    example_answer = set(example.answer)
    pred_answer = set(pred.answer)
    intersect = len(example_answer.intersection(pred_answer))
    recall = 0
    if intersect == 0 or len(example_answer) == 0:
        recall = 0
    else:
        recall = intersect/len(example_answer)
    return recall

# define validation metric
def precision_and_recall(example, pred, trace=None):
    example_answer = set(example.answer)
    pred_answer = set(pred.answer)
    intersect = len(example_answer.intersection(pred_answer))
    precision = 0
    recall = 0
    if intersect != 0:
        precision = intersect/len(pred_answer)
        recall = intersect/len(example_answer)
    return {'recall': recall, 'precision': precision}

examplar = """
Usecases:
1. 'Research reputable sources on the history of the Civil Rights movement in the United States.',
2. 'Provide a list of recommended family-friendly resorts in Hawaii and suggest activities to do with kids.',
3. "Provide the user with the list of academic papers and articles discussing criticisms and controversies around Jean-Luc Godard's works.",
4. "Research Akira Kurosawa's filmography and find reliable sources for critical analysis and reviews of his movies.",
5. 'Provide background information on the Western film genre and recommend famous films.'
------------------------------------------------------
API description: search - <search>' API for concise summaries on various topics from trusted sources, catering to diverse research needs.
score: 0.022435897435897433

API description: <search> find relevant information on a specific topic by searching Wikipedia articles. It allows users to explore related entities, expand their knowledge, and discover new topics of interest.
score: 0.1807692307692308
------------------------------------------------------
"""
api_des = '''<getWolframAlphaResults>: Get Wolfram|Alpha results using natural query.
; <get_name>:prints the possible 3 synonyms of the queried compound ID. 
; <get_allname>: prints all the possible synonyms (might be too many, use this function carefully).
; <get_id_by_struct>: prints the ID of the queried compound SMILES. This should only be used if smiles is provided or retrieved in the previous step. The input should not be a string, but a SMILES formula.
; <get_id>: prints the ID of the queried compound name, and prints the possible 5 names if the queried name can not been precisely matched.
; <get_prop>: prints the properties of the queried compound ID. 
; <search_places>: Run Places search.
; <create_file>: Create a pptx file with specific theme, available thems  green / orange / tech / wooden / flat.
; <get_image>: get_image(keywords str) -> str Get an image given comma seperated keywords, return the image path.
; <add_first_page>:Add the first page of ppt.
; <add_text_image_page>: Add a text page with one image. 
; <submit_file>: When all steps done, YOU MUST use submit_file() to submit your work in your topic name.
; <search>: The input is an exact entity name. The action will search this entity name on Wikipedia and returns the first five sentences if it exists. If not, it will return some related entities to search next.
; <lookup>: The input is a concise keyword. This action will look up in the current passage and return the next several sentences containing the keyword in current passage.
; <disambiguation>: The input is an entity name. This action will disambiguate this entity name to find other entities with similar names in Wikipedia.
; <search_literature>: Search for the given topic literatures in the database if the literature file is not provided. This will return the path of a pickle file recording literatures, and the number of literatures. 
; <split_criteria>: Split the screening requirements in the criteria of the literatures into a series of simple yes/no problems, and return the path of the splitted questions. 
; <literature_filter>: Check each literatures saved in the literature path according to the questions saved in the question path, and return the literatures that match the requirements. 
; <draw_table>: Extract the important elements of the literatures recorded in the given pickle file and return the path of table records. 
; <combine_table>: Combine several tables recorded in the table path into one comprehensive record table and return. give the literature path, table path and the exploring topic as the input.
; <generate_summary>: Given the exploring topic and the record table path of the literatures, this tool generates a paragraph of summary. 
; <print_literature>: given the literature path and number that are required to display, this tool returns the title and abstract of the literature.
; <print_tablefile>: given the table file path that are required to display, this tool reads the file and returns the string of the table.
; <get_weather_today>: gets the weather today for the specific given location.
; <forecast_weather>: Forecast weather in the upcoming number of days for the specific given location.
; <write_file>: write file to disk. 
; <read_file>: read file from disk and return the content.
; <convert_pickle>: Combine the file in txt directory into a dict and save in pickle file, which can be further processed and analyzed. 
; <search_top3>: Search key words, return top 3 search results. 
; <get_arxiv_article_information>: Run Arxiv search and get the article meta information.
; <get_distance>: Get the distance between two locations in miles.
; <get_coordinates>: Get the coordinates of a location.
; <get_route>: Get the route between two locations in miles.
; <search_nearby>: Search for places nearby a location, within a given radius, and return the results into a list.
; <get_translation>: Retrieves a translation for a given text in the specified target language.
; <get_database_schema>: Retrieve the schema of a specified database.
; <translate_nlp_to_sql>: A function that takes a natural language query as input and translates it into an equivalent SQL query, returning the SQL query as a string.
; <rewrite_sql>: Query is the output of translate_nlp_to_sql API. Rewrite the input query, which is inputted into the get_result_query_plan API.
; <get_result_query_plan>: get the result query plan of the input query. Query is the output of rewrite_sql API.
; <get_today_date>: Get today's date.
; <add_date>: Add days to a date.
; <get_daily_prices>: Get the stock price of an entity in the stock market.
; <get_open_info>: get information about if the market in the region is open.
; <get_exchange_rate>: This API returns the realtime exchange rate for a pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD).
'''
lines = api_des.split(";")

def reformat(line):
    tmp = ""
    for i in line:
        if i=="<" or i==">":
            continue
        if i==":":
            tmp = tmp.replace(' ', '')
            tmp+=" [SEP] "
        tmp+=i
    return tmp



def dot_products(v, vecs):
    """
    Compute the dot product of vector v against each vector in vecs.

    Parameters:
    - v: numpy array representing the vector.
    - vecs: list of numpy arrays representing the vectors.

    Returns:
    - List of dot products.
    """
    return [np.dot(v, vec) for vec in vecs]


def get_passages():
    passages = []
    for line in lines:
        p = reformat(line)
        passages.append(p)
    return passages

def calculate_matches(l1,l2):
    if len(l2) ==0:
        return None
    matches = 0
    for i in l1:
        if i in l2:
            matches+=1
    return min(matches/len(l2),1)

def top_k_indices(input_list, k):
    # Convert the list to a numpy array
    array = np.array(input_list)
    # Find the indices of the top-k values
    top_k_indices = array.argsort()[-k:][::-1]
    # Convert numpy array of indices back to list and return
    return top_k_indices.tolist()


def multiply_lists_elementwise(list1, list2):
    # Check if both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length for element-wise multiplication.")

    # Perform element-wise multiplication
    result = [list1[i] * list2[i] for i in range(len(list1))]
    return result

def add_lists_elementwise(list1, list2, alpha):
    # Check if both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length for element-wise multiplication.")

    # Perform element-wise multiplication
    result = [list1[i] + alpha*list2[i] for i in range(len(list1))]
    return result

def calculate_matches(l1,l2):
    if len(l2) ==0:
        return None
    matches = 0
    for i in l1:
        if i in l2:
            matches+=1
    return min(matches/len(l2),1)


def concatenate_corpus(data):
    corpus = ""
    for entry in data:
        corpus += "=== QUERY ===\n"
        corpus += entry['query'] + "\n"
        corpus += "=== REASON ===\n"
        corpus += entry['reason'] + "\n\n"
    return corpus

def get_optimizee(d4opt):
    optimizees = []
    for api in d4opt:
        if len(d4opt[api]) > Config.OPTIMIZE_THRESHOLD:
            optimizees.append(api)
    return optimizees