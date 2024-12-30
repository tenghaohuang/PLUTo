
import os
import random
import sys
import dspy
import pickle
from IPython import embed
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives import Example
# from modules import ProceduralReasoning
from faiss_rm import FaissRM
from config import Config
from utils import load_tool_desc, load_data, preprocess, get_optimizee
from tqdm import tqdm
from DSP_functions import predict_api_useful_reason
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(Config)
api_des_dict = load_tool_desc(path=Config.DESCRIPTION_PATH)

# train_data = load_data(path=Config.TRAIN_DATA_PATH)
# dev_data = load_data(path=Config.DEV_DATA_PATH)
# test_data = load_data(path=Config.TEST_DATA_PATH)

# rag = ProceduralReasoning(api_dict=api_des_dict, max_hops=
rag = FaissRM(Config, k=5, api_dict=api_des_dict)
plan_lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=Config.OPENAI_API_KEY, temperature=0.9, n=5)
predict_lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=Config.OPENAI_API_KEY, temperature=0.3, n=1)


Predict_API_reason = dspy.Predict(predict_api_useful_reason)
@retry(wait=wait_fixed(5), stop=stop_after_attempt(5))
def get_useful_reason(query, API):
    return  Predict_API_reason(query=query, API=API +' - '+ api_des_dict[API]['description']).useful_reason

def optimize_toward_failed_queries():
    results = []
    d4opt = {i: [] for i in api_des_dict}
    for num, item in enumerate(tqdm(train_data)):
        if num % 100 == 0 and num>0:
            print('-------------------', num, '-------------------')
            pickle.dump(rag.data, open('queries4optimization/rag_'+str(num)+'_data.p', 'wb'))
            pickle.dump(results, open('retrieval_results/round2_retrieval_results_'+str(num)+'.p', 'wb'))

        query = item['query']
        answer = list(set([gt[0] for gt in item['gts']]))
        rt = rag(query)
        if Config.OPTIMIZE:

            for i in answer:
                if i not in rt.answer:
                    embed()
                    reason = get_useful_reason(query, i)
                    print('reason: ', reason)
                    d4opt[i].append({'query': query, 'reason': reason})
            optimizees = get_optimizee(d4opt)
            # optimize
            d4opt = rag.discrete_description_optimization(optimizees, d4opt)
        results.append({'query': query, 'answer': answer, 'rt': rt})
        print('query: ', query)
        print('Ground_truth: ', answer)
        print('rt: ', rt)
        print('------------------------------------------------------')

def optimize_toward_queries():
    Config.ERASE_OPTIMIZE_CACHE = False
    print(Config.ERASE_OPTIMIZE_CACHE)
    results = []
    d4opt = {i: [] for i in api_des_dict}
    api2q = pickle.load(open(Config.API2Q_DICT,"rb"))
    for num, api in enumerate(tqdm(api2q)):
        try:
            dspy.settings.configure(lm=predict_lm)
            current_items = api2q[api][:Config.OPTIMIZE_THRESHOLD]
            if num % 300 == 0 and num>0:
                print('-------------------', num, '-------------------')
                pickle.dump(rag, open('queries4optimization/rag_'+str(num)+'_data.p', 'wb'))

            for item in current_items:
                query = item['query']
                reason = get_useful_reason(query, api)

                print('reason: ', reason)
                d4opt[api].append({'query': query, 'reason': reason})
            dspy.settings.configure(lm=plan_lm)
            d4opt = rag.discrete_description_optimization([api], d4opt)
        except:
            continue

optimize_toward_queries()


