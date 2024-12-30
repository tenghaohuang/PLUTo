import time
import os

import dspy
import random
from DSP_functions import *

from utils import concatenate_corpus


from utils import load_tool_desc, concatenate_corpus, calculate_matches, normalize_text

from faiss_rm import FaissRM
from finetuned_faiss_rm import FinetunedFaissRM
from IPython import embed
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config
from sklearn.cluster import KMeans
import random as rand
import sys
from utils import load_tool_desc
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed)

def concatenate_corpus(data):
    corpus = ""
    for entry in data:
        corpus += "=== QUERY ===\n"
        corpus += entry['query'] + "\n"
        corpus += "=== REASON ===\n"
        corpus += entry['reason'] + "\n\n"
    return corpus
# Usage:


# api_des_dict = load_api_data()

api_des_dict = load_tool_desc(path=Config.DESCRIPTION_PATH)
retriever = FaissRM(k=5, api_dict=api_des_dict)
plan_lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=Config.OPENAI_API_KEY, temperature=0.9, n=5)
predict_lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=Config.OPENAI_API_KEY, temperature=0.3, n=1)


class OneShot(dspy.Module):
    def __init__(self, config, api_dict):
        super().__init__()
        self.data = api_dict
        if config.DPR_PATH.split('/')[0] == 'finetune':
            self.retriever = FinetunedFaissRM(config, k=10, api_dict=api_dict)
        else:
            self.retriever = FaissRM(config, k=10, api_dict=api_dict)
        self.predict_apis = dspy.Predict(PredictAPIs_OneShot)
        if config.MODEL_NAME == 'gpt-3.5-turbo':
            lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=config.OPENAI_API_KEY, request_timeout=30)
        elif config.MODEL_NAME == 'meta-llama/Llama-2-7b-chat-hf':
            lm = dspy.Llama(model='meta-llama/Llama-2-7b-chat-hf', hf_device_map='auto')
        elif config.MODEL_NAME == 'meta-llama/Llama-2-13b-chat-hf':
            lm = dspy.Llama(model='meta-llama/Llama-2-13b-chat-hf', hf_device_map='auto')
        dspy.settings.configure(lm=lm)
        
    def concat_description(self, api):
        return api + ' : ' + self.data[api]['description']

    def process(self, retrieved_apis):
        candidate_apis = ""
        retrieved_apis = [i for i in retrieved_apis if i in self.data.keys()]
        for idx, api in enumerate(retrieved_apis):
            candidate_apis += str(idx+1) +  ". " + self.concat_description(api) + '\n'
        return candidate_apis
        
    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    def get_predict_api_result(self, retrieved_apis, user_query, config):
        return self.predict_apis(candidate_apis=self.process(retrieved_apis),
                          user_query=user_query, config=config).predicted_apis

    def forward(self, user_request):
        retrieved_apis = self.retriever(user_request).passages
        predicted_apis = self.get_predict_api_result(retrieved_apis, user_request, config={'temperature':0.1, 'n':1})
        predicted_apis = [retrieved_api for retrieved_api in retrieved_apis if retrieved_api in predicted_apis]
        # predicted_apis = predicted_apis.split(',')
        # predicted_apis = [api.strip() for api in predicted_apis]
        # predicted_apis = [api for api in predicted_apis if api in self.data]
        return dspy.Prediction(answer=predicted_apis)

class ProceduralReasoning(dspy.Module):
    def __init__(self, config, api_dict, max_hops=3, num_plans=5):
        super().__init__()
        self.data = api_dict
        if config.DPR_PATH.split('/')[0] == 'finetune':
            self.retriever = FinetunedFaissRM(config, k=5, api_dict=api_dict)
        else:
            self.retriever = FaissRM(config, k=5, api_dict=api_dict)
        self.plan_config = {'temperature':0.7, 'n':5}
        self.predict_config = {'temperature':0.1, 'n':1}
        if config.MODEL_NAME == 'gpt-3.5-turbo':
            lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=config.OPENAI_API_KEY, request_timeout=30)
        elif config.MODEL_NAME == 'meta-llama/Llama-2-7b-chat-hf':
            lm = dspy.Llama(model='meta-llama/Llama-2-7b-chat-hf', hf_device_map='auto')
        elif config.MODEL_NAME == 'meta-llama/Llama-2-13b-chat-hf':
            lm = dspy.Llama(model='meta-llama/Llama-2-13b-chat-hf', hf_device_map='auto')
        dspy.settings.configure(lm=lm)
        self.lm = lm
        self.max_hops = max_hops
        self.num_plans = num_plans
        self.first_planning = dspy.ChainOfThought(FirstPlanning)
        self.latter_planning = dspy.ChainOfThought(LatterPlanning)
        self.predict_apis = dspy.Predict(PredictAPIs) #TODO: ponder the choice of predict vs. chain of thought
        self.description_optimization = dspy.Predict(DescriptionOptimization_new)
        self.d4opt = {}
        for i in self.data:
            self.d4opt[i] = []
        # implement a data structure to store the history of optimization
        if config.OPTIMIZE:
            self.opt_history = pickle.load(open('opt_history.p', 'rb'))
            # self.opt_history = {}
            # for i in tqdm(self.data):
            #     self.opt_history[i] = [{'description': self.data[i]['description'],\
            #                            'mrr': self.compute_usecase_mrr(i)}]
            # pickle.dump(self.opt_history, open('opt_history.p', 'wb'))
        print('initialization finished')



    def concat_description(self, api):
        if isinstance(self.data[api], str):
            return api + ' : ' + self.data[api]['description']
        else:
            return api + ' : ' + api

    def process(self, retrieved_apis):
        candidate_apis = ""
        retrieved_apis = [i for i in retrieved_apis if i in self.data.keys()]
        for idx, api in enumerate(retrieved_apis):
            candidate_apis += str(idx+1) +  ". " + self.concat_description(api) + '\n'
        return candidate_apis

    def compute_usecase_mrr(self,api):
        dev_mrr = 0
        # embed()
        if 'usecase' not in self.data[api]:
            return 0
        usecases = self.data[api]['usecase'][:5]
        # embed()
        for q_d in usecases:
            user_request = q_d
            pred_apis = self.retriever(user_request).passages
            dev_mrr += self.compute_mrr(api, pred_apis)
        return dev_mrr / 5

    def query_filtering(self, previous_queries, candidate_queries):
        if not previous_queries:
            return candidate_queries
        total_queries = candidate_queries + previous_queries
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(total_queries).toarray()
        clustering = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(feature_vectors)
        len_previous_queries = len(previous_queries)
        previous_queries_clusters = clustering.labels_[-len_previous_queries:]
        filtered_queries = []
        for idx, label in enumerate(clustering.labels_[:-len_previous_queries]):
            if label not in previous_queries_clusters:
                filtered_queries.append(candidate_queries[idx])
        if not filtered_queries:
            return candidate_queries
        return filtered_queries

    def log_history(self, lm):
        original_stdout = sys.stdout
        if os.path.exists('./logs.txt'):
            with open(f'./logs.txt', 'a') as f:
                sys.stdout = f
                print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
                print(lm.inspect_history(n=1))
                print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
                sys.stdout = original_stdout
        else:
            with open(f'./logs.txt', 'w') as f:
                sys.stdout = f
                print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
                print(lm.inspect_history(n=1))
                print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
                sys.stdout = original_stdout

    def log(self, message):
        pass
        #print(message)


    def update_opt_history(self, api, qs, description, mrr):
        self.opt_history[api].append({'description': description, 'q4opt': qs, 'mrr': mrr})

    def get_best_description(self, api):
        return max(self.opt_history[api], key=lambda x: x['mrr'])['description'],max(self.opt_history[api], key=lambda x: x['mrr'])['mrr'],

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(5))
    def discrete_description_optimization(self, optimizees, d4opt):

        optmization_history = {}
        optimized_des = ""
        total_attempts = 1

        for key in optimizees:
            optmization_history[key] = []
            opt_log = "N/A"
            # filter out the queries that are not useful
            d4opt[key] = [i for i in d4opt[key] if 'not' not in i['reason']]

            for round in range(1):
                current_goals = d4opt[key]
                current_goals = current_goals[:5]
                optimized_des = self.description_optimization(use_cases=concatenate_corpus(current_goals)).new_api_description


                usecase_mrr = self.compute_usecase_mrr(key)
                prev_best_des, prev_best_mrr = self.get_best_description(key)
                self.update_opt_history(key, current_goals, optimized_des, usecase_mrr)
                global_optimized_des, global_optimized_mrr = self.get_best_description(key)
                self.retriever.update_api_des(key, global_optimized_des)
                if global_optimized_mrr > prev_best_mrr:
                    print('previous best description: ', prev_best_des[:30])
                    print('previous best mrr: ', prev_best_mrr)
                    print('current best description: ', global_optimized_des[:30])
                    print('current best mrr: ', global_optimized_mrr)
                    self.data[key]['description'] = global_optimized_des
            if Config.ERASE_OPTIMIZE_CACHE:
                d4opt[key] = []
        return d4opt


    def compute_mrr(self,groundtruth, candidates):
        """
        Compute the MRR (Mean Reciprocal Rank).

        Parameters:
        - groundtruth: the correct item
        - candidates: a list of predicted items

        Returns:
        - MRR score
        """

        if groundtruth in candidates:
            rank = candidates.index(groundtruth) + 1
            return 1.0 / rank
        else:
            return 0.0

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    def get_predict_api_result(self, retrieved_apis, selected_query, config):
        return self.predict_apis(candidate_apis=self.process(retrieved_apis),
                          current_goal=selected_query, config=config).predicted_apis

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    def get_first_planning(self, user_request, config):
        return self.first_planning(user_request=user_request, config=config).completions.first_action

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    def get_latter_planning(self, user_request, previous_actions, config):
        return self.latter_planning(user_request=user_request,
                                             previous_actions=previous_actions, config=config).completions.next_action

    def forward(self, user_request):
        self.log("User Request: " + user_request)
        answer = []
        previous_queries = []
        previous_actions_str = ""
        for iter in range(self.max_hops):
            plans = []
            if iter == 0:
                #plans = self.get_first_planning(user_request, config=self.plan_config)
                plans = self.first_planning(user_request=user_request, config=self.plan_config).completions.first_action
                plans = [plan.split('\n')[0].lstrip("First Action:") if plan.startswith("First Action:") else plan.split('\n')[0] for plan in plans]
            else:
                #plans = self.get_latter_planning(user_request, previous_actions_str, config=self.plan_config)
                plans = self.latter_planning(user_request=user_request, previous_actions=previous_actions_str, config=self.plan_config).completions.next_action
                plans = [plan.split('\n')[0].lstrip("Next Action:") if plan.startswith("Next Action:") else plan.split('\n')[0] for plan in plans]
            #self.log_history(self.lm)
            candidate_queries = [plan for plan in plans if 'STOP' not in plan]
            if not candidate_queries:  # all plans are 'STOP', thus stop.
                self.log('All STOP triggered')
                break

            # query filtering
            # TODO: change temperature according to the filtering result
            # Right now, we cannot control the temperature on lm
            deduplicate_candidate_queries = list(set(candidate_queries))
            #self.log("Candidate queries: " + str(deduplicate_candidate_queries))
            filtered_queries = self.query_filtering(previous_queries, deduplicate_candidate_queries)

            num_stops = len(plans) - len(candidate_queries)
            if num_stops / (len(filtered_queries) + num_stops) > 0.5:  # decide whether to stop
                self.log("# of STOP larger than normal queries")
                break
            # There needs to be a Query Scoring module to select the most proming query.
            # Right now, we are just choosing a query randomly from the filtered_queries
            if len(filtered_queries) > 1:
                selected_query = random.choice(filtered_queries)
            else:
                selected_query = filtered_queries[0]
            previous_queries.append(selected_query)
            self.log(f"[{iter}] Selected Query: " + selected_query)
            retrieved_apis = self.retriever(selected_query).passages
            #predicted_apis = self.get_predict_api_result(retrieved_apis, selected_query, config=self.predict_config)
            predicted_apis = self.predict_apis(candidate_apis=self.process(retrieved_apis), current_goal=selected_query, config=self.predict_config).relevant_apis
            self.log(f"[{iter}] Raw Predicted APIs: "+ str(predicted_apis))
            #self.log_history(self.lm)
            if "NONE" in predicted_apis:
                self.log("None of the apis are predicted in this iteration")
            predicted_apis = [retrieved_api for retrieved_api in retrieved_apis if retrieved_api in predicted_apis]
            #predicted_apis = predicted_apis.split(',')
            #predicted_apis = [api.strip() for api in predicted_apis]
            #predicted_apis = [predicted_api for predicted_api in predicted_apis for retrieved_api in retrieved_apis if normalize_text(retrieved_api) == normalize_text(predicted_api)]
            self.log(f"[{iter}] Retrieved APIs: " + str(retrieved_apis))
            self.log(f"[{iter}] Predicted APIs: " + str(predicted_apis))
            answer.extend(predicted_apis)
            previous_actions_str += f"{iter + 1}. {selected_query}\n"
            self.log(f"[{iter}] Previous Actions: " + str(previous_actions_str))
        answer = list(set(answer))
        self.log("FINAL ANSWER: " + str(answer))
        return dspy.Prediction(answer=answer)

