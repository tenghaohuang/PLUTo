import dspy
import faiss
import numpy as np
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from os import path
from IPython import embed
import os
from config import Config
class FaissRM(dspy.Retrieve):
    """
        A class that uses Faiss to retrieve the top passages for a given query.
    """
    def __init__(self, config, api_dict, k=5):
        super().__init__(k=k)
        self.api_dict = api_dict
        self.topk = k
        self.query_encoder = SentenceTransformer(config.DEFAULT_QUERY_ENCODER)
        self.passage_encoder = SentenceTransformer(config.DEFAULT_PASSAGE_ENCODER)
        if config.LM_SCORING == 'True':
            self.language_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")     
        self.id2api = {id: k for id, k in enumerate(api_dict.keys())}
        self.api2id = {k: id for id, k in enumerate(api_dict.keys())}
        if config.RETRIEVE_WITH_USECASE == 'True':
            self.retrieve_with_usecase = True
        else:
            self.retrieve_with_usecase = False
        if config.LM_SCORING == 'True':
            self.lm_scoring = True
        else:
            self.lm_scoring = False
        
        index_file = config.DESCRIPTION_FILE.split('.')[0]+'.index'
        if path.exists(config.DESCRIPTION_INDEX_PATH+index_file):
            self.desc_index = faiss.read_index(config.DESCRIPTION_INDEX_PATH+index_file)
        else:
            # TODO: remove hard coded
            if config.PASSAGE_ENCODER == "intfloat/e5-large-v2":
                descriptions = ['passage: '+k+' [SEP] '+v['description'] if v['description'] else 'passage: '+k+' [SEP] '+' ' for k, v in api_dict.items()]

            print("... Creating description index")
            if Config.PASSAGE_ENCODER == "intfloat/e5-large-v2":
                descriptions = ["passage: "+v['description'] for v in api_dict.values()]
            else:
                descriptions = [k+' [SEP] '+v['description'] if v['description'] else k+' [SEP] '+' ' for k, v in api_dict.items()]
            desc_embeds = self.passage_encoder.encode(descriptions,  show_progress_bar=False)
            self.desc_index = faiss.IndexIDMap(faiss.IndexFlatIP(desc_embeds.shape[1]))
            self.desc_index.add_with_ids(desc_embeds, np.array(range(desc_embeds.shape[0])))

            faiss.write_index(self.desc_index, config.DESCRIPTION_INDEX_PATH+index_file)
        if self.retrieve_with_usecase:
            if path.exists(config.USECASE_INDEX_PATH):
                self.usecase_index = faiss.read_index(config.USECASE_INDEX_PATH)
            else:
                usecases, usecase_ids = [], []
                for id, v in enumerate(api_dict.values()):
                    if 'usecase' in v:
                        # limit the number of usecase per tool to 10
                        for usecase in v['usecase'][:10]:
                            if config.DEFAULT_PASSAGE_ENCODER == "intfloat/e5-large-v2":
                                usecase = "passage: "+usecase                        
                            usecases.append(usecase)
                            usecase_ids.append(id)
                usecase_embeds = self.passage_encoder.encode(usecases,  show_progress_bar=False)
                self.usecase_index = faiss.IndexIDMap(faiss.IndexFlatIP(usecase_embeds.shape[1]))
                self.usecase_index.add_with_ids(usecase_embeds, np.array(usecase_ids))
                faiss.write_index(self.usecase_index, config.USECASE_INDEX_PATH)
            faiss.write_index(self.desc_index, Config.DESCRIPTION_INDEX_PATH)
        if path.exists(Config.USECASE_INDEX_PATH):
            self.usecase_index = faiss.read_index(Config.USECASE_INDEX_PATH)
        else:
            print("... Creating usecase index")
            usecases, usecase_ids = [], []
            for id, v in enumerate(api_dict.values()):
                if 'usecase' in v:
                    # limit the number of usecass per tool to 10
                    for usecase in v['usecase'][:10]:
                        if Config.PASSAGE_ENCODER == "intfloat/e5-large-v2":
                            usecase = "passage: "+usecase                        
                        usecases.append(usecase)
                        usecase_ids.append(id)
            usecase_embeds = self.passage_encoder.encode(usecases)
            self.usecase_index = faiss.IndexIDMap(faiss.IndexFlatIP(usecase_embeds.shape[1]))
            self.usecase_index.add_with_ids(usecase_embeds, np.array(usecase_ids))
            faiss.write_index(self.usecase_index, Config.USECASE_INDEX_PATH)


    def calculate_perplexity(self, query, desc):
        # adapted from https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixed-length-models
        text = f"{query} {desc}"
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.language_model.config.n_positions
        stride = 1024
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.language_model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl

    def update_api_des(self, api, description):
        # embed()
        id = np.array([self.api2id[api]])
        updated_desc_embed = self.passage_encoder.encode([description], show_progress_bar=False)
        self.desc_index.remove_ids(id)
        self.desc_index.add_with_ids(updated_desc_embed, id)

    def min_max_scaling(self, score, min_score, max_score, a, b):
        return ((score - min_score)*(b-a) / (max_score - min_score)) + a

    def forward(self, query):
        query_vec = self.query_encoder.encode([query], show_progress_bar=False)

        desc_score, desc_id = self.desc_index.search(query_vec, k=self.k)
        desc_matching = Counter({id: score for id, score in zip(desc_id.tolist()[0], desc_score.tolist()[0])})
        if self.retrieve_with_usecase:
            usecase_score, usecase_id = self.usecase_index.search(query_vec, k=self.k)
            # Combine desc and usecase scores. Weight 1:1
            usecase_matching = Counter({id: score for id, score in sorted(zip(usecase_id.tolist()[0], usecase_score.tolist()[0]), 
                                                                key=lambda item: item[1])})
            rt_score = desc_matching + usecase_matching
        else:
            rt_score = desc_matching
        # normalize rt scores to 0.5 ~ 1.0
        max_score, min_score = max(rt_score.values()), min(rt_score.values())
        rt_score = {id: self.min_max_scaling(score, min_score, max_score, 0, 1.0) for id, score in rt_score.items()}
        #print("Retriever Score:", sorted(rt_score.items(), key=lambda item: item[1], reverse=True))
        if self.lm_scoring == 'True':
            lm_score = dict()
            for id in rt_score.keys():

                    ppl = self.calculate_perplexity(queries[0], self.api_dict[self.id2api[id]]['description'])
                    lm_score[id] = ppl.item()
            # normalize lm scores to 0.5 ~ 1.0
            max_score, min_score = max(lm_score.values()), min(lm_score.values())
            lm_score = {id: self.min_max_scaling(score, min_score, max_score, 0, 1.0) for id, score in lm_score.items()}
        #print("LM Score:",sorted(lm_score.items(), key=lambda item: item[1]))
            total_score = {id: rt_score[id] - lm_score[id] for id in rt_score.keys()}
        else:
            total_score = rt_score
        #print("Total Score: ",sorted(total_score.items(), key=lambda item: item[1], reverse=True))
        passages = [self.id2api[id] for id, _ in sorted(total_score.items(), key=lambda item: item[1], reverse=True)[:self.topk]]
        return dspy.Prediction(passages=passages)
