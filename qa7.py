# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:35:58 2021

@author: user
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import matplotlib.pyplot as plt
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

model.base_model.config

pytorch_total_params = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)    
pytorch_trainable_params = sum(p.numel() for p in model.base_model.parameters() )    
print("Total number of params", pytorch_total_params)
print("Total number of trainable params", pytorch_trainable_params)

text = r"""Humans are the most abundant and widespread species of primates, characterized by bipedality and large, complex brains enabling the development of advanced tools, culture and language. 
Humans are highly social beings and tend to live in complex social structures composed of many cooperating and competing groups, from families and kinship networks to political states. 
Social interactions between humans have established a wide variety of values, social norms, and rituals, which bolster human society. 
Curiosity and the human desire to understand and influence the environment and to explain and manipulate phenomena have motivated humanity's development of science, philosophy, mythology, religion, and other fields of knowledge.
Humans drink wine, bear, water. Humans eat bread, fruits.
"""

import numpy as np
def get_top_answers(possible_starts,possible_ends,input_ids):
  answers = []
  for start,end in zip(possible_starts,possible_ends):
    #+1 for end
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))
    answers.append( answer )
  return answers  

def answer_question(question,context,topN):

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    model_out = model(**inputs)
     
    answer_start_scores = model_out["start_logits"]
    answer_end_scores = model_out["end_logits"]

    possible_starts = np.argsort(answer_start_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    possible_ends = np.argsort(answer_end_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    
    #get best answer
    answer_start = torch.argmax(answer_start_scores)  
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    answers = get_top_answers(possible_starts,possible_ends,input_ids )

    return { "answer":answer,"answer_start":answer_start,"answer_end":answer_end,"input_ids":input_ids,
            "answer_start_scores":answer_start_scores,"answer_end_scores":answer_end_scores,"inputs":inputs,"answers":answers,
            "possible_starts":possible_starts,"possible_ends":possible_ends}

questions = [
    
    "What is the human ?",
    "Which is most successful primate ?",
    "What is the country with most population?",
    "What is the topic here?",
    "What are we talking about?",
    "What is the main idea here?",
    "Do humans eat ?",
    "What do they eat ? "
]



for q in questions:
  answer_map = answer_question(q,text,5)    
  print("Question:",q)
  print("Answers:")
  [print((index+1)," ) ",ans) for index,ans in  enumerate(answer_map["answers"]) if len(ans) > 0 ]
  
#answer_map = answer_question("Where is most populous in the world?",text,3)