from transformers import * # library for NLP
import torch
import numpy as np
import math
import pandas as pd

model_gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')  #pretrained model of Open-AI GPT
model_gpt.eval()
tokenizer_gpt = OpenAIGPTTokenizer.from_pretrained('openai-gpt') #pretrained tokenizer of Open-AI GPT

def perp_score(sentence):                                       # this function calculates which sentence is against common sense
    tokenize_input = tokenizer_gpt.tokenize(sentence)           #tokenize sentence to be passed to tensor
    tensor_input = torch.tensor([tokenizer_gpt.convert_tokens_to_ids(tokenize_input)])  #convert to tensor
    output = model_gpt(tensor_input, labels=tensor_input)           #predict sentence using the Open-AI GPT model
    return math.exp(output.loss)

X = []
y = []

df = pd.read_csv("task1.csv")               # read test file
for index, row in df.iterrows():
    X.append([row['sent0'], row['sent1']])  # append sentences in X
    y.append(row['id'])                     # append ids in Y

X = np.array(X)
y = np.array(y)


out = []
for i, sentences in enumerate(X):
    pred = [perp_score(i) for i in sentences]   #run on each sentence from test file

    if (pred[0] > pred[1]):         # check if sentence 0 is againstcommon sense or not
        out.append([y[i], '0'])
    else:
        out.append([y[i], '1'])

out = np.array(out)

pred_df = pd.DataFrame(out, columns=['id', 'sent'])     # output results into dataframes
pred_df.to_csv('subtaskA_answers.csv', header=False, index=False)       # write to file