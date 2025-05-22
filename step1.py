## Code to read the data and call a LLM to make summaries 
##
## This code depends on a number of libraries and
## interterface the LLM through ollama (ollama.com)
## For installation of LLMs via Ollama and the Ollama Python
## library please see https://ollama.com/ and
## https://github.com/ollama/ollama-python


## Anders Ledberg, 2025 05 15
## anders.ledberg@gmail.com
import numpy as np
import pandas as pd
import re
from collections import Counter
import glob
from datetime import datetime
from tqdm import tqdm
import time
import pickle

## use LLMs via ollama
from ollama import chat
from ollama import generate
from ollama import ChatResponse
from ollama import embed

## In the manuscript we analyze text files from Sweden's administrative courts.
## These files contains sensitive personal information, here we instead use openly available
## text files. These files were taken from consultation responses (remisssvar) related to
## SOU 2023:62, a Swedish Governement Official Report from the Drug Commission of Inquiry.
## The data can be found here: https://www.regeringen.se/remisser/2024/01/remiss-sou-202362-vi-kan-battre-kunskapsbaserad-narkotikapolitik-med-liv-och-halsa-i-fokus/

## These text have a radically different content compared to the court descisions analysed in the
## manuscript, but they also have some similarities: they are in Swedish, are very heterogenous,
## and contain names of persons that we will try to remove.

## get the name of the files, assuming all text-files in this directory are related to the project
path="../remiss/*txt"
flist=sorted(glob.glob(path))

def read_files(filenames):
    texts = []
    for file in filenames:
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

textlist=read_files(flist)

##############################################3
## here we define the model to be used, to install it on your machine
## type: (ollama pull <model name>)

## this particular model fits on my A5000 GPU
model_name="mistral-small"

## system prompt, note that for the models to see this we need to set
## the context length to be long enough (the num_ctx parameter). You can also
## try experimenting with the temperature 

system_prompt = """You will be given a Swedish text that is a consultation response (remissvar) to the Swedish Government Official Report SOU 2023:62, produced by the Drug Commission of Inquiry (Narkotikautredningen). Your task is to anonymize and summarize the text. Anonymize by removing or replacing any personally identifiable information (e.g., names of individuals, email addresses, phone numbers), while preserving the name of the organization or institution that submitted the response. Summarize the content in English. The summary should clearly state:\n - Which organization or institution submitted the response.\n - Whether the organization is generally positive, negative, or neutral toward the conclusions or proposals in SOU 2023:62.\n- The main arguments or reasons for their position (e.g., legal, ethical, public health, economic, or societal considerations). \nThe output should be written in a clear and professional style."""

start=time.time()
summaries=[]
for i in tqdm(range(len(textlist)),"summarizing data"):
##for i in range(2):
    response: ChatResponse = generate(model=model_name, prompt=textlist[i],system=system_prompt,options={"num_ctx":32768,"stream":False, "temperature": 0.1})
    summaries.append(response['response'])

end=time.time()
print(f"Duration: {end-start}")


## save these summaries
with open("results_of_step1.pkl", "wb") as f:
    pickle.dump(summaries, f)


## some stats on the lenght of the texts    
len_orig=np.array([len(s) for s in textlist])
len_sum=np.array([len(s) for s in summaries])

print(f"orignal texts: mean characters {round(np.mean(len_orig),1)}, CV {round(np.std(len_orig)/np.mean(len_orig),2)}")
print(f"summaries: mean characters {round(np.mean(len_sum),1)}, CV {round(np.std(len_sum)/np.mean(len_sum),2)}")





