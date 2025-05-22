## Code to read the anonymized summaries and use an embedding model to embedd these data
##
## The code calls a specific embedding model via sentence-transformers,
## to install sentence transformers please see https://sbert.net/docs/installation.html

## Anders Ledberg, 2025 05 15
## anders.ledberg@gmail.com
import numpy as np
import pandas as pd
import re
from collections import Counter
import pickle
import glob
from tqdm import tqdm
import time
from sentence_transformers import SentenceTransformer

## read these summaries
with open("results_of_step2.pkl", "rb") as f:
    summaries=pickle.load(f)

## this was as the time of writing the best model on the MTEB leaderboard that fit on my GPU
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

## do the embedding
start=time.time()
emb=model.encode(summaries)
stop=time.time()
print(f"duration: {stop-start}")

## add the embedding to the summaries
encdf=pd.DataFrame(emb,columns=[f"embeddings_{i}" for i in range(emb[0].shape[0])])

alldat=pd.concat([pd.DataFrame(summaries,columns=["summaries"]),encdf],axis=1)

## save the output in a convenient format 
alldat.to_parquet("results_of_step3.parquet")
