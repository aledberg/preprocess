## Code to read the summaries and make sure that they are anonymized
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
from datetime import datetime
##from sentence_transformers import SentenceTransformer
##import torch
import gc
from tqdm import tqdm
current_date = datetime.now().strftime('%Y_%m_%d')


sumdat=pd.read_pickle("results_of_step1.pkl")

## use stanza for named entity recognition
## see https://stanfordnlp.github.io/stanza/
import stanza
## download support for swedish language 
##stanza.download("sv")
nlp = stanza.Pipeline("sv", processors="tokenize,ner")

## go through all cases and extract the names still present
##for s in tqdm(allsum[0:1000],"finding names"):
dnames=[]
for s in tqdm(sumdat,"finding names"):
    tmp=nlp(s)
    dnames.extend([ent.text for ent in tmp.ents if ent.type == "PER"])

dcc=Counter(dnames)

## The identified names are saved and reviewed off-line to decide which names to actually
## remove from the text. For example "Maria Ungdom" is not the name of a person but a
## care provider, so we want to keep that in the data
## real names
unique_names = [name for name, count in dcc.items() if len(name.split())>=1]
##unique_names = [name for name, count in dcc.items() if len(name.split())>=2]
print(unique_names)
with open('potential_names.txt', 'w', encoding='utf-8') as f:
    for name in unique_names:
        f.write(name + '\n')


## Edit these names "by hand" and load them again
with open('potential_names_edit.txt', 'r', encoding='utf-8') as f:
    edited_names = [line.strip() for line in f]


## replace the names with [NAME]
print(edited_names)
newsumdat=[]
for text in sumdat:
    for name in edited_names:
        if name in text:
            pattern = r'\b' + re.escape(name) + r'\b'
            text = re.sub(pattern, '[NAME]', text)
    newsumdat.append(text)
            

[i for i,x in enumerate(sumdat) if edited_names[1] in x]
print(newsumdat[1])

## To take out addresses we search for specific patterns: 
## Match e.g. "Storgatan 12", "Storgatan 12B", "Götgatan 77A", "Karlavägen 14"
address_pattern = r'\b[A-ZÅÄÖa-zåäöéÉ\-]+(?:gatan|vägen)\s\d+[A-Za-z]?\b'
rkn=0
newsumdat2=[]
for text in newsumdat:
    if re.search(address_pattern,text)!=None:
        text = re.sub(address_pattern, '[ADDRESS]', text)        
        newsumdat2.append(text)
        rkn+=1
    else:
        newsumdat2.append(text)

print(f"Replaced {rkn} addresses")

## need also to purge personal numbers from these summaries
import regex
## this is done in steps, first we look for any number of geneic personal
## number form and replace this
pattern = r"\b(?:\d{6}|\d{8})-\d{4}\b"
cleaned_text=[]
for i in range(len(sumdat)):
    summary=newsumdat2[i]
    mat=re.findall(pattern,summary)
    if len(mat)!=0 :
        print(i)
        summary=re.sub(mat[0],'[PIN-NUMBER]',summary)
    cleaned_text.append(summary)

## then we replace any occurenses of the complete personal number (there were none)
## this step builds on that we have access to the real personal number so it cannot be
## used for this example
# for i in range(len(sumdat)):
#     num=domdat['pnr'].iloc[i]
#     summary=cleaned_text[i]
#     pattern=pattern = fr"({regex.escape(num)}){{e<=3}}"
#     mat=regex.search(pattern,summary)
#     if mat!=None :
#         print(i)
#         restring=re.sub(r"[^0-9-]","",mat.group())
#         summary=re.sub(restring,'[PIN-NUMBER]',summary)
#     ##cleaned_text.append(summary)

## then we replace any occurenses of the first 8 digits YYYYMMDD
# cleaned_text2=[]
# for i in range(domdat.shape[0]):
#     num=domdat['pnr'].iloc[i][0:8]
#     summary=cleaned_text[i]
#     mat=re.findall(num,summary)
#     if len(mat)!=0 :
#         print(i)
#         summary=re.sub(mat[0][4:8],'[MMDD]',summary)
#     cleaned_text2.append(summary)
    
    
# ## then we replace any occurenses of the first 6 digits YYYYMMDD
# cleaned_text3=[]
# for i in range(domdat.shape[0]):
#     num=domdat['pnr'].iloc[i][2:8]
#     summary=cleaned_text2[i]
#     mat=re.findall(num,summary)
#     if len(mat)!=0 :
#         print(i)
#         summary=re.sub(mat[0][2:6],'[MMDD]',summary)
#     cleaned_text3.append(summary)
    

## now we should have no occurences of personal numbers left and can save these new data
## (cleaned_text3)

## finally we remove exact date references in the text
def mask_dates(text):
    # ISO-style: YYYY-MM-DD → YYYY-MM
    text = re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', r'\1-\2', text)

    # "7th of October 1993" → "October 1993"
    text = re.sub(
        r'\b\d{1,2}(st|nd|rd|th)?\s+of\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+(\d{4})\b',
        r'\2 \3',
        text,
        flags=re.IGNORECASE
    )

    # "7th February 1962" or "18 June 2015" → "February 1962"
    text = re.sub(
        r'\b\d{1,2}(st|nd|rd|th)?\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+(\d{4})\b',
        r'\2 \3',
        text,
        flags=re.IGNORECASE
    )

    # "November 30th, 2022" or "August 21, 2024" → "November 2022"
    text = re.sub(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'\d{1,2}(st|nd|rd|th)?(,)?\s+(\d{4})\b',
        r'\1 \4',
        text,
        flags=re.IGNORECASE
    )

    # "April 21st" (no year) → "April"
    text = re.sub(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'\d{1,2}(st|nd|rd|th)?\b',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    return text


print(cleaned_text[78])
print(mask_dates(cleaned_text[78]))

cdat=[mask_dates(x) for x in cleaned_text]

for i in range(len(cleaned_text)):
    print(cleaned_text[i]==cdat[i])

## check that it worked
idx=10
print(cleaned_text[idx])
print(cdat[idx])

## 
with open("results_of_step2.pkl", "wb") as f:
    pickle.dump(cdat, f)
