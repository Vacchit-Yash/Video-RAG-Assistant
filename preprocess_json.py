import requests
import os
import json
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

#####slow process taking one by one so changung it so embedings can be in bulk
# def create_embedding(text):
#     r = requests.post("http://localhost:11434/api/embeddings",json = {
#         "model" : "bge-m3",
#         "prompt" : text
#     })

#     embeddings = r.json()['embedding']

#     return embeddings



def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json = {
        "model" : "bge-m3",
        "input" : text_list
    })

    

    

    embeddings = r.json()['embeddings']

    return embeddings



jsons = sorted(os.listdir("newjsons") , key=lambda x:int(x.split("_")[0]))
my_dict = []
chunk_id =0
for js in jsons:
    with open(f"newjsons/{js}") as f:
        content = json.load(f)
    batch_size = 100

    for start in range(0, len(content['chunks']), batch_size):

        batch_chunks = content['chunks'][start:start + batch_size]

        embeddings = create_embedding([c['text'] for c in batch_chunks])

        for i, chunk in enumerate(batch_chunks):
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dict.append(chunk)
           # print(chunk)

# print(my_dict)

df = pd.DataFrame.from_records(my_dict)
#Save this DF
joblib.dump(df ,"embedding_df.joblib")





# a = create_embedding(["He is Tall" ,"He is Dumb"])
# print(a)