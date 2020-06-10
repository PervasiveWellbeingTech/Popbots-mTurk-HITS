import asyncio
import aiofiles
import aiohttp
from aiohttp.client import ClientTimeout
import pandas as pd
import json
import numpy as np
import time
from os import listdir

#from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('bert-base-nli-mean-tokens')
np.set_printoptions(threshold=np.inf,suppress=True)

BASE_URL = 'http://commuter.stanford.edu:9001'
BASE_PATH = '/commuter/PopBots/NLP/Popbots-mTurk-HITS/bert-pipeline/datasets/'
DATASET_NAME = '2020-04-29-MainTurkAggregation-5-Turkers_v0_Sorted'




DATA_COLUMN = 'Input.text'
LABEL_COLUMN_RAW = 'top_label'#'Answer.Label'

MAPPING_DICT = {'Other': 0, 'Everyday Decision Making': 1, 'Work': 2, 'Social Relationships': 3, 'Financial Problem': 4, 'Emotional Turmoil': 5, 'Health, Fatigue, or Physical Pain': 6, 'School': 7, 'Family Issues': 8,'Not Stressful':9}

SCRAPED_CATEGORIES= ['Everyday Decision Making', 'Social Relationships', 'Emotional Turmoil', 'Family Issues']


OUTPUT_PATH = './inquire_scraped_data/'

scraped_datasets = ["livejournal"]#,"reddit"]


START = time.monotonic()



def read_process_dataset():
    df = pd.read_csv(BASE_PATH+DATASET_NAME+'.csv',sep=",")
    df[LABEL_COLUMN_RAW] = df[LABEL_COLUMN_RAW].astype(str)
    df[DATA_COLUMN] = df[DATA_COLUMN].apply(lambda x: x.lower())
    df = df[df[LABEL_COLUMN_RAW].isin(SCRAPED_CATEGORIES)]

    #df_columns = ['category','mean_embedding', 'nb_sentences','source_sentences']
    df_columns = ['category' 'nb_sentences','source_sentences']

    bootstrapped_df = pd.DataFrame(columns=df_columns)

    boostrap_number = 5

    for category in SCRAPED_CATEGORIES:

        nb_sentences = 38
        all_mean = []
        for i in range(boostrap_number):
            category_df = df[df[LABEL_COLUMN_RAW] == category].sample(n=nb_sentences)
            #category_df_unsampled = df[df[LABEL_COLUMN_RAW] == category]
            
            #category_df['embedding'] = model.encode(category_df[DATA_COLUMN].values)
            #category_df['embedding'] = category_df['embedding'].apply(lambda x: np.array(x))
            #average_mean = np.mean(np.array(category_df['embedding'].values),axis=0) # vector of 768 dim
            
            bootstrapped_df=bootstrapped_df.append({'category':category,'nb_sentences':str(nb_sentences),'source_sentences':" ".join(list(category_df[DATA_COLUMN].values))},ignore_index=True)
            #all_mean.append(average_mean)
        
        
        #mean_all_mean = np.mean(np.array(all_mean),axis=0)
        category_df_unsampled = df[df[LABEL_COLUMN_RAW] == category]
        #print(" ".join(list(category_df_unsampled[DATA_COLUMN].values)))
        bootstrapped_df=bootstrapped_df.append({'category':"All "+category,'nb_sentences':str(len(category_df_unsampled)),'source_sentences':" ".join(list(category_df_unsampled[DATA_COLUMN].values))},ignore_index=True)

    return bootstrapped_df

def aggregate_json_convert_tocsv():
    final_df = pd.DataFrame()

    for category in SCRAPED_CATEGORIES:
        filenames_cat_list = []
        for path in listdir(OUTPUT_PATH):
            if category.replace(" ","_") in path:
                filenames_cat_list.append(path)
        with open(f'./final_data/json/w2v_{category}.json', 'w+') as outfile:
            result = []
            for fname in filenames_cat_list:
                print(fname)
                try:
                    with open(OUTPUT_PATH+fname,"rb") as infile:
                        result += json.load(infile)
                except Exception as error:
                    print(f"Error while trying to load {fname} with error: {error}") 
            json.dump(result,outfile)
        df = pd.read_json(f'./final_data/json/w2v_{category}.json',orient='records')
        df.drop_duplicates(subset='text', keep='first', inplace=True)

        final_df = pd.concat([df,final_df])
        df.to_csv(f'./final_data/csv/w2v_{category}.csv')




    final_df.to_csv(f'./final_data/csv/w2v_final_df.csv')





class RateLimiter:
    """Rate limits an HTTP client that would make get() and post() calls.
    Calls are rate-limited by host.
    https://quentin.pradet.me/blog/how-do-you-rate-limit-calls-with-aiohttp.html
    This class is not thread-safe."""
    RATE = 1  # one request per second
    MAX_TOKENS = 10

    def __init__(self, client):
        self.client = client
        self.tokens = self.MAX_TOKENS
        self.updated_at = time.monotonic()

    async def get(self, *args, **kwargs):
        await self.wait_for_token()
        now = time.monotonic() - START
        print(f'{now:.0f}s: ask {args[0]}')
        return self.client.get(*args, **kwargs)

    async def wait_for_token(self):
        while self.tokens < 1:
            self.add_new_tokens()
            await asyncio.sleep(0.1)
        self.tokens -= 1

    def add_new_tokens(self):
        now = time.monotonic()
        time_since_update = now - self.updated_at
        new_tokens = time_since_update * self.RATE
        if self.tokens + new_tokens >= 1:
            self.tokens = min(self.tokens + new_tokens, self.MAX_TOKENS)
            self.updated_at = now


async def get_stressors(session,dataset,stressor_index,category, stressor_sentence,embedding,nb_sentences):
    stressor_data = stressor_sentence
    print(stressor_data)
    output_data = []
    endpoint = '/query'
    params = {'data':stressor_data,'dataset': dataset,'maxWords':15,'minWords':4,'top':10,'percent':'0.01','model':'default'}#,'query_vector':str(list(embedding))}
    url = f'{BASE_URL}{endpoint}'
    #print(f'Getting {category} stressor for sentence {stressor_sentence}')
    
    timeout = ClientTimeout(total=0)  # `0` value to disable timeout
    async with await session.get(url, headers={}, params=params) as resp:
        #print(resp)
        #await asyncio.sleep(2)
        data = await resp.json()
        print(data)
        for elem in data['query_results']:
            json_line  = {"dataset":str(dataset),"category":str(category),"text":str(elem['sent_text']),"similarity":str(elem['similarity']),"stressor":str(stressor_data)}#,"source_embedding":str(list(embedding)),"nb_sentences":str(nb_sentences)}
            output_data.append(json_line)

    async with aiofiles.open(f'./inquire_scraped_data/{category.replace(" ", "_")}_{stressor_index}_{dataset}_w2v.json', 'a+') as file:
        await file.write(json.dumps(output_data))


    
        
async def main(df):
    for dataset in scraped_datasets:

        stressor_args = []    
        # Iterate over each row 
        for index, rows in df.iterrows(): 
            # Create list for the current row 
            l =[dataset, index, rows['category'],rows['source_sentences'],"NA",rows['nb_sentences']] 
           
            # append the list to the final list 
            stressor_args.append(l) 
        async with aiohttp.ClientSession() as session:
            session = RateLimiter(session)
            
            tasks = [asyncio.ensure_future(get_stressors(session,*args)) for args in stressor_args]
            await asyncio.gather(*tasks)



if __name__ == "__main__":    
    df = read_process_dataset()
    #asyncio.run(main(df))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(df))
    aggregate_json_convert_tocsv() # aggregate all the CSVs at the end
