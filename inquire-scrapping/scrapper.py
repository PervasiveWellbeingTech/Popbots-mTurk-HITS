import asyncio
import aiofiles
import aiohttp
from aiohttp.client import ClientTimeout
import pandas as pd
import json
import numpy as np
import time
from os import listdir

BASE_URL = 'http://commuter.stanford.edu:9001'
BASE_PATH = '/home/thierry/Desktop/stanford_pwbt/datasets/'
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
    return df

def aggregate_json_convert_tocsv():
    final_df = pd.DataFrame()

    for category in SCRAPED_CATEGORIES:
        filenames_cat_list = []
        for path in listdir(OUTPUT_PATH):
            if category.replace(" ","_") in path:
                filenames_cat_list.append(path)
        with open(f'./final_data/json/{category}.json', 'w+') as outfile:
            result = []
            for fname in filenames_cat_list:
                print(fname)
                try:
                    with open(OUTPUT_PATH+fname,"rb") as infile:
                        result += json.load(infile)
                except Exception as error:
                    print(f"Error while trying to load {fname}") 
            json.dump(result,outfile)
        df = pd.read_json(f'./final_data/json/{category}.json',orient='records')
        df.drop_duplicates(subset='text', keep='first', inplace=True)

        final_df = pd.concat([df,final_df])
        df.to_csv(f'./final_data/csv/{category}.csv')




    final_df.to_csv(f'./final_data/csv/final_df.csv')





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


async def get_stressors(session,dataset,stressor_index,category, stressor_sentence):

    stressor_data = category.lower().replace("_"," ") +" stress "+ stressor_sentence
    output_data = []
    endpoint = '/query'
    params = {'data':stressor_data,'dataset': dataset,'maxWords':30,'minWords':4,'top':10,'percent':'0.01','model':'bert'}
    url = f'{BASE_URL}{endpoint}'
    #print(f'Getting {category} stressor for sentence {stressor_sentence}')
    
    timeout = ClientTimeout(total=0)  # `0` value to disable timeout
    async with await session.get(url, headers={}, params=params) as resp:
        #print(resp)
        #await asyncio.sleep(2)
        data = await resp.json()
        for elem in data['query_results']:
            json_line  = {"dataset":str(dataset),"stressor":str(stressor_data),"category":str(category),"text":str(elem['sent_text']),"similarity":str(elem['similarity'])}
            output_data.append(json_line)

    async with aiofiles.open(f'./inquire_scraped_data/{category.replace(" ", "_")}_{stressor_index}_{dataset}.json', 'a+') as file:
        await file.write(json.dumps(output_data))


    
        
async def main(df):
    for dataset in scraped_datasets:

        stressor_args = []    
        # Iterate over each row 
        for index, rows in df.iterrows(): 
            # Create list for the current row 
            l =[dataset, index, rows[LABEL_COLUMN_RAW],rows[DATA_COLUMN]] 
            # append the list to the final list 
            stressor_args.append(l) 
        async with aiohttp.ClientSession() as session:
            session = RateLimiter(session)
            
            tasks = [asyncio.ensure_future(get_stressors(session,*args)) for args in stressor_args]
            await asyncio.gather(*tasks)



if __name__ == "__main__":    
    #df = read_process_dataset()
    #asyncio.run(main(df))
    aggregate_json_convert_tocsv() # aggregate all the CSVs at the end