import asyncio
import aiofiles
import aiohttp
from aiohttp.client import ClientTimeout
import pandas as pd
import json
import numpy as np
import time

START = time.monotonic()

base_path = '/Users/thierrylincoln/Box/PopBots/mTurk Data/MTurk - QA data (Stress Detection Survey)/'
base_url = 'http://172.27.76.112:9001'

df = pd.read_csv(base_path+'2020-02-07_mturk900balanced_v1.csv',sep=",")

#df = df.drop(columns=['Answer.Severity'])
df['Answer.Label'] = df['Answer.Label'].astype(str)
df['Answer.Label'] = df['Answer.Label'].apply(lambda x: x.lower().replace("/","_").replace(" ","_"))
df['Answer.Stressor'] = df['Answer.Stressor'].apply(lambda x: x.lower())
#df = df[df['Answer.Label']!= 'work_school_productivity']
#df = df[df['Answer.Label']!= 'financial_problem']
#df = df[df['Answer.Label']!= 'other']


print(f'Len of df is {len(df)}')
datasets = ["livejournal"]#,"reddit"]


START = time.monotonic()


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

    stressor_data = category.replace("_"," ") +" stress "+ stressor_sentence
    output_data = []
    endpoint = '/query'
    params = {'data':stressor_data,'dataset': dataset,'maxWords':30,'minWords':4,'top':10,'percent':'0.01','model':'bert'}
    url = f'{base_url}{endpoint}'
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

"""
for dataset in datasets:

    loop = asyncio.get_event_loop()
    stressor_args = []    
    # Iterate over each row 
    for index, rows in df.iterrows(): 
        # Create list for the current row 
        l =[dataset, index, rows['Answer.Label'],rows['Answer.Stressor']] 
        # append the list to the final list 
        stressor_args.append(l) 
    async with aiohttp.ClientSession() as session:
        session = RateLimiter(session)
        timeout = ClientTimeout(total=0)  # `0` value to disable timeout
    loop.run_until_complete(
        asyncio.gather(
            *(get_stressors(*args) for args in stressor_args)
        )
    )
"""


    
    
        
async def main():
    for dataset in datasets:

        stressor_args = []    
        # Iterate over each row 
        for index, rows in df.iterrows(): 
            # Create list for the current row 
            l =[dataset, index, rows['Answer.Label'],rows['Answer.Stressor']] 
            # append the list to the final list 
            stressor_args.append(l) 
        async with aiohttp.ClientSession() as session:
            session = RateLimiter(session)
            
            tasks = [asyncio.ensure_future(get_stressors(session,*args)) for args in stressor_args]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())