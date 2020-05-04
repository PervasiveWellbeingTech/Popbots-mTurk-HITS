from os import listdir
import json
import pandas as pd

PATH = './inquire_scraped_data/'
categories = ['work','financial_problem','family_issue','health_or_physical_pain','emotional_turmoil','school']


final_df = pd.DataFrame()

for category in categories:
    filenames_cat_list = []
    for path in listdir(PATH):
        if category in path:
            filenames_cat_list.append(path)
    with open(f'./final_data/json/{category}.json', 'w+') as outfile:
        result = []
        for fname in filenames_cat_list:
            print(fname)
            try:
                with open(PATH+fname,"rb") as infile:
                    result += json.load(infile)
            except Exception as error:
                print(f"Error while trying to load {fname}") 
        json.dump(result,outfile)
    df = pd.read_json(f'./final_data/json/{category}.json',orient='records')
    df.drop_duplicates(subset='text', keep='first', inplace=True)

    final_df = pd.concat([df,final_df])
    df.to_csv(f'./final_data/csv/{category}.csv')




"""

actual_categories = {'financial_problem':'Financial Problem',
        'work_school_productivity':'Work/School Productivity',
        'travel_holiday_stress':'Travel/Holiday Stress',
        'personal_social_issues':'Personal/Social Issues',
        'family_issues':'Family Issues',
        'health_or_physical_pain':'Health or Physical Pain',
        'exhaustion_fatigue':'Exhaustion/Fatigue',
        'other':'Other',
        'everyday_decision_making':'Everyday Decision Making',
        'confidence_issue':'Confidence Issue'}


final_df.category = final_df.category.apply(lambda x:actual_categories[x])
"""
final_df.to_csv(f'./final_data/csv/final_df.csv')
