#!/usr/bin/env python
# coding: utf-8

import pandas as pd, numpy as np

def get_final_voice(row):
    cdict = {-1 : 'sentinel', 0 : 'voice_abstraction', 1 : 'voice_grading_level', 2 : 'voice_reading_ease', 3 : 'voice_sgrank'}
    arr = [row['final_abstraction'] * row['abstraction_weight'], row['final_grading_level'] * row['grading_level_weight'], row['final_reading_ease'] * row['reading_ease_weight'], row['final_sgrank'] * row['sgrank_weight']]
    idx = arr.index(max(arr))
    return row[cdict[idx]]

def main_final_output_scorer(df, result_file, debug_file, WEIGHT_METRICS = True):
    df['idx_abstraction_score'] = df['abstraction_score_normalized'].apply(lambda arr: arr.index(max(arr)) if len(arr)>0 else -1)
    df['idx_grading_level'] = df['grading_level_normalized'].apply(lambda arr: arr.index(max(arr)) if len(arr)>0 else -1)
    df['idx_reading_ease'] = df['reading_ease_normalized'].apply(lambda arr: arr.index(max(arr)) if len(arr)>0 else -1)
    df['idx_sgrank'] = df['sgrank_normalized'].apply(lambda arr: arr.index(max(arr)) if len(arr)>0 else -1)
    
    df['voice_abstraction'] = df.apply(lambda row : row['voice'][row['idx_abstraction_score']] if row['idx_abstraction_score'] >=0 else "Indeterminate", axis = 1)
    df['voice_grading_level'] = df.apply(lambda row : row['voice'][row['idx_grading_level']] if row['idx_grading_level'] >=0 else "Indeterminate", axis = 1)
    df['voice_reading_ease'] = df.apply(lambda row : row['voice'][row['idx_reading_ease']] if row['idx_reading_ease'] >=0 else "Indeterminate", axis = 1)
    df['voice_sgrank'] = df.apply(lambda row : row['voice'][row['idx_sgrank']] if row['idx_sgrank'] >=0 else "Indeterminate", axis = 1)
    
    df['final_abstraction'] = df.apply(lambda row : row['abstraction_score_normalized'][row['idx_abstraction_score']] if row['idx_abstraction_score'] >=0 else -1, axis = 1)
    df['final_grading_level'] = df.apply(lambda row : row['grading_level_normalized'][row['idx_grading_level']] if row['idx_grading_level'] >=0 else -1, axis = 1)
    df['final_reading_ease'] = df.apply(lambda row : row['reading_ease_normalized'][row['idx_reading_ease']] if row['idx_reading_ease'] >=0 else -1, axis = 1)
    df['final_sgrank'] = df.apply(lambda row : row['sgrank_normalized'][row['idx_sgrank']] if row['idx_sgrank'] >=0 else -1, axis = 1)
        
    df['sentinel'] = "Indeterminate"
    
    df['abstraction_weight'] = 1/df['final_abstraction'].mean() if WEIGHT_METRICS else 1
    df['grading_level_weight'] = 1/df['final_grading_level'].mean() if WEIGHT_METRICS else 1
    df['reading_ease_weight'] = 1/df['final_reading_ease'].mean() if WEIGHT_METRICS else 1
    df['sgrank_weight'] = 1/df['final_sgrank'].mean() if WEIGHT_METRICS else 1
    
    df['final_score'] = df.apply(lambda row: (row['final_abstraction'] + row['final_grading_level'] + row['final_reading_ease'] + row['final_sgrank'])/4 if row['final_abstraction'] >=0 else -1 , axis = 1)
    df['final_voice'] = df.apply(get_final_voice, axis = 1)
    df.final_score = df.final_score.apply(lambda x : round(x, 2))
    
    prefix = "weighted_"  if WEIGHT_METRICS else ""
    df1 = df[['UID', 'survey_id', 'prompt_number', 'prompt_id', 'prompt', 'response', 'clauses_text_final', 'final_score', 'final_voice']]
    df1.to_csv("./{}{}".format(prefix, result_file), index = False)
    df2 = df[['UID', 'survey_id', 'prompt_number', 'prompt_id', 'score', 'PassAct','prompt', 'response', 'clauses_text_final', 'voice', 'abstraction_score_normalized', 'reading_ease_normalized',        'grading_level_normalized', 'sgrank_normalized',       'abstraction_weight', 'grading_level_weight', 'reading_ease_weight', 'sgrank_weight',     'final_abstraction', 'final_grading_level', 'final_reading_ease', 'final_sgrank', 'final_score',        'voice_abstraction', 'voice_grading_level', 'voice_reading_ease', 'voice_sgrank', 'final_voice']]
    df2.to_csv("./{}{}".format(prefix, debug_file), index = False)
    arrcols = ["clauses_text_final", "voice", "abstraction_score_normalized", "reading_ease_normalized", "grading_level_normalized", "sgrank_normalized"]
    dfdel = df.copy(deep = True)
    for x in arrcols:
        dfdel[x + "_length"] = dfdel[x].apply(len)
    dfdel = dfdel[[x for x in dfdel.columns if x.endswith("length")]]
    dfdel["flag"] = dfdel.eq(dfdel.iloc[:, 0], axis=0).all(1)
    print("Is this an empty dataframe?", dfdel[dfdel["flag"] == False]) #Works like a charm if this df is empty
    
    arrcols = ["clauses_text_final", "voice", "abstraction_score_normalized", "reading_ease_normalized", "grading_level_normalized", "sgrank_normalized"]
    non_arrcols = list(set(df2.columns) - set(arrcols))
    dfs = []
    for i, row in df2.iterrows():
        tempdf1 = pd.DataFrame([row])
        tempdf1["num_clauses"] = len(row["clauses_text_final"]) 
        tempdf1["num_words"] = len(row["response"].split())
        tempdf1["clause_num"] = 0
        tempdf2 = pd.DataFrame()
        for x in arrcols:
            tempdf2[x] = row[x]
        for x in non_arrcols:
            tempdf2[x] = ''
        tempdf2["num_clauses"] = ''
        tempdf2["num_words"] = ''
        tempdf2["clause_num"] = tempdf2.index + 1
        tempdf2["UID"] = row['UID']
        tempdf2["score"] = row['score']
        dfs.append(pd.concat([tempdf1, tempdf2], sort = True))
    
    df_dfs = pd.concat(dfs, sort=True)
    df_dfs.to_csv("debug.csv")
    
    df_dfs['UID'] = df_dfs.apply(lambda row: row['UID'] + '{0:0{width}}'.format(row['clause_num'], width=2),  axis=1)
    df_dfs = df_dfs.set_index('UID')
    df_result = df_dfs[['prompt_number', 'prompt_id',  'clause_num','num_clauses', 'num_words','score','PassAct', 'clauses_text_final','prompt', 'response',  'voice']]
    df_result = df_result.rename(columns={'score': 'respScore'}) # inplace=True gives warning
    df_result['ClauseCutOK?']=''
    df_result['AP_OK?']=''
    df_result['AP_OK?']=''
    df_result['AP_rule']=''
    df_result['Comments']=''
    df_result.to_csv("result.csv") 
