#!/usr/bin/env python
# coding: utf-8

import spacy
import json
import html
import re
import os
from io import StringIO
import pandas as pd, numpy as np

with open('config.json', 'r') as f:
    config_json = json.load(f)
    contraction_map = config_json['contractions']
    verb_ing_tokens = config_json['verb_ing_tokens']

def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

def get_children(doc):
    if len([x for x in doc.children]) == 0:
        return [doc]
    if doc.pos_ == "VERB" and doc.dep_ not in ["xcomp", "aux"]:
        return []

    op = flatten_list([get_children(l) for l in doc.lefts]) + [doc] + flatten_list([get_children(r) for r in doc.rights])
    return op

def postprocess(tokens_arr):
    if len(tokens_arr) == 1 and ( tokens_arr[0].dep_ in ["aux", "auxpass"] or tokens_arr[0].tag_ in ["VBG"]): 
        return []
    return tokens_arr

def get_text_from_tokens(tokens_arr):
    op = ' '.join([x.text for x in tokens_arr])
    op = op.replace(" nt", "nt").replace(" '", "'")
    return op

def clause_split_by_verbs(doc):
    op = []
    for token in doc:
        if token.pos_ == "VERB":
            arr = flatten_list([get_children(l) for l in token.lefts]) + [token] + flatten_list([get_children(r) for r in token.rights])
            arr = postprocess(arr)
            op.append(arr)
    if len(op)==0:
        op.append(doc)
    return op

def remove_prompts(df, nlp):
    prompt, tokens_arr = df.prompt, df.split_by_verbs_arr
    pdoc = nlp(prompt)
    ignore_indices = [x.i for x in pdoc]
    new_arr = []
    for clause in tokens_arr:
        new_clause = [t for t in clause if t.i not in ignore_indices]
        if len(new_clause) >= 0:
            new_arr.append(new_clause)
    return [x for x in new_arr if len(x) != 0]

def filter_valid_text_df(clauses_arr):
    new_arr = []
    # first pass
    first_pass = []
    tok_arr = [[ tok.i for tok in clause] for clause in clauses_arr]

    for i in range(len(tok_arr)):
        x = tok_arr[i]
        if len(x) ==  0:
            continue
        is_subset = False
        for y in tok_arr:
            if set(x).issubset(y) and not set(x) == set(y):
                is_subset = True
        if not is_subset:
            first_pass.append(i)
    new_arr = [idx for idx in first_pass if len(clauses_arr[idx]) > 0]
    return new_arr

def get_valid_text_df(row):
    clauses_arr = row["clauses_doc_final"]
    valid_indices = row["valid_indices_per_doc"]
    filtered_clauses = [get_text_from_tokens(clauses_arr[x]) for x in valid_indices]
    return filtered_clauses

def process_verbs_df(clauses_arr):
    new_arr = []
    # first pass
    first_pass = []
    tok_arr = [[ tok.i for tok in clause] for clause in clauses_arr]

    for i in range(len(tok_arr)):
        x = tok_arr[i]
        if len(x) ==  0:
            continue
        is_subset = False
        for y in tok_arr:
            if set(x).issubset(y) and not set(x) == set(y):
                is_subset = True
        if not is_subset:
            first_pass.append(clauses_arr[i])
    
    for clauses in first_pass:
        if len(clauses) == 0:
            continue
        txt = get_text_from_tokens(clauses)
        new_arr.append(txt)
    
    return new_arr
        
p_stem, a_poss, p_yn, p_verb_ing, a_verb_ing, p_beverb, p_get, a_def, undef = "P_stem", "A_pron_x", "P_yn", "P_very_ing", "A_ing", "P_bevb_x", "P_get_x", "A_def", "U_undefined"

def voice_rule_engine(clause, prompt):
    PROMPT_VERBS = ['is', 'am', 'are', 'get', 'feel']
    BEING_VERBS = ['be', 'am', 'is', 'isn', 'are', 'aren', 'was', 'were', 'wasn', 'weren', 'been', 'being', 'have', 'haven', 'has', 'hasn', 'could', 'couldn', 'should', 'shouldn', 'would', 'wouldn', 'may', 'might', 'mightn', 'must','mustn', 'shall', 'can', 'will',  'do', 'don', 'did', 'didn', 'does', 'doesn', 'having']
    
    if len(list(set(PROMPT_VERBS).intersection(set([x.text.lower() for x in clause])))) > 0:
        return p_stem

    if True not in [x.pos_ == "VERB" for x in clause]:
        return undef
    
    for i in range(1, len(clause)):
        x, xprev = clause[i], clause[i-1]
        if x.dep_ == "poss" and xprev.text.lower() in BEING_VERBS:
            return a_poss
        
    ct = 0
    for x in clause:
        if x.text.lower().strip() in ['yes', 'no']:
            ct += 1
    if ct >= len(clause)/2:
        return p_yn

    for i in range(len(clause)-1):
        x, xnext = clause[i], clause[i+1]
        if x.text.lower().strip() in BEING_VERBS and x.pos_ == "VERB":
            if xnext.text.lower() in verb_ing_tokens:
                return p_verb_ing
            return a_verb_ing

    for x in clause:
        if x.text.lower().strip() in BEING_VERBS and x.pos_ == "VERB":
            return p_beverb

    for x in clause:
        if x.dep_ in ["advcl", "ROOT"] and x.text in ["get", "seem", "feel", "gets", "seems", "feels", "got", "seemed", "felt"]:
            return p_get
    
    return a_def
    
def clauses_voice(row):
    arr_of_clauses, prompt = row['clauses_doc_final'], row['prompt']
    op = []
    for clause in arr_of_clauses:
        voice = voice_rule_engine(clause, prompt)
        op.append(voice)         
    return op

def replace_contractions(sent):
    tokens = sent.split()
    contractions = contraction_map.keys()
    op_tokens = []
    for token in tokens:
        op_token = token
        if token.lower() in contractions:
            op_token = contraction_map[token.lower()]
            beg, end = op_token[0], op_token[1:]
            beg = beg.upper() if token[0].isupper() else beg.lower()
            op_token = beg + end
        op_tokens.append(op_token)
    return ' '.join(op_tokens)

def htmlise(row, file_dir):
    html_fs = """
    <html>
        <head>
            <title>{}</title>
        </head>
        <body>
            <div>{}</div>
            <div>{}</div>
            <div>{}</div>
        </body>
    </html>"""
    op = spacy.displacy.render(row.nlp_doc, style='dep')
    file_loc = os.path.join(file_dir, "file_{}.html".format(row.UID))
    with open(file_loc, "w") as f:
        f.write(html_fs.format(row.prompt, row.response, row.clauses_text_final, op))
    return file_loc

def main_voice_classifier(model_type, ip_file, ignore_prompt=True, save_html=True, file_dir=None):
    nlp = spacy.load(model_type)
    print("Loaded models")
    df = pd.read_csv(ip_file)
    print(df.columns)
    if "prompt" in df.columns: #Original dataset
        df['sentence'] = df.apply(lambda row : "{} {}".format(row['prompt'], row['response']), axis = 1)
        
    df['preprocessed_sentence'] = df['sentence'].apply(replace_contractions)
    PATTERN = "[^a-zA-Z0-9\s]+"
    rgx = re.compile(PATTERN, re.IGNORECASE)
    df['preprocessed_sentence'] = df['preprocessed_sentence'].apply(lambda ip : re.sub('\s+', ' ', rgx.sub(' ', html.unescape(ip))))
    df['nlp_doc'] = df['preprocessed_sentence'].apply(lambda ip : nlp(ip))
    df['split_by_verbs_arr'] = df['nlp_doc'].apply(clause_split_by_verbs)
    if ignore_prompt:
        df['clauses_doc_final'] = df[['prompt', 'split_by_verbs_arr']].apply(lambda x : remove_prompts(x, nlp), axis = 1)
    else:
        df['clauses_doc_final'] = df['split_by_verbs_arr'].copy()
        
    df["valid_indices_per_doc"] = df['clauses_doc_final'].apply(filter_valid_text_df)
    df['clauses_text_final'] = df.apply(lambda row: get_valid_text_df(row), axis = 1)
    if save_html:
        df['file_loc'] = df.apply(lambda row: htmlise(row, file_dir), axis=1)
    
    df['split_by_verbs_arr_cleaned'] = df['split_by_verbs_arr'].apply(process_verbs_df)
    #We will solve the inconsistency in voice length  using valid_indices
    df[df["clauses_text_final"].apply(len) != df["clauses_doc_final"].apply(len)]
    df['voice'] = df[['clauses_doc_final', 'prompt']].apply(clauses_voice, axis=1)
    df["voice_filtered"] = df.apply(lambda row: [row["voice"][i] for i in range(len(row["voice"])) if i in row["valid_indices_per_doc"]], axis = 1)
    df["voice"] = df["voice_filtered"]
    print("Is this 0?", df[df["clauses_text_final"].apply(len) != df["voice"].apply(len)].shape[0])
    op_cols = ['UID', 'survey_id', 'prompt_number', 'prompt_id', 'prompt', 'response', 'clauses_text_final', 'clauses_doc_final', 'voice', 'score','PassAct']
    if save_html:
        op_cols = op_cols + ['file_loc']
    df_out = df[op_cols]
    return df_out
