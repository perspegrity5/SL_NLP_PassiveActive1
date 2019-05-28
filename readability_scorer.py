#!/usr/bin/env python
# coding: utf-8

import textacy
import pandas as pd
from textacy.keyterms import sgrank

def score_readability(text, model_type):
    doc = textacy.make_spacy_doc(text, lang=model_type)
    ts = textacy.TextStats(doc)
    return ts.readability_stats

def normalize(row, x_max, x_min, reverse_arr = False):
    if not reverse_arr:
        return [round((x - x_min)/(x_max - x_min), 2) for x in row]
    return [round((-1*x - x_min)/(x_max - x_min), 2) for x in row]

def get_normalized_importance(df):
    clauses = df["clauses_text_final"]
    rank_tuples = dict(df['sgrank'])
    ngram_keys = rank_tuples.keys()
    op = []
    for clause in clauses:
        str_clause = "".join(clause)
        denominator = 0
        numerator = 0
        for x in ngram_keys:
            if x in str_clause:
                numerator += rank_tuples[x]
                denominator += 1
        op.append(round(numerator / denominator, 2) if denominator > 0 else 0.0)
    return op

def main_readability_scorer(df, model_type):
    en = textacy.load_spacy_lang(model_type)
    df['readability_attributes_score'] = df.clauses_text_final.apply(lambda arr: [score_readability(x, model_type=en) for x in arr])
    df['grading_level'] = df.readability_attributes_score.apply(lambda dct_arr: [round(dct['flesch_kincaid_grade_level'], 2) for dct in dct_arr])
    df['reading_ease'] = df.readability_attributes_score.apply(lambda dct_arr: [round(dct['flesch_reading_ease'], 2) for dct in dct_arr])
    _ = """[{'flesch_kincaid_grade_level': 0.6257142857142846, 'flesch_reading_ease': 103.04428571428573, 'smog_index': 3.1291, 'gunning_fog_index': 2.8000000000000003, 'coleman_liau_index': 2.6518669999999993, 'automated_readability_index': 0.23714285714285666, 'lix': 7.0, 'gulpease_index': 93.28571428571428, 'wiener_sachtextformel': -2.5074571428571426}]"""
    del df['readability_attributes_score']
    reading_ease = df['reading_ease'].tolist()
    reading_ease = [j for i in reading_ease for j in i]
    reading_ease = [-1*x for x in reading_ease]
    x_max, x_min = max(reading_ease), min(reading_ease)
    df['reading_ease_normalized'] = df['reading_ease'].apply(lambda arr : normalize(arr, x_max, x_min, reverse_arr = True))
    grading_levels = df['grading_level'].tolist()
    grading_levels = [j for i in grading_levels for j in i]
    x_max, x_min = max(grading_levels), min(grading_levels)
    df['grading_level_normalized'] = df['grading_level'].apply(lambda arr : normalize(arr, x_max, x_min, reverse_arr = False))
    #Cross verify that they are correct
    reading_ease = df['reading_ease_normalized'].tolist()
    reading_ease = [j for i in reading_ease for j in i]
    grade = df['grading_level_normalized'].tolist()
    grade = [j for i in grade for j in i]
    print(max(reading_ease), min(reading_ease), max(grade), min(grade))
    df['textacy_doc'] = df.apply(lambda row : textacy.make_spacy_doc(row['prompt'] + " " + row['response'], lang=en), axis = 1)
    df['sgrank'] = df['textacy_doc'].apply(lambda doc : sgrank(doc, n_keyterms = len(doc)))
    df["sgrank_normalized"] = df.apply(get_normalized_importance, axis = 1)
    return df
