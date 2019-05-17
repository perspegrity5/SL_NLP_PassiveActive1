#!/usr/bin/env python
# coding: utf-8

import nltk
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
import pandas as pd


# This part is vital to determine the actual abstraction score recursively. It consists of 2 classes that represent a tree datastructure with multiple children (https://en.wikipedia.org/wiki/Tree_structure):
# * The Hypernym Tree: Its attributes are:
#     * The input token (text)
#     * The input token's hypernyms as detected by NLTK. Each hypernym token is another object of HypernymTree
# * The Hyponym Tree
#     * The input token (text)
#     * The input token's hyponyms as detected by NLTK. Each hyponym token is another object of HyponymTree
#     
# Both classes also have two methods:
# * Get Max Depth: This gives the total depth of a node in the tree.
# * Print Tree: A debug method to print a node with its children indented appropriately.

from nltk.corpus import wordnet as wn, stopwords

class GenericTree:
    def __init__(self, nltk_word, prev_words, print_pad = ""):
        self.prev_words = prev_words + [nltk_word]
        self.word = nltk_word
        self.print_pad = print_pad

    def get_max_depth(self):
        if len(self.children) == 0:
            return 0
        return max([tree.get_max_depth() for tree in self.children ])+1
    
    def print_tree(self):
        print("{}>{}".format(self.print_pad, self.word.lemma_names()[0]))
        for tree in self.children:
            tree.print_tree()
            
class HypernymTree(GenericTree):
    def __init__(self, nltk_word, pos, prev_words, print_pad = ""):
        super().__init__(nltk_word, prev_words, print_pad)
        if len(nltk_word.hypernyms()) == 0 or self.prev_words.count(nltk_word) > 1:
            self.children = [] 
        else:
            hyper = nltk_word.hypernyms() 
            hyper = [x for x in hyper if x.pos() == pos]
            self.children = [HypernymTree(x, pos, self.prev_words, print_pad = "{}==".format(self.print_pad)) for x in hyper]
        
class HyponymTree(GenericTree):
    def __init__(self, nltk_word, pos, prev_words, print_pad = ""): 
        super().__init__(nltk_word, prev_words, print_pad)
        if len(nltk_word.hyponyms()) == 0 or self.prev_words.count(nltk_word) > 1:
            self.children = [] 
        else:
            hypo = nltk_word.hyponyms()
            hypo = [x for x in hypo if x.pos() == pos]
            self.children = [HyponymTree(x, pos, self.prev_words, print_pad = "{}==".format(self.print_pad)) for x in hypo]


from nltk.tokenize import word_tokenize

def hypernym_hierarchy(toks, pos, DEBUG_ABSTRACTION_HIERARCHY = False):
    hyper_trees = [HypernymTree(tok, pos, []) for tok in toks]
    hypers = [tree.get_max_depth() for tree in hyper_trees]
    hypo_trees = [HyponymTree(tok, pos, []) for tok in toks]
    hypos = [tree.get_max_depth() for tree in hypo_trees]
    if DEBUG_ABSTRACTION_HIERARCHY:
        for tree in hyper_trees:
            print("#####################################################3")
            tree.print_tree()
        for tree in hypo_trees:
            print("#####################################################3")
            tree.print_tree()

    assert len(hypers) == len(hypos)
    op = [hypos[i]*1./(hypers[i] + hypos[i]) if hypers[i]>0 or hypos[i]>0 else 0 for i in range(len(hypers))]
    #print(op)
    return max(op)

def hypernym_score(word, spacy_pos, ignore_words, DEBUG_ABSTRACTION_HIERARCHY = False):
    toks = wn.synsets(word)
    if len(toks) == 0 or word in ignore_words:
        return 0

    nltk_pos = "n"
    if spacy_pos not in ["NOUN", "PRON", "PROPN"]:
        return 0
    op = hypernym_hierarchy(toks, nltk_pos, DEBUG_ABSTRACTION_HIERARCHY = DEBUG_ABSTRACTION_HIERARCHY)
    return round(op, 2)
    
def score_abstraction(clauses, clauses_doc, ignore_words, DEBUG_ABSTRACTION_HIERARCHY = False):
    op = []
    for clause in clauses_doc:
        check_texts = ' '.join([x.text for x in clause])
        if check_texts not in clauses:
            continue
        op.append(max([hypernym_score(x.text, x.pos_, ignore_words, DEBUG_ABSTRACTION_HIERARCHY = DEBUG_ABSTRACTION_HIERARCHY) for x in clause]))
    return op

# The abstraction score is already betwee 0 and 1. But it is normalized to determine the valid metric value in later notebooks

def normalize(row, x_max, x_min, reverse_arr = False):
    if not reverse_arr:
        return [round((x - x_min)/(x_max - x_min), 2) for x in row]
    return [round((-1*x - x_min)/(x_max - x_min), 2) for x in row]


def main_abstraction_scorer(df, DEBUG_ABSTRACTION_HIERARCHY = False):
    ignore_words = list(set(stopwords.words('english')))
    ignore_words = ignore_words + ['keep']
    df['abstraction_score'] = df.apply(lambda row: score_abstraction(row['clauses_text_final'], row['clauses_doc_final'], ignore_words, DEBUG_ABSTRACTION_HIERARCHY = DEBUG_ABSTRACTION_HIERARCHY), axis=1)
    abstraction_score = df['abstraction_score'].tolist()
    abstraction_score = [j for i in abstraction_score for j in i]
    x_max, x_min = max(abstraction_score), min(abstraction_score)
    df['abstraction_score_normalized'] = df['abstraction_score'].apply(lambda arr : normalize(arr, x_max, x_min))
    return df
