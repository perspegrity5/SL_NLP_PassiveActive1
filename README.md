# SL_NLP_PassiveActive1
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/perspegrity5/SL_NLP_PassiveActive1/master)


Run the notebooks in the exact order. The output files from each notebook is the input for the next notebook. 

0. Place an 'input.csv' file in the root folder. It should have 2 required columns which are "prompt" and "response".
1. Run `spacy_classifiers.ipynb`. This will split the responses into clauses in the file `voice_classified.csv`
2. Run `abstraction_scores.ipynb`. This will split the responses into clauses in the file `abstraction_scored.csv`
3. Run `readability_scorer.ipynb`. This will split the responses into clauses in the file `readability_scored.csv`
4. Finally run `final_output_nb.ipynb`. This will produce two files `output.csv` and `debug.csv`. Output.csv is the minimal output which contains the split clauses, the final score and final voice based on maximum internal scores. Debug.csv contains a bit more granular details and scores of each internal terms.    
