# SL_NLP_PassiveActive1

Test ignore
## Run notebooks
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/perspegrity5/SL_NLP_PassiveActive1/master)


Run the notebooks in the exact order. The output files from each notebook is the input for the next notebook. 

0. Place an 'input.csv' file in the root folder. It should have 2 required columns which are "prompt" and "response".
1. Run `spacy_classifiers.ipynb`. This will split the responses into clauses in the file `voice_classified.csv`
2. Run `abstraction_scores.ipynb`. This will split the responses into clauses in the file `abstraction_scored.csv`
3. Run `readability_scorer.ipynb`. This will split the responses into clauses in the file `readability_scored.csv`
4. Finally run `final_output_nb.ipynb`. This will produce two files `output.csv` and `debug.csv`. Output.csv is the minimal output which contains the split clauses, the final score and final voice based on maximum internal scores. Debug.csv contains a bit more granular details and scores of each internal terms.    

## Run locally

Steps to set up an internal tool are:
1. Clone this project. Move to the project folder. 
2. Run `pip install -r requirements.txt`
3. Run `python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt');"`
4. Run `python -m spacy download <model-type>`. Model type can be `en_core_web_lg`, `en_core_web_md` or `en_core_web_sm`.
5. Run `python voice_identifier.py --help` to get started.
