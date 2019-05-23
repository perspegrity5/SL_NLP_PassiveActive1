from spacy_classifiers import main_voice_classifier
from abstraction_scores import main_abstraction_scorer
from readability_scorer import main_readability_scorer
from final_output_nb import main_final_output_scorer

def pipeline(ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY = False, WEIGHT_METRICS = True, ignore_prompt = True):
    df = main_voice_classifier(model_type, ip_file, ignore_prompt)
    df = main_abstraction_scorer(df, DEBUG_ABSTRACTION_HIERARCHY = DEBUG_ABSTRACTION_HIERARCHY)
    df = main_readability_scorer(df, model_type)
    _ = main_final_output_scorer(df, result_file, debug_file, WEIGHT_METRICS = WEIGHT_METRICS)

import argparse

def main():
    parser = argparse.ArgumentParser(description='Will classify active-passive voice of input CSV sentences based on custom rules and adds some useful scores.')
    parser.add_argument('ip_file', type=str,help='Input CSV file\'s relative or absolute location')
    parser.add_argument('result_file', type=str,help='Output CSV file\'s relative or absolute location')
    parser.add_argument('debug_file', type=str,help='Debug CSV file\'s relative or absolute location')
    parser.add_argument('model_type', type=str, help='Spacy model type you have downloaded in your machine', choices=['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg'])
    parser.add_argument('--debug_abstraction_hierarchy', type=bool, default = False)
    parser.add_argument('--weighted_metrics', type=bool, default = True)
    parser.add_argument('--ignore_prompt', dest='ignore_prompt', action='store_true')
    args = parser.parse_args()
    print(args)
    ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY, WEIGHT_METRICS, ignore_prompt = args.ip_file, args.result_file, args.debug_file, args.model_type, args.debug_abstraction_hierarchy, args.weighted_metrics, args.ignore_prompt
    pipeline(ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY, WEIGHT_METRICS, ignore_prompt)

if __name__ == "__main__":
    main()
