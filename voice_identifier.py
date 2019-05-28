from spacy_classifiers import main_voice_classifier
from abstraction_scores import main_abstraction_scorer
from readability_scorer import main_readability_scorer
from final_output_nb import main_final_output_scorer
import os

def passes_basic_checks(ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY = False, WEIGHT_METRICS = True, ignore_prompt = True, save_html = True, file_dir=None):
   if save_html and not os.path.isdir(file_dir):
       return False, file_dir + " is not a valid directory to add parsed HTML files. You can either turn off HTML generation by setting save_html flag or use a valid directory."
   return True, None


def pipeline(ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY = False, WEIGHT_METRICS = True, ignore_prompt = True, save_html = True, file_dir=None):
    is_successful, errors = passes_basic_checks(ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY, WEIGHT_METRICS, ignore_prompt, save_html, file_dir)
    if not is_successful:
        print("ERRORS", errors)
        return
    df = main_voice_classifier(model_type, ip_file, ignore_prompt, save_html, file_dir)
    df = main_abstraction_scorer(df, DEBUG_ABSTRACTION_HIERARCHY = DEBUG_ABSTRACTION_HIERARCHY)
    df = main_readability_scorer(df, model_type)
    _ = main_final_output_scorer(df, result_file, debug_file, WEIGHT_METRICS = WEIGHT_METRICS)
    return

import argparse

def main():
    parser = argparse.ArgumentParser(description='Will classify active-passive voice of input CSV sentences based on custom rules and adds some useful scores.')
    parser.add_argument('ip_file', type=str,help='Input CSV file\'s relative or absolute location')
    parser.add_argument('result_file', type=str,help='Output CSV file\'s relative or absolute location')
    parser.add_argument('debug_file', type=str,help='Debug CSV file\'s relative or absolute location')
    parser.add_argument('model_type', type=str, help='Spacy model type you have downloaded in your machine', choices=['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg'])
    parser.add_argument('--debug_abstraction_hierarchy', dest='debug_abstraction_hierarchy', action='store_true', help="Show Hypernym/Hyponym trees during abstraction processing?")
    parser.add_argument('--weighted_metrics', dest='weighted_metrics', action='store_true', help="If set to true, all scores are scaled with their mean.")
    parser.add_argument('--ignore_prompt', dest='ignore_prompt', action='store_true', help="If set to true, prompt clause is not displayed in the outputs")
    parser.add_argument('--save_html', dest='save_html', action='store_true', help="If set to true, Spacy's parse trees are generated and saved as HTML files in FILE_DIR")
    parser.add_argument('--file_dir', type=str, help="If save_html is true, the output HTML files are stored in this directory")
    args = parser.parse_args()
    print(args)
    ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY, WEIGHT_METRICS, ignore_prompt, save_html, file_dir = args.ip_file, args.result_file, args.debug_file, args.model_type, args.debug_abstraction_hierarchy, args.weighted_metrics, args.ignore_prompt, args.save_html, args.file_dir
    pipeline(ip_file, result_file, debug_file, model_type, DEBUG_ABSTRACTION_HIERARCHY, WEIGHT_METRICS, ignore_prompt, save_html, file_dir)

if __name__ == "__main__":
    main()
