import warnings
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Suppress specific FutureWarnings for clean_up_tokenization_spaces
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(text:str):
    # instantiate the classifier
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    # # set the clean_up_tokenization_spaces in tokenizer
    # tokenizer = classifier.tokenizer
    # tokenizer.clean_up_tokenization_spaces = True

    # list file options 
    file_labels = ['json', 'csv']

    # sequence format
    sequence = f"What is the file format for the given input: {text}"

    # run the classifier
    results = classifier(sequence, file_labels, multi_label=False)
    results_labels, results_score = results['labels'], results['scores']
    for i in range(len(results_labels)):
        print(f"Label : {results_labels[i]} was predicted a Score of {results_score[i]}")

if __name__ == '__main__':
    raw_text = input()
    main(raw_text)