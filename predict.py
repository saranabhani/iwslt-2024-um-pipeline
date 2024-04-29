from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_metric
import pandas as pd
import argparse
import string
import torch

def batch_iterable(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def generate_predictions(input_texts, model, tokenizer, batch_size=32):
    predictions = []
    for batch_texts in batch_iterable(input_texts, batch_size):
        tokenized_inputs = tokenizer(batch_texts, return_tensors="pt", padding="max_length", truncation=True)
        tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}

        output_sequences = model.generate(**tokenized_inputs)

        batch_predictions = [tokenizer.decode(generated_sequence, skip_special_tokens=True) for generated_sequence in output_sequences]
        predictions.extend(batch_predictions)

    return predictions

def preprocess_texts(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def main():
    parser = argparse.ArgumentParser('Evaluate a model on a given dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--input_col', type=str, required=True, help='columns containing input text')
    parser.add_argument('--save_path', type=str, required=False, help='Path to save the predictions')
    parser.add_argument('--score', action='store_true', required=False, help='Whether to calculate BLEU score')
    parser.add_argument('--output_col', type=str, required=False, help='columns containing output text')

    args = parser.parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model.device)
    df = pd.read_csv(args.dataset)
    input_texts = df[args.input_col].tolist()
    predictions = generate_predictions(input_texts, model, tokenizer)
    if args.save_path:
        with open(args.save_path, 'wb') as f:
            for prediction in predictions:
                prediction = preprocess_texts(prediction)
                f.write(prediction.encode('utf-8') + b"\n")

    if args.score:
        bleu_metric = load_metric("bleu")
        references = df[args.output_col].tolist() 
        preprocessed_prediction_texts = [preprocess_texts(pred) for pred in predictions]
        preprocessed_reference_texts = [preprocess_texts(refs) for refs in references]

        tokenized_predictions = [pred.split() for pred in preprocessed_prediction_texts]
        tokenized_references = [[ref.split()] for ref in preprocessed_reference_texts]
        scores = bleu_metric.compute(predictions=tokenized_predictions, references=tokenized_references)
        print(f"scores: {scores}")


if __name__ == "__main__":
    main()
