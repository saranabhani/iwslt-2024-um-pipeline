import argparse
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import string

def preprocess_function(examples, tokenizer, max_input_length, max_target_length, source_lang, target_lang, source, target):
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang
    
    inputs = [ex for ex in examples[source]]
    targets = [ex for ex in examples[target]]
    model_inputs = tokenizer(text=inputs, max_length=max_input_length, truncation=True, padding="max_length")

    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_texts(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate an NLLB model.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from Huggingface.co models.")
    parser.add_argument("--train_file", type=str, required=True, help="A csv file containing the training data.")
    parser.add_argument("--validation_file", type=str, required=True, help="A csv file containing the validation data.")
    parser.add_argument("--source_column", type=str, required=True, help="Name of the source text column in the csv.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target text column in the csv.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to store the fine-tuned model.")
    parser.add_argument("--max_input_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_target_length", type=int, default=128, help="The maximum total target sequence length after tokenization.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device during evaluation.")
    parser.add_argument("--src_lang_code", type=str, required=True, help="source language")
    parser.add_argument("--tgt_lang_code", type=str, required=True, help="target language")
    parser.add_argument("--checkpoint", type=str, required=False, help="checkpoint path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    raw_datasets = load_dataset("csv", data_files={"train": args.train_file, "validation": args.validation_file})
    tokenized_datasets = raw_datasets.map(lambda examples: preprocess_function(examples, tokenizer, args.max_input_length, args.max_target_length, args.src_lang_code, args.tgt_lang_code, args.source_column, args.target_column), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = trainer.evaluate()
    print(f"Evaluation Results: {results}")

    bleu_metric = load_metric("bleu")
    predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
    predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in predictions]
    references = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=False) for l in labels]
    
    preprocessed_prediction_texts = [preprocess_texts(pred) for pred in predictions]
    preprocessed_reference_texts = [preprocess_texts(refs) for refs in references]

    tokenized_predictions = [pred.split() for pred in preprocessed_prediction_texts]
    tokenized_references = [[ref.split()] for ref in preprocessed_reference_texts] 
    scores = bleu_metric.compute(predictions=tokenized_predictions, references=tokenized_references)
    print(f"scores: {scores}")

if __name__ == "__main__":
    main()
