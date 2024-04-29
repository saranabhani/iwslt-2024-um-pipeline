# Machine Translation Training and Evaluation

This repository contains Python scripts designed for training and evaluating machine translation models using the Hugging Face transformers library. These scripts allow users to train models on a dataset of choice and then generate translations to evaluate model performance.

## Files
- `train.py`: Trains a machine translation model.
- `predict.py`: Generates translations using a pre-trained machine translation model.


## Usage
### Training
Run the training script with the necessary parameters to start the training process. Here is a typical command:

`python train.py --model_name_or_path <model_path> --train_file <train_data.csv> --validation_file <validation_data.csv> --source_column <src_column> --target_column <tgt_column> --output_dir <output_directory> --max_input_length 128 --max_target_length 128 --num_train_epochs 5 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --src_lang_code ar --tgt_lang_code en`

Replace the placeholders with the relevant values:
- `<model_path>`: Path to pretrained model or model identifier from Huggingface.co models.
- `<train_data.csv>`: A CSV file containing the training data.
- `<validation_data.csv>`: A CSV file containing the validation data.
- `<src_column>` and `<tgt_column>`: Names of the source and target text columns in the CSV file.
- `<output_directory>`: Directory where the fine-tuned model will be stored.

### Evaluation
To evaluate the model and generate translations, use the prediction script:
`python predict.py --model <model_path> --dataset <dataset_path> --input_col <input_column> --output_col <output_column> --save_path <save_predictions_path> --score`

- `<model_path>`: Path to your trained model.
- `<dataset_path>`: Path to the dataset file.
- `<input_column>`: Name of the column containing input text.
- `<output_column>` (optional): Name of the column containing output text, needed if `--score` is used.
- `<save_predictions_path>` (optional): Path to save the predictions.
- `--score` (optional): Include this flag to calculate the BLEU score for translation evaluation.

