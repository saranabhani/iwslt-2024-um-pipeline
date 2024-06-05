import re
import unicodedata

def remove_punctuation(text):
    # Define a regular expression to match punctuation
    punctuation_pattern = r'[^\w\sċĊġĠħĦżŻ\'-]'
    # Remove punctuation using regex
    text_without_punctuation = re.sub(punctuation_pattern, '', text)
    return text_without_punctuation

def make_lowercase(text):
    # Convert text to lowercase
    text_lowercase = text.lower()
    # Convert capital Maltese characters to lowercase
    text_lowercase = text_lowercase.replace('Ċ', 'ċ').replace('Ġ', 'ġ').replace('Ħ', 'ħ').replace('Ż', 'ż')
    return text_lowercase

def process_text(text):
    # Remove punctuation
    text_without_punctuation = remove_punctuation(text)
    # Convert text to lowercase
    text_lowercase = make_lowercase(text_without_punctuation)
    # Tokenize text using whitespace and hyphens, keeping the hyphen with the right-hand word
    tokens = re.split(r'[\s]+(?=\w)', text_lowercase)
    # tokens = re.split(r'[\s-]+(?=\w)', text)
    # Filter out empty tokens
    tokens = [token for token in tokens if token]
    return tokens

# Read text from file
input_file_path = 'train.txt'
output_file_path = 'train3.txt'

# Read input from a file
with open('train.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()

# Process the text
processed_text = process_text(input_text)

# Convert the list of tokens into a single string
processed_string = ' '.join(processed_text)

# Write the processed text to a new file
with open('train3.txt', 'w', encoding='utf-8') as file:
    file.write(processed_string)
