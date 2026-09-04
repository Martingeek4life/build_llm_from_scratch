import urllib.request  # Standard library module to download files from the 
import re              # regular expression library
import string
# URL of the raw text file used as training data (short story "The Verdict" by Edith Wharton)
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")

file_path = "the-verdict.txt"  # Local filename where the downloaded text will be saved

urllib.request.urlretrieve(url, file_path)  # Download the file from the URL and save it locally

with open(file_path, "r", encoding="utf-8") as f:  # Open the file in read mode with UTF-8 encoding
    raw_text = f.read()  # Read the entire file content into a single string

print("Total number of character:", len(raw_text))  # Print the total number of characters in the text
print(raw_text[:99] + "\n")  # Print the first 99 characters as a quick preview of the content

# ----------------------------------------------------------------
# 1- How can we best split text to obtain a list of tokens ?         |
# ----------------------------------------------------------------|


# We will implement white space characters split Approach         
# ----------------------------------------------------------------|

print("text_splitting\n")
text_splitting = re.split(r'(\s)', raw_text)
print(text_splitting[:99])

# -------------------------------------------------------------------- #
# Because splitting only on whitespace keeps punctuation attached to    #
# words, we want to treat each punctuation mark as a separate token     #
# as well.                                                              #
#                                                                       #
# So instead of using a regular expression that splits only on          #
# whitespace, we modify it to split on:                                 #
#                                                                       #
#          Whitespace  OR  any punctuation character.                   #
# -------------------------------------------------------------------- #

print("text_splitting_with_punctuation\n")
pattern = rf'([{re.escape(string.punctuation)}]|\s)'
print("\n" + pattern + "\n")
text_splitting_with_punctuation = re.split(pattern, raw_text)
preprocessed = [item.strip() for item in text_splitting_with_punctuation if item.strip()]
print(preprocessed[:99])

# ----------------------------------------------------------------
# 2- Converting Tokens into tokens IDs          |
# ----------------------------------------------------------------|

# list all of unique tokens and sorted them alphabetically to determine  the vocabulary size
all_tokens = sorted(set(preprocessed))
vocab_size = len(all_tokens)
print("Total number of unique tokens:", vocab_size)

# creating a vocabulary
vocab = {token:interger for interger, token in enumerate(all_tokens)}
for i, item in enumerate(vocab.items()):
    print(f"{i}: {item}")
    if i>=50:
        break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab # store the vocabulary as a class attribute for access in the encode and decode methods
        self.int_to_str = {i:s for s,i in vocab.items()} # create an inverse vocabulary , that map token IDs back to the original text tokens
    
    def encode(self, text):
        preprocessed = re.split(pattern, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()] # preprocesse input text in to token IDs
        IDs = [self.str_to_int[item] for item in preprocessed]
        return IDs
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])   # Convert token IDs back into text
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # remove spaces before the specified punctuation
        return text

# ----------------------------------------------------------------
# Let instantiate the SimpleTokenizerV1          |
# ----------------------------------------------------------------
tokenizer = SimpleTokenizerV1(vocab)
text = " ".join(preprocessed[:99])  # join the list back into a string before encoding
ids = tokenizer.encode(text)
print(ids , "\n")
print(tokenizer.decode(ids))