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

# -------------------------------------------------------------------- #
# PROBLEM with SimpleTokenizerV1:                                       #
#                                                                       #
# If we pass a text containing words NOT in the vocabulary, we get a   #
# KeyError — those tokens have no associated ID.                        #
#                                                                       #
# Example: tokenizer.encode("Hello, world!")                            #
#   → KeyError: 'Hello'  (not seen in "The Verdict")                   #
#                                                                       #
# SOLUTION → SimpleTokenizerV2 adds two special tokens:                #
#   <|unk|>        : replaces any out-of-vocabulary token               #
#   <|endoftext>   : marks the end of the text document dataset         #

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|unk|>", "<|endoftext>"])
vocab = {token:interger for interger, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# ----------------------------------------------------------------
# Let write the SimpleTokenizerV2, that handle UNK and ENDOFTEXT        |
# ----------------------------------------------------------------

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        # Step 1: split on special tokens FIRST to preserve them intact
        # (if we apply the regex directly, <|endoftext> gets broken into
        #  '<', '|', 'endoftext', '>' — none of which are in the vocab)
        parts = re.split(r'(<\|endoftext>|<\|unk\|>)', text)

        preprocessed = []
        for part in parts:
            if part in ('<|endoftext>', '<|unk|>'):
                preprocessed.append(part)       # keep special token as-is
            else:
                # Step 2: apply normal regex split only on regular text
                tokens = re.split(pattern, part)
                preprocessed.extend([t.strip() for t in tokens if t.strip()])

        # Step 3: replace any remaining unknown token with <|unk|>
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# let try this new tokenizer
#-------------------------------------------------------------------

text1 = "Hello, do you like tea ?"
text2 = "In the sunlit terraces of the palace"
text = "<|endoftext>".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
