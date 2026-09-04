import urllib.request  # Standard library module to download files from the internet

# URL of the raw text file used as training data (short story "The Verdict" by Edith Wharton)
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")

file_path = "the-verdict.txt"  # Local filename where the downloaded text will be saved

urllib.request.urlretrieve(url, file_path)  # Download the file from the URL and save it locally

with open(file_path, "r", encoding="utf-8") as f:  # Open the file in read mode with UTF-8 encoding
    raw_text = f.read()  # Read the entire file content into a single string

print("Total number of character:", len(raw_text))  # Print the total number of characters in the text
print(raw_text[:99])  # Print the first 99 characters as a quick preview of the content