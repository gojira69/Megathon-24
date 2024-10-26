import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords as sw
import contractions
import string

nltk.download('stopwords', quiet=True)

def getStopwords():
    stopwords = set(sw.words("english"))
    negators = set({"n't", "not", "never", "no"})
    additionalSymbols = {'--', '...', '–', '—'}
    punctuations = set(string.punctuation).union(additionalSymbols)
    filtered_stopwords = stopwords.union(punctuations) - negators
    return filtered_stopwords

def getExtraction(sentence):
    sentence = contractions.fix(sentence)
    words = word_tokenize(sentence.lower())
    stopwords = getStopwords()
    extraction = [word for word in words if word not in stopwords]
    return extraction

def main():
    sentence = "I am constantly worried these days."
    extraction = getExtraction(sentence)
    for word in extraction:
        print(word, end = " ")
    print()

if __name__ == "__main__":
    main()
