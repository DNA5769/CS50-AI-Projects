import nltk
import sys

#Additional imports
import os
import string
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()

    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        if file.endswith('.txt'):
            with open(path, 'r', encoding='utf8') as f:
                files[file] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    words_copy = words.copy()

    for word in words_copy:
        if word in nltk.corpus.stopwords.words('english') or word in string.punctuation:
            words.remove(word)

    words = [w.lower() for w in words]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()

    for doc in documents:
        for word in documents[doc]:
            words.add(word)

    idfs = dict()
    tot_num_of_docs = len(documents)

    for word in words:
        if word in nltk.corpus.stopwords.words('english') or word in string.punctuation:
            continue

        num_of_docs_word_appears = 0

        for doc in documents:
            if word in documents[doc]:
                num_of_docs_word_appears += 1

        idfs[word] = math.log(tot_num_of_docs/num_of_docs_word_appears)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = dict()

    for file in files:
        tf_idfs[file] = 0

    for word in query:
        if word in nltk.corpus.stopwords.words('english') or word in string.punctuation:
            continue

        for file in files:
            if word in files[file]:
                idf = idfs[word]
                tf = Counter(files[file])[word]
                tf_idfs[file] += tf*idf

    filenames = list(files.keys())
    filenames.sort(key=lambda f: tf_idfs[f], reverse=True)

    return filenames[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    matching_word_measure = dict()

    for sentence in sentences:
        matching_word_measure[sentence] = 0

    for word in query:
        if word in nltk.corpus.stopwords.words('english') or word in string.punctuation:
            continue

        for sentence in sentences:
            if word in sentences[sentence]:
                matching_word_measure[sentence] += idfs[word]

    topsentences = list(sentences.keys())
    topsentences.sort(key= lambda s: matching_word_measure[s], reverse=True)

    max_matching_word_measure = matching_word_measure[topsentences[0]]
    tie = False
    tied_sentences = []
    tie_temp = topsentences.copy()

    for sentence in tie_temp[1:]:
        if matching_word_measure[sentence] == max_matching_word_measure:
            tie = True

            tied_sentences.append(sentence)
            topsentences.remove(sentence)

    if tie:
        query_term_density = dict()

        tied_sentences.append(topsentences[0])
        topsentences.remove(topsentences[0])

        for sentence in tied_sentences:
            words_in_query = 0

            for word in query:
                if word in nltk.corpus.stopwords.words('english') or word in string.punctuation:
                    continue

                if word in sentences[sentence]:
                    words_in_query += 1

            query_term_density[sentence] = words_in_query/len(sentences[sentence])

        tied_sentences.sort(key= lambda s: query_term_density[s], reverse=True)
        topsentences = tied_sentences + topsentences

    return topsentences[:n]


if __name__ == "__main__":
    main()
