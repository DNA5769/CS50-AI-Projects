import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP NP | NP VP PP | S Conj S | S P S

NP -> N | Det N | Det AP | NP Conj NP
VP -> V | Adv VP | VP Adv | V NP | V PP | V NP PP | VP Conj VP
PP -> P NP
AP -> Adj N | Adj AP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    s = nltk.word_tokenize(sentence)
    s_copy = s.copy()

    #Removing any word that does not contain at least one alphabet
    for word in s_copy:
        alphabet_check = False

        for ch in word:
            if ch.isalpha():
                alphabet_check = True
                break

        if not alphabet_check:
            s.remove(word)

    #Converting all characters to lowercase
    for i in range(len(s)):
        s[i] = s[i].lower()

    return s


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []

    #Iterating through all the subtrees of the sentence, excluding itself using the filter
    for subtree in tree.subtrees(lambda t: t.height() != tree.height()):
        #Checking if label of the subtree is "NP"
        if subtree.label() == 'NP':
            np_chunk_check = False

            #Checking if the subtree doesn't contain any other noun phrases as subtrees, excluding itself using the filter
            for leaves in subtree.subtrees(lambda t: t.height() != subtree.height()):
                if leaves.label() == 'NP':
                    np_chunk_check = True
                    break

            if not np_chunk_check:
                np_chunks.append(subtree)

    return np_chunks


if __name__ == "__main__":
    main()
