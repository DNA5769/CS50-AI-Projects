import os
import random
import re
import sys

#Additional Imports
import numpy as np
from collections import Counter
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distrib = dict()

    #Initialising probability distribution to 0 for each page
    for pg in corpus.keys():
        prob_distrib[pg] = 0

    if len(corpus[page]) > 0:
        #With probability damping_factor, assigning links from the page with equal probability
        for pg in corpus[page]:
            prob_distrib[pg] += damping_factor/len(corpus[page])

        #With probability (1-damping_factor), assigning all pages of the corpus with equal probability
        for pg in corpus.keys():
            prob_distrib[pg] += (1-damping_factor)/len(corpus.keys())
    else:
        #Assigning all pages of the corpus with equal probability
        for pg in corpus.keys():
            prob_distrib[pg] += 1/len(corpus.keys())

    return prob_distrib


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    data = []

    #Choosing a random page from the corpus and adding to data
    sample = random.choice(list(corpus.keys()))
    data.append(sample)

    for _ in range(n-1):
        prob_distrib = transition_model(corpus, sample, damping_factor)

        #Choosing a page from the corpus based on transition model and adding to data
        sample = np.random.choice(list(prob_distrib.keys()), p=list(prob_distrib.values()))
        data.append(sample)

    #Dividing the number of times each page was visited by numebr of samples 
    pagerank = {k : v/n for k, v in Counter(data).items()}

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict()

    #Modifying the corpus, to account the fact that,
    #"A page that has no links at all should be interpreted as having one link for every page in the corpus"
    modif_corpus = copy.deepcopy(corpus)
    for pg in modif_corpus.keys():
        if len(modif_corpus[pg]) == 0:
            modif_corpus[pg] = list(corpus.keys())

    #Assigning each page a rank of 1 / N, where N is the total number of pages in the corpus
    for pg in modif_corpus.keys():
        pagerank[pg] = 1/len(modif_corpus.keys())

    convergence_check = False
    while not convergence_check:
        old_pagerank = copy.deepcopy(pagerank)

        for page in pagerank.keys():
            sigma = 0
            for pg in pagerank.keys():
                if page in modif_corpus[pg]: #Finding all the pages that link to 'page'
                    sigma += pagerank[pg]/len(modif_corpus[pg])
                    
            pagerank[page] = (1-damping_factor)/len(modif_corpus.keys()) + damping_factor*sigma

        #Making sure the new values differ more than 0.001
        convergence_check = True
        for pg in modif_corpus.keys():
            if abs(pagerank[pg] - old_pagerank[pg]) > 0.001:
                convergence_check = False
                break

    return pagerank
    

if __name__ == "__main__":
    main()
