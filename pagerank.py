import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python pagerank.py corpus")
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
    probabilities = {}
    num_pages = len(corpus)
    links = corpus[page]

    if links:
        # If the current page has outgoing links
        for link in corpus:
            probabilities[link] = (1 - damping_factor) / num_pages
            if link in links:
                probabilities[link] += damping_factor / len(links)
    else:
        # If the current page has no outgoing links, treat it as if it links to all pages
        for link in corpus:
            probabilities[link] = 1 / num_pages

    return probabilities

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    sample = random.choice(list(corpus.keys())) # Starting from a random page

    for _ in range(n):
        page_rank[sample] += 1
        probabilities = transition_model(corpus, sample, damping_factor)
        sample = random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]

    # Normalizing the page rank
    page_rank = {page: rank / n for page, rank in page_rank.items()}

    return page_rank

    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_rank = page_rank.copy()

    # Iterative calculation of page rank
    change = True
    while change:
        change = False
        for page in corpus:
            rank_sum = sum(page_rank[p] / len(corpus[p]) for p in corpus if page in corpus[p])
            new_rank[page] = ((1 - damping_factor) / num_pages) + (damping_factor * rank_sum)

            # Check if the change is small enough to stop
            if not change and abs(new_rank[page] - page_rank[page]) > 0.001:
                change = True
        
        page_rank = new_rank.copy()

    return page_rank

    raise NotImplementedError


if __name__ == "__main__":
    main()
