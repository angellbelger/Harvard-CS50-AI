import os
import random
import re
import sys

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
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename), encoding="utf-8") as f:
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

    With probability `damping_factor`, choose uniformly among links from `page`.
    With probability `1 - damping_factor`, choose uniformly among all pages.

    If `page` has no outgoing links, treat it as linking to every page.
    """
    pages = list(corpus.keys())
    N = len(pages)

    # Base teleport probability
    probs = {p: (1 - damping_factor) / N for p in pages}

    links = corpus.get(page, set())
    if not links:
        # Dangling page: acts like it links to every page (including itself)
        links = set(pages)

    share = damping_factor / len(links)
    for p in links:
        probs[p] += share

    # Tiny normalization to offset float drift
    s = sum(probs.values())
    if s != 0.0:
        for p in probs:
            probs[p] /= s
    return probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a random page.

    Returns dict mapping page -> estimated PageRank; values sum to 1.
    """
    pages = list(corpus.keys())
    counts = {p: 0 for p in pages}

    # First sample: uniformly random page
    current = random.choice(pages)
    counts[current] += 1

    # Subsequent samples follow the transition model
    for _ in range(1, n):
        dist = transition_model(corpus, current, damping_factor)
        weights = [dist[p] for p in pages]
        current = random.choices(pages, weights=weights, k=1)[0]
        counts[current] += 1

    ranks = {p: counts[p] / n for p in pages}

    # Normalize (safety)
    s = sum(ranks.values())
    if s != 0.0:
        for p in ranks:
            ranks[p] /= s
    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    until convergence (no value changes by more than 0.001).

    Handles dangling pages by treating them as linking to every page.
    """
    pages = list(corpus.keys())
    N = len(pages)

    # Initialize equally
    ranks = {p: 1 / N for p in pages}

    # Work with sets for outlinks
    outlinks = {p: set(corpus[p]) for p in pages}

    threshold = 0.001
    while True:
        new_ranks = {}
        for p in pages:
            # Teleport term
            rank = (1 - damping_factor) / N

            # Link-follow term
            link_sum = 0.0
            for q in pages:
                if len(outlinks[q]) == 0:
                    # Dangling: distributes to all pages equally
                    link_sum += ranks[q] / N
                elif p in outlinks[q]:
                    link_sum += ranks[q] / len(outlinks[q])

            new_ranks[p] = rank + damping_factor * link_sum

        # Optional normalize to keep tidy
        total = sum(new_ranks.values())
        if total != 0.0:
            for p in new_ranks:
                new_ranks[p] /= total

        # Convergence check
        if max(abs(new_ranks[p] - ranks[p]) for p in pages) <= threshold:
            return new_ranks

        ranks = new_ranks


if __name__ == "__main__":
    main()
