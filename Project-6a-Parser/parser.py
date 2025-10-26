import nltk
import sys

# Terminal symbols
TERMINALS = r"""
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# Nonterminal symbols (a compact, expressive grammar for CS50 Parser)
NONTERMINALS = r"""
S -> NP VP
S -> S Conj S
S -> NP VP Conj VP

NP -> N
NP -> Det N
NP -> Det AP N
NP -> NP PP
NP -> AP NP
NP -> N PP

VP -> V
VP -> V NP
VP -> V PP
VP -> V NP PP
VP -> Adv VP
VP -> VP Adv

PP -> P NP
AP -> Adj
AP -> Adj AP
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
    - Lowercase all tokens.
    - Keep only tokens that contain at least one alphabetic character.
    """
    # Tokenize with NLTK
    tokens = nltk.word_tokenize(sentence)

    # Lowercase and filter out tokens without any alphabetic chars
    words = []
    for tok in tokens:
        t = tok.lower()
        # Keep if any a-z present
        if any(ch.isalpha() for ch in t):
            words.append(t)
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase (NP) chunks in the sentence tree.
    A noun phrase chunk is any subtree labeled "NP" that does not
    itself contain other NP subtrees.
    """
    chunks = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        # Check if this NP has any NP descendants
        has_np_descendant = any(child.label() == "NP" for child in subtree.subtrees(lambda t: True)[1:])
        if not has_np_descendant:
            chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
