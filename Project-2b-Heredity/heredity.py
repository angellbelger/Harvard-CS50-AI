import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return the joint probability that
      - everyone in one_gene has 1 copy of the gene,
      - everyone in two_genes has 2 copies,
      - everyone else has 0 copies,
      - everyone in have_trait exhibits the trait, and the rest do not.
    """
    def gene_count(name):
        if name in two_genes:
            return 2
        if name in one_gene:
            return 1
        return 0

    def pass_prob(parent_name):
        """
        Probability that a parent passes the gene to the child,
        given the parent's gene count and mutation rate.
        """
        g = gene_count(parent_name)
        m = PROBS["mutation"]
        if g == 2:
            return 1 - m
        if g == 1:
            return 0.5
        return m

    p_joint = 1.0
    for person, info in people.items():
        g = gene_count(person)
        has_trait = (person in have_trait)

        # Probability of person's gene count
        if info["mother"] is None and info["father"] is None:
            # No parents known: use unconditional prior
            p_gene = PROBS["gene"][g]
        else:
            mom = info["mother"]
            dad = info["father"]
            pm = pass_prob(mom)
            pd = pass_prob(dad)

            if g == 2:
                p_gene = pm * pd
            elif g == 1:
                p_gene = pm * (1 - pd) + (1 - pm) * pd
            else:  # g == 0
                p_gene = (1 - pm) * (1 - pd)

        # Probability of trait given gene count
        p_trait = PROBS["trait"][g][has_trait]

        # Multiply into joint
        p_joint *= p_gene * p_trait

    return p_joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Update both the "gene" and "trait" distributions for each person.
    """
    for person in probabilities:
        if person in two_genes:
            g = 2
        elif person in one_gene:
            g = 1
        else:
            g = 0

        t = (person in have_trait)

        probabilities[person]["gene"][g] += p
        probabilities[person]["trait"][t] += p


def normalize(probabilities):
    """
    Update `probabilities` so that each distribution (gene, trait)
    for each person sums to 1, preserving relative proportions.
    """
    for person in probabilities:
        # Normalize gene distribution
        total_gene = sum(probabilities[person]["gene"].values())
        if total_gene != 0:
            for g in probabilities[person]["gene"]:
                probabilities[person]["gene"][g] /= total_gene

        # Normalize trait distribution
        total_trait = sum(probabilities[person]["trait"].values())
        if total_trait != 0:
            for t in probabilities[person]["trait"]:
                probabilities[person]["trait"][t] /= total_trait


if __name__ == "__main__":
    main()
