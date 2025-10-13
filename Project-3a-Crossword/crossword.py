import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generator.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    # -----------------------------
    # CSP methods (implemented)
    # -----------------------------

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent:
        keep only words of the correct length.
        """
        for var in self.domains:
            remove = {w for w in self.domains[var] if len(w) != var.length}
            if remove:
                self.domains[var] -= remove

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        Remove values from `self.domains[x]` for which there is no
        corresponding value in `self.domains[y]` that satisfies the overlap.
        Return True if a revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps.get((x, y))
        if overlap is None:
            return revised

        i, j = overlap
        to_remove = set()
        for xw in self.domains[x]:
            # keep xw only if ∃ yw in Dom(y) such that letters match at overlap
            if not any(xw[i] == yw[j] for yw in self.domains[y]):
                to_remove.add(xw)

        if to_remove:
            self.domains[x] -= to_remove
            revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Enforce arc consistency. If `arcs` is None, initialize with all arcs.
        Return False if any domain becomes empty; True otherwise.
        """
        if arcs is None:
            queue = [(x, y)
                     for x in self.crossword.variables
                     for y in self.crossword.neighbors(x)]
        else:
            queue = list(arcs)

        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` assigns a word to each variable.
        """
        return (set(assignment.keys()) == self.crossword.variables
                and all(assignment[v] is not None for v in assignment))

    def consistent(self, assignment):
        """
        Return True if assignment is consistent:
          - all words distinct
          - all words correct length
          - all overlaps agree on letters
        """
        # All words distinct
        vals = list(assignment.values())
        if len(vals) != len(set(vals)):
            return False

        # Correct length and overlap agreement
        for var, word in assignment.items():
            if len(word) != var.length:
                return False
            for n in self.crossword.neighbors(var):
                if n in assignment:
                    overlap = self.crossword.overlaps[(var, n)]
                    if overlap is None:
                        continue
                    i, j = overlap
                    if word[i] != assignment[n][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Least-Constraining-Value ordering:
        sort words by how few neighbor values they eliminate.
        """
        def eliminated_count(word):
            total = 0
            for n in self.crossword.neighbors(var):
                if n in assignment:
                    continue
                overlap = self.crossword.overlaps[(var, n)]
                if overlap is None:
                    continue
                i, j = overlap
                total += sum(1 for nw in self.domains[n] if word[i] != nw[j])
            return total

        return sorted(self.domains[var], key=eliminated_count)

    def select_unassigned_variable(self, assignment):
        """
        MRV (fewest remaining values), tie-break by highest degree.
        """
        unassigned = [v for v in self.crossword.variables if v not in assignment]
        # MRV
        m = min(len(self.domains[v]) for v in unassigned)
        candidates = [v for v in unassigned if len(self.domains[v]) == m]
        if len(candidates) == 1:
            return candidates[0]
        # Degree heuristic
        return max(candidates, key=lambda v: len(self.crossword.neighbors(v)))

    def backtrack(self, assignment):
        """
        Backtracking search with forward checking (AC-3 after each choice).
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            # Try value
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if not self.consistent(new_assignment):
                continue

            # Save domains to restore on failure
            saved_domains = {v: self.domains[v].copy() for v in self.domains}

            # Forward check: restrict var to chosen value
            self.domains[var] = {value}

            # Propagate with AC-3 (neighbors depend on var)
            if self.ac3(arcs=[(n, var) for n in self.crossword.neighbors(var)]):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

            # Restore on failure
            self.domains = saved_domains

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
