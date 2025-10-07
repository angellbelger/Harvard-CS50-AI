import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) > 0 and self.count == len(self.cells):
            return set(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        if cell in self.mines:
            return
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        if cell in self.safes:
            return
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def _neighbors(self, cell):
        """Return set of in-bounds neighbors around cell (8-connected)."""
        (r, c) = cell
        nbrs = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    nbrs.add((nr, nc))
        return nbrs

    def add_knowledge(self, cell, count):
        """
        Called when the board tells us, for a given safe cell,
        how many neighboring cells have mines.

        Steps:
          1) mark the cell as a move that has been made
          2) mark the cell as safe
          3) add a new sentence based on the neighbors and the count
          4) deduce new safes/mines from knowledge
          5) infer new sentences via subset rule
        """
        # 1) record move
        self.moves_made.add(cell)

        # 2) we now know it's safe
        self.mark_safe(cell)

        # 3) build sentence from unknown neighbors
        neighbors = self._neighbors(cell)

        # remove any neighbors already known safe/mine to normalize sentence
        unknown_neighbors = set()
        adjusted_count = count

        for n in neighbors:
            if n in self.mines:
                adjusted_count -= 1
            elif n in self.safes:
                continue
            elif n == cell:
                continue
            else:
                unknown_neighbors.add(n)

        if unknown_neighbors:
            new_sentence = Sentence(unknown_neighbors, adjusted_count)
            if new_sentence not in self.knowledge:
                self.knowledge.append(new_sentence)

        # 4 & 5) Iterate until no more changes
        changed = True
        while changed:
            changed = False

            # 4) Check each sentence for conclusive info
            safes_to_mark = set()
            mines_to_mark = set()

            for s in self.knowledge:
                safes_to_mark |= s.known_safes()
                mines_to_mark |= s.known_mines()

            for s in safes_to_mark:
                if s not in self.safes:
                    self.mark_safe(s)
                    changed = True

            for m in mines_to_mark:
                if m not in self.mines:
                    self.mark_mine(m)
                    changed = True

            # Clean out empty sentences
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

            # 5) Subset inference: if S1 ⊂ S2, then S2−S1 = count2−count1
            new_inferred = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 is s2:
                        continue
                    if s1.cells and s1.cells.issubset(s2.cells):
                        diff_cells = s2.cells - s1.cells
                        diff_count = s2.count - s1.count
                        if diff_cells and diff_count >= 0:
                            candidate = Sentence(diff_cells, diff_count)
                            # Avoid duplicates
                            if candidate not in self.knowledge and candidate not in new_inferred:
                                new_inferred.append(candidate)

            if new_inferred:
                self.knowledge.extend(new_inferred)
                changed = True

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the board that hasn't been chosen yet.
        """
        candidates = self.safes - self.moves_made
        return next(iter(candidates), None)

    def make_random_move(self):
        """
        Returns a random valid move:
          - not already chosen
          - not known to be a mine
        """
        all_cells = {(r, c) for r in range(self.height) for c in range(self.width)}
        choices = list(all_cells - self.moves_made - self.mines)
        if not choices:
            return None
        return random.choice(choices)
