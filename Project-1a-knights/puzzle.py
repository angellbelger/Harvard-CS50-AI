from logic import *

# Symbols
AKnight = Symbol("A is a Knight")
AKnave  = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave  = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave  = Symbol("C is a Knave")

# Helpers: each person is exactly one of Knight/Knave
A_xor = And(Or(AKnight, AKnave), Not(And(AKnight, AKnave)))
B_xor = And(Or(BKnight, BKnave), Not(And(BKnight, BKnave)))
C_xor = And(Or(CKnight, CKnave), Not(And(CKnight, CKnave)))

# Puzzle 0
# A says "I am both a knight and a knave."
A_claim0 = And(AKnight, AKnave)
knowledge0 = And(
    A_xor,
    Implication(AKnight, A_claim0),
    Implication(AKnave, Not(A_claim0))
)

# Puzzle 1
# A says "We are both knaves."
A_claim1 = And(AKnave, BKnave)
knowledge1 = And(
    A_xor, B_xor,
    Implication(AKnight, A_claim1),
    Implication(AKnave, Not(A_claim1))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
A_stmt2 = Or(And(AKnight, BKnight), And(AKnave, BKnave))
B_stmt2 = Or(And(AKnight, BKnave), And(AKnave, BKnight))
knowledge2 = And(
    A_xor, B_xor,
    Implication(AKnight, A_stmt2),
    Implication(AKnave, Not(A_stmt2)),
    Implication(BKnight, B_stmt2),
    Implication(BKnave, Not(B_stmt2))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'." and "C is a knave."
# C says "A is a knight."
ASaidKnight = Symbol("A said 'I am a Knight'")
ASaidKnave  = Symbol("A said 'I am a Knave'")

knowledge3 = And(
    A_xor, B_xor, C_xor,

    # Exactly one of the two was said by A
    Or(ASaidKnight, ASaidKnave),
    Not(And(ASaidKnight, ASaidKnave)),

    # If A is a knight, what he said is true; if a knave, what he said is false
    Implication(AKnight, Or(And(ASaidKnight, AKnight),
                            And(ASaidKnave,  AKnave))),
    Implication(AKnave,  Or(And(ASaidKnight, Not(AKnight)),
                            And(ASaidKnave,  Not(AKnave)))),

    # B’s claims
    Implication(BKnight, ASaidKnave),
    Implication(BKnave, Not(ASaidKnave)),
    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),

    # C’s claim
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
