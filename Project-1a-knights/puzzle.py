from logic import *

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
S0 = And(AKnight, AKnave)
knowledge0 = And(
    A_xor,
    Implication(AKnight, S0),
    Implication(AKnave, Not(S0))
)

# Puzzle 1
# A says "We are both knaves." (i.e., A and B are knaves)
S1 = And(AKnave, BKnave)
knowledge1 = And(
    A_xor, B_xor,
    Implication(AKnight, S1),
    Implication(AKnave, Not(S1))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
A_stmt = Or(And(AKnight, BKnight), And(AKnave, BKnave))
B_stmt = Or(And(AKnight, BKnave), And(AKnave, BKnight))
knowledge2 = And(
    A_xor, B_xor,
    Implication(AKnight, A_stmt),
    Implication(AKnave, Not(A_stmt)),
    Implication(BKnight, B_stmt),
    Implication(BKnave, Not(B_stmt))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave." (unknown which)
# B says "A said 'I am a knave'." and "C is a knave."
# C says "A is a knight."
ASaidKnight = Symbol("A said 'I am a Knight'")
ASaidKnave  = Symbol("A said 'I am a Knave'")

knowledge3 = And(
    A_xor, B_xor, C_xor,

    # Exactly one of the two things was said by A
    Or(ASaidKnight, ASaidKnave),
    Not(And(ASaidKnight, ASaidKnave)),

    # If A is a knight, the statement A uttered is true; if A is a knave, it's false
    Implication(AKnight, Or(And(ASaidKnight, AKnight),
                            And(ASaidKnave,  AKnave))),
    Implication(AKnave,  Or(And(ASaidKnight, Not(AKnight)),
                            And(ASaidKnave,  Not(AKnave)))),

    # B's two claims:
    Implication(BKnight, ASaidKnave),
    Implication(BKnave, Not(ASaidKnave)),
    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),

    # C's claim:
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight))
)
