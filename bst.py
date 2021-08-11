from __future__ import annotations
from typing import List
from abc import ABC
from dataclasses import dataclass
import math

MIN_VAL = 0
MAX_VAL = 9


class Tree(ABC):
    def is_bst(self) -> bool:
        ...

    def size(self) -> int:
        ...

    def to_list(self) -> List[int]:
        ...

    def is_complete(self) -> bool:
        ...


@dataclass
class Leaf(Tree):
    def is_bst(self) -> bool:
        return True

    def size(self) -> int:
        return 0

    def to_list(self) -> List[int]:
        return []

    def is_complete(self) -> bool:
        return True

    def __hash__(self):
        return repr(self).__hash__()


@dataclass
class Node(Tree):
    left: Tree
    value: int
    right: Tree

    def is_bst(self) -> bool:
        return (
            self.left.is_bst()
            and self.right.is_bst()
            and all([self.value > x for x in self.left.to_list()])
            and all([self.value < x for x in self.right.to_list()])
        )

    def size(self) -> int:
        return self.left.size() + 1 + self.right.size()

    def to_list(self) -> List[int]:
        return self.left.to_list() + [self.value] + self.right.to_list()

    def is_complete(self) -> bool:
        return self.left.is_complete() and self.right.is_complete()

    def __hash__(self):
        return repr(self).__hash__()


@dataclass
class Hole(Tree):
    def is_bst(self) -> bool:
        return True

    def size(self) -> int:
        return 0

    def to_list(self) -> List[int]:
        return []

    def is_complete(self) -> bool:
        return False

    def __hash__(self):
        return repr(self).__hash__()


class Choice(ABC):
    def print(self) -> str:
        ...


@dataclass
class ChooseLeaf(Choice):
    def __repr__(self) -> str:
        return "L"


@dataclass
class ChooseNode(Choice):
    value: int

    def __repr__(self) -> str:
        return "N" + str(self.value)


def parse_choices(choices: List[Choice]) -> List[Tree]:
    def expand_hole(c: Choice, t: Tree) -> Tree:
        if isinstance(t, Leaf):
            return Leaf()
        elif isinstance(t, Node):
            new_left = expand_hole(c, t.left)
            if new_left != t.left:
                return Node(new_left, t.value, t.right)
            else:
                return Node(t.left, t.value, expand_hole(c, t.right))
        else:  # Hole
            if isinstance(c, ChooseNode):
                return Node(Hole(), c.value, Hole())
            else:
                return Leaf()

    trees: List[Tree] = [Hole()]
    for choice in choices:
        trees.append(expand_hole(choice, trees[-1]))
    return trees


def choose(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def catalan(n):
    return (1 / (n + 1)) * choose(2 * n, n)


def count_bsts(n):
    return sum(catalan(k) * choose(n, k) for k in range(1, n + 1)) + 1


def fitness(tree: Tree, g_minimum=MIN_VAL, g_maximum=MAX_VAL) -> float:
    if not tree.is_bst():
        return 0

    if tree.is_complete():
        return 1

    def count_completions(
        t: Tree, minimum: int = g_minimum, maximum: int = g_maximum
    ) -> float:
        if isinstance(t, Leaf):
            return 1

        elif isinstance(t, Node):
            l = count_completions(t.left, minimum=minimum, maximum=t.value - 1)
            r = count_completions(t.right, minimum=t.value + 1, maximum=maximum)
            return l * r

        else:  # isinstance(t, Hole):
            return count_bsts(maximum - minimum + 1)

    return count_completions(tree) / count_bsts(g_maximum - g_minimum + 1)


if __name__ == "__main__":
    choices = [ChooseNode(5), ChooseNode(1), ChooseLeaf(), ChooseLeaf(), ChooseLeaf()]
    for partial in parse_choices(choices):
        print(f"{fitness(partial):.6f}: {partial}")