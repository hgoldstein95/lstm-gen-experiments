from __future__ import annotations
from bigen import Gen
from typing import Any, List, cast
from abc import ABC
from dataclasses import dataclass


class BST(ABC):
    def is_bst(self) -> bool:
        ...

    def size(self) -> int:
        ...

    def to_list(self) -> List[int]:
        ...

    def is_complete(self) -> bool:
        ...


@dataclass
class Leaf(BST):
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
class Node(BST):
    left: BST
    value: int
    right: BST

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
class Hole(BST):
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
        return "L_"


@dataclass
class ChooseNode(Choice):
    value: int

    def __repr__(self) -> str:
        return "N" + str(self.value)


MAX_DEPTH = 4


def parse_choices(choices: List[Choice]) -> List[BST]:
    def expand_hole(c: Choice, t: BST) -> BST:
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

    trees: List[BST] = [Hole()]
    for choice in choices:
        trees.append(expand_hole(choice, trees[-1]))
    return trees


if __name__ == "__main__":
    print(
        parse_choices(
            [ChooseNode(3), ChooseLeaf(), ChooseNode(5), ChooseLeaf(), ChooseLeaf()]
        )
    )
