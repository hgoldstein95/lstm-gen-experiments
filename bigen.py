from __future__ import annotations
from typing import Callable, List, Optional, Tuple, TypeVar, Generic
import random

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


class Unit:
    pass


class Iso(Generic[A, B]):
    apply: Callable[[A], B]
    unapply: Callable[[B], Optional[A]]

    def __init__(self, apply: Callable[[A], B], unapply: Callable[[B], Optional[A]]):
        self.apply = apply
        self.unapply = unapply


class Gen(Generic[B, A]):
    sample: Callable[[], A]

    def __init__(self, f: Callable[[], A]):
        self.sample = f

    @staticmethod
    def pure(x: A) -> Gen[B, A]:
        return Gen(lambda: x)

    @staticmethod
    def select(name: str, *gs: Gen[A, A]) -> Gen[A, A]:
        return random.choice(gs)

    def bind(self, f: Callable[[A], Gen[B, C]]) -> Gen[B, C]:
        a = self.sample()
        return f(a)

    def comap(self, g: Callable[[C], Optional[B]]) -> Gen[C, A]:
        return Gen(self.sample)


Choice = Tuple[str, int]


class UnGen(Generic[B, A]):
    ungen: Callable[[B], Optional[Tuple[A, List[Choice]]]]

    def __init__(self, f: Callable[[B], Optional[Tuple[A, List[Choice]]]]):
        self.ungen = f

    @staticmethod
    def pure(x: A) -> UnGen[B, A]:
        return UnGen(lambda _: (x, []))

    @staticmethod
    def select(name: str, *gs: UnGen[A, A]) -> UnGen[A, A]:
        def ungen(a1: A) -> Optional[Tuple[A, List[Choice]]]:
            for i, g in enumerate(gs):
                res = g.ungen(a1)
                if res is not None:
                    a2, choices = res
                    if a1 != a2:
                        continue
                    return (a2, [(name, i)] + choices)
            return None

        return UnGen(ungen)

    def bind(self, f: Callable[[A], UnGen[B, C]]) -> UnGen[B, C]:
        def ungen(b: B) -> Optional[Tuple[C, List[Choice]]]:
            res1 = self.ungen(b)
            if res1 is None:
                return None
            a, choices1 = res1
            res2 = f(a).ungen(b)
            if res2 is None:
                return None
            c, choices2 = res2
            return (c, choices1 + choices2)

        return UnGen(ungen)

    def comap(self, g: Callable[[C], Optional[B]]) -> UnGen[C, A]:
        return UnGen(lambda c: self.ungen(g(c)))
