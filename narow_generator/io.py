# -*- coding: utf-8 -*-
from __future__ import print_function
from typing import Iterable

class IdConverter(object):
    def __init__(self, limit=1000):
        self.id_dict = {}  # type: dict[str, int]
        self.character_dict = []  # type: list[str]
        self.limit = limit

    def convert(self, character: str) -> int:
        if character in self.id_dict:
            return self.id_dict[character]
        else:
            if len(self.character_dict) >= self.limit:
                return self.limit
            else:
                self.id_dict[character] = len(self.character_dict)
                self.character_dict.append(character)
                return self.id_dict[character]
    def inverse(self, character_id: int) -> str:
        return self.character_dict[character_id]

    def convert_sentense(self, sentense: str) -> Iterable[int]:
        return [self.convert(character) for character in sentense]
