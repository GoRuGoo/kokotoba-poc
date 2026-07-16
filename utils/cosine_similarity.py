from __future__ import annotations

import math


class CosineSimilarity:
    @staticmethod
    def calculate(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        length = min(len(left), len(right))
        dot_product = sum(left[index] * right[index]
                          for index in range(length))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot_product / (left_norm * right_norm)
