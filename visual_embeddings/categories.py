"""Catálogo configurable, validado por sesión, sin persistencia de ediciones."""
import json
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class Category:
    label: str
    description: str


def parse_categories(rows: Sequence[Sequence[str]]) -> list[Category]:
    categories: list[Category] = []
    for row in rows:
        if len(row) != 2:
            raise ValueError("Cada categoría requiere nombre y descripción.")
        label, description = (str(value or "").strip() for value in row)
        if not label and not description:
            continue
        if not label or not description:
            raise ValueError("Completa el nombre y la descripción de cada categoría.")
        if len(label) > 100 or len(description) > 1000:
            raise ValueError("Máximo: 100 caracteres por nombre y 1000 por descripción.")
        if any(c.label.casefold() == label.casefold() for c in categories):
            raise ValueError(f"Categoría duplicada: {label}")
        categories.append(Category(label, description))
    if not 1 <= len(categories) <= 100:
        raise ValueError("El catálogo debe contener entre 1 y 100 categorías.")
    return categories


def defaults() -> list[Category]:
    return parse_categories(json.loads(Path(__file__).with_name("categories.json").read_text()))


def rows(categories: list[Category]) -> list[list[str]]:
    return [[c.label, c.description] for c in categories]
