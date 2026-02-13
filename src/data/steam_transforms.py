"""Funciones de transformación y parseo específicas del dataset Steam.

Este módulo centraliza la lógica de parseo de campos anidados y cálculo
de features derivadas del dataset de Steam Games, haciéndola reutilizable
entre notebooks y módulos sin duplicación.

Funciones públicas:
    extract_list_names  — Extrae nombres de lista de strings o dicts
    extract_tags_top    — Extrae top-N tags de un diccionario {tag: votos}
    parse_languages     — Normaliza el campo supported_languages a string
    parse_owners_midpoint — Convierte rango de owners a punto medio numérico
"""

from __future__ import annotations

import numpy as np


def extract_list_names(value) -> str:
    """Extrae nombres de una lista de strings o lista de dicts con clave 'description'.

    Args:
        value: Lista de strings o dicts, string, o valor nulo (None/NaN).

    Returns:
        String con nombres separados por '|', o string vacío si el input
        es nulo, vacío o de tipo no soportado.

    Examples:
        >>> extract_list_names(['Action', 'Indie'])
        'Action|Indie'
        >>> extract_list_names([{'description': 'RPG'}, {'description': 'Strategy'}])
        'RPG|Strategy'
        >>> extract_list_names(None)
        ''
    """
    if value is None:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict):
                names.append(item.get('description', item.get('name', str(item))))
            elif isinstance(item, str):
                names.append(item)
        return '|'.join(names)
    return str(value)


def extract_tags_top(value, n: int = 5) -> str:
    """Extrae los top-N tags de un diccionario {tag: votos}.

    Args:
        value: Diccionario de tags con votos como valores, o cualquier
               valor no válido (None, lista vacía, etc.).
        n: Número de tags a retornar. Por defecto 5.

    Returns:
        String con top tags ordenados por votos descendente, separados
        por '|'. Retorna string vacío si el input no es un dict válido.

    Examples:
        >>> extract_tags_top({'RPG': 500, 'Action': 800, 'Indie': 300}, n=2)
        'Action|RPG'
        >>> extract_tags_top({})
        ''
        >>> extract_tags_top(None)
        ''
    """
    if not isinstance(value, dict) or len(value) == 0:
        return ''
    top_tags = sorted(value.items(), key=lambda x: x[1], reverse=True)[:n]
    return '|'.join([tag for tag, _ in top_tags])


def parse_languages(value) -> str:
    """Normaliza el campo supported_languages a string delimitado por comas.

    El campo puede llegar como lista (versiones antiguas del dataset) o
    como string (versiones nuevas). Maneja valores nulos de forma segura.

    Args:
        value: Lista de strings, string, o valor nulo (None/NaN).

    Returns:
        String con idiomas separados por ', ', o string vacío si el
        input es nulo o no parseable.

    Examples:
        >>> parse_languages(['English', 'Spanish', 'French'])
        'English, Spanish, French'
        >>> parse_languages('English, Spanish')
        'English, Spanish'
        >>> parse_languages(None)
        ''
    """
    if value is None:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return ''
    if isinstance(value, list):
        return ', '.join(str(v) for v in value)
    return str(value)


def parse_owners_midpoint(owners_str: str) -> float:
    """Convierte un rango de propietarios estimados a su punto medio numérico.

    El campo 'estimated_owners' del dataset Steam usa el formato
    'low - high' (ej: '20000 - 50000'). Esta función extrae el punto
    medio como float para uso en análisis cuantitativo.

    Soporta:
    - Formato estándar: '20000 - 50000' → 35000.0
    - Valores con comas: '1,000,000 - 2,000,000' → 1500000.0
    - Rango cero: '0 - 0' → 0.0
    - Inputs inválidos (None, '', valores no numéricos) → 0.0

    Args:
        owners_str: String con rango en formato 'low - high', o cualquier
                    valor inválido.

    Returns:
        Punto medio del rango como float. Retorna 0.0 si el input no
        es parseable, en lugar de lanzar excepción.

    Examples:
        >>> parse_owners_midpoint('20000 - 50000')
        35000.0
        >>> parse_owners_midpoint('1,000,000 - 2,000,000')
        1500000.0
        >>> parse_owners_midpoint('0 - 0')
        0.0
        >>> parse_owners_midpoint('')
        0.0
        >>> parse_owners_midpoint(None)
        0.0
    """
    try:
        if not isinstance(owners_str, str) or owners_str.strip() == '':
            return 0.0
        parts = [p.strip().replace(',', '') for p in owners_str.split('-')]
        if len(parts) == 2:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2.0
        return 0.0
    except (ValueError, AttributeError):
        return 0.0
