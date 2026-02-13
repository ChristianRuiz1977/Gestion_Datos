"""Detección centralizada de la raíz del proyecto.

Este módulo unifica la lógica de resolución de PROJECT_ROOT que antes
se duplicaba en cada notebook con variantes ligeramente distintas.

Estrategias de detección (en orden de prioridad):
    1. Subir desde ``cwd`` buscando marcadores de estructura (``src/`` + ``data/``).
    2. Variable de entorno ``PROJECT_ROOT`` (definida en ``.env`` o shell).

Uso en notebooks::

    from src.utils.paths import get_project_root
    PROJECT_ROOT = get_project_root()

Uso con dotenv (opcional)::

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    PROJECT_ROOT = get_project_root()
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Resuelve la raíz del proyecto de forma robusta.

    Busca hacia arriba desde el directorio de trabajo actual un directorio
    que contenga tanto ``src/`` como ``data/``.  Si no lo encuentra, intenta
    con la variable de entorno ``PROJECT_ROOT``.

    Returns:
        Path absoluto a la raíz del proyecto.

    Raises:
        EnvironmentError: Si ninguna estrategia logra resolver la raíz.
    """
    # Estrategia 1: subir desde cwd buscando marcadores de estructura
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent

    # Estrategia 2: variable de entorno
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        root = Path(env_root)
        if root.is_dir():
            return root

    raise EnvironmentError(
        "No se pudo detectar PROJECT_ROOT automáticamente.\n"
        "Opciones:\n"
        "  1. Ejecuta el notebook desde el directorio del proyecto.\n"
        "  2. Define PROJECT_ROOT en un archivo .env.\n"
        "  3. Exporta la variable en tu shell: set PROJECT_ROOT=<ruta>"
    )


def get_data_dirs(root: Path | None = None) -> dict[str, Path]:
    """Retorna las rutas estándar de datos del proyecto.

    Args:
        root: Raíz del proyecto. Si es ``None``, se detecta automáticamente.

    Returns:
        Diccionario con claves ``raw``, ``processed``, ``external``,
        ``figures`` y ``tables``.
    """
    if root is None:
        root = get_project_root()

    return {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "external": root / "data" / "external",
        "figures": root / "reports" / "figures",
        "tables": root / "reports" / "tables",
    }
