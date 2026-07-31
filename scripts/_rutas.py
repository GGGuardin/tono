"""Localiza el repositorio del pipeline (chest-xray-pneumonia) esté donde esté.

Necesario porque estos scripts se ejecutan de dos formas distintas:

- **Importados** desde un notebook que ya ha puesto las rutas en `sys.path`.
- **Como subproceso** (`python scripts/loquesea.py`), donde `sys.path` está vacío
  de todo eso y una ruta relativa cableada falla.

El fallo real que motivó esto: `evaluar_multirecorte.py` asumía que el pipeline
estaba en el directorio hermano, cierto en local (`.../Torax/src`) pero falso en
Kaggle, donde los repos quedan en `/tmp/tono` y `/tmp/cxr`. Se busca por
contenido —un directorio que contenga `src/model.py`— en vez de por posición.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CANDIDATOS = [
    Path(__file__).resolve().parent.parent.parent,   # hermano del repo (local)
    Path("/tmp/cxr"),                                 # Kaggle
    Path("/kaggle/working/cxr"),
]


def ruta_pipeline() -> Path:
    """Devuelve el directorio que contiene `src/model.py`, o falla con un mensaje útil."""
    variable = os.environ.get("RUTA_PIPELINE")
    if variable:
        CANDIDATOS.insert(0, Path(variable))
    for c in CANDIDATOS:
        if (c / "src" / "model.py").exists():
            return c
    raise ImportError(
        "No encuentro el repositorio del pipeline (un directorio con src/model.py). "
        f"Buscado en: {[str(c) for c in CANDIDATOS]}. "
        "Clónalo desde https://github.com/GGGuardin/chest-xray-pneumonia o define "
        "la variable de entorno RUTA_PIPELINE."
    )


def anadir_al_path() -> Path:
    """Pone el pipeline en `sys.path` y devuelve su ruta."""
    p = ruta_pipeline()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
