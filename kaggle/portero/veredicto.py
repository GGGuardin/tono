"""Decide si el portero aprendido es desplegable, mirando su sesgo por tono.

Vive en un fichero aparte, y no dentro del notebook, por una razón práctica: las
cadenas largas con continuación de línea dentro de una celda JSON son una fuente
constante de errores de escapado. Un módulo importable se prueba en local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NO_DESPLEGAR = (
    "NO DESPLEGAR: descarta mas fotos buenas cuanto mas oscura es la piel. "
    "El modelo copio el sesgo que ya traia la etiqueta humana."
)
CAUTELA = "CAUTELA: brecha grande entre tonos, aunque sin tendencia clara con la oscuridad."
LIMPIO = "Sin evidencia de que descarte fotos buenas segun el tono de piel."
INDETERMINADO = "Indeterminado: pocos tonos con muestra suficiente."


def evaluar_sesgo(subgrupos: pd.DataFrame, min_tonos: int = 3) -> dict:
    """Veredicto a partir de la tabla de subgrupos de `src.fairness`.

    La métrica que decide **no es el AUROC sino la FPR**: cuántas fotos *buenas*
    descarta en cada tono. Un portero con AUROC alto que descarte sistemáticamente
    fotos válidas de piel oscura excluye a ese grupo del sistema entero, y eso no
    lo compensa ninguna métrica global.
    """
    fitz = subgrupos[subgrupos["atributo"] == "fitzpatrick"].copy()
    fitz["tono"] = pd.to_numeric(fitz["subgrupo"], errors="coerce")
    fitz = fitz.dropna(subset=["tono"]).sort_values("tono")

    if len(fitz) < min_tonos:
        return {"veredicto": INDETERMINADO, "tonos_evaluados": len(fitz), "tabla": fitz}

    # Correlación entre oscuridad de la piel y descarte de fotos buenas
    r = float(np.corrcoef(fitz["tono"].values, fitz["fpr"].values)[0, 1])
    brecha = float(fitz["fpr"].max() - fitz["fpr"].min())

    if r > 0.5 and brecha > 0.10:
        veredicto = NO_DESPLEGAR
    elif brecha > 0.15:
        veredicto = CAUTELA
    else:
        veredicto = LIMPIO

    return {
        "veredicto": veredicto,
        "correlacion_tono_fpr": round(r, 3),
        "brecha_fpr": round(brecha, 4),
        "tonos_evaluados": len(fitz),
        "fpr_por_tono": fitz[["subgrupo", "n", "fpr"]].to_dict("records"),
        "tabla": fitz,
    }
