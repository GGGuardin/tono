"""Varias fotos de la misma zona: coherencia entre ellas como medida de confianza.

Pedir tres fotos no sirve para promediar y quedarse tranquilo. Sirve para lo
contrario: **si las tres predicciones no se parecen, el modelo no sabe lo que
está mirando y no se debe dar un resultado.** El desacuerdo es información, y es
la única estimación de incertidumbre que se obtiene gratis, sin ensamblados ni
modelos bayesianos.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .checks import normalizar_escala


@dataclass
class Concordancia:
    n: int
    media: float
    desviacion: float
    rango: float
    coherente: bool
    mensaje: str


def concordancia_predicciones(
    probabilidades: list[float],
    desviacion_max: float = 0.15,
    rango_max: float = 0.30,
) -> Concordancia:
    """¿Las predicciones sobre las distintas fotos dicen lo mismo?

    Umbrales provisionales: hay que calibrarlos comprobando si el desacuerdo
    predice de verdad el error, no suponiéndolo.
    """
    p = np.asarray(probabilidades, dtype=float)
    if p.size == 0:
        raise ValueError("Hacen falta al menos una predicción.")

    desv = float(p.std(ddof=1)) if p.size > 1 else 0.0
    rango = float(p.max() - p.min())
    coherente = desv <= desviacion_max and rango <= rango_max

    if p.size == 1:
        mensaje = ("Una sola foto: sin segunda opinión no hay forma de estimar la "
                   "incertidumbre. Añade dos más desde ángulos distintos.")
    elif coherente:
        mensaje = "Las fotos coinciden entre sí."
    else:
        mensaje = ("Las fotos no coinciden entre sí, así que no se da un resultado. "
                   "Repítelas con luz uniforme, encuadrando la misma zona y sin "
                   "cambiar mucho la distancia.")

    return Concordancia(
        n=int(p.size),
        media=round(float(p.mean()), 4),
        desviacion=round(desv, 4),
        rango=round(rango, 4),
        coherente=bool(coherente),
        mensaje=mensaje,
    )


def _histograma_hs(img: np.ndarray) -> np.ndarray:
    """Histograma de tono y saturación, normalizado."""
    peq = normalizar_escala(img, 256)
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def mismo_sujeto(imagenes: list[np.ndarray], similitud_min: float = 0.55) -> dict:
    """¿Las fotos son de la misma zona, o alguien fotografió cosas distintas?

    Comparación por correlación de histogramas de tono y saturación. Es
    deliberadamente tosca: detecta el caso real —fotos de sitios distintos— sin
    exigir que sean casi idénticas, porque la idea es justamente variar el ángulo.
    """
    if len(imagenes) < 2:
        return {"n": len(imagenes), "similitud_minima": 1.0, "coherente": True,
                "mensaje": "Una sola foto; no hay nada que comparar."}

    hists = [_histograma_hs(i) for i in imagenes]
    similitudes = []
    for a in range(len(hists)):
        for b in range(a + 1, len(hists)):
            similitudes.append(float(cv2.compareHist(hists[a], hists[b], cv2.HISTCMP_CORREL)))

    minima = float(min(similitudes))
    coherente = minima >= similitud_min
    return {
        "n": len(imagenes),
        "similitud_minima": round(minima, 4),
        "similitud_media": round(float(np.mean(similitudes)), 4),
        "coherente": coherente,
        "mensaje": ("Las fotos parecen de la misma zona." if coherente else
                    "Alguna foto parece de otra zona o con luz muy distinta. "
                    "Asegúrate de fotografiar la misma lesión en las tres."),
    }
