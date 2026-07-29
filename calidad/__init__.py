"""Portero de calidad para fotos tomadas con la cámara de un móvil.

PROYECTO EDUCATIVO Y EXPERIMENTAL. No es un dispositivo médico.

Su único trabajo es decidir si una foto sirve para analizarse y, si no sirve,
decir qué hacer para arreglarla. Es independiente del modelo y del dominio: vale
igual para piel, ojo externo o boca.
"""

from .checks import (
    corregir_balance_gris,
    corregir_con_referencia,
    dominante_color,
    encuadre,
    exposicion,
    medir_todo,
    nitidez,
    reflejos,
    resolucion,
)
from .gate import Informe, Umbrales, Veredicto, evaluar
from .multi import Concordancia, concordancia_predicciones, mismo_sujeto

__all__ = [
    "Concordancia",
    "Informe",
    "Umbrales",
    "Veredicto",
    "concordancia_predicciones",
    "corregir_balance_gris",
    "corregir_con_referencia",
    "dominante_color",
    "encuadre",
    "evaluar",
    "exposicion",
    "medir_todo",
    "mismo_sujeto",
    "nitidez",
    "reflejos",
    "resolucion",
]
__version__ = "0.1.0"
