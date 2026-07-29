"""Decisión de aceptar o rechazar una foto, con un motivo accionable.

Principio de diseño: **el portero solo bloquea por lo que impide analizar la
imagen**, nunca por lo que se parezca al sujeto. Y cuando rechaza, dice qué
hacer para arreglarlo — un "foto no válida" a secas hace que la gente reintente
lo mismo y se rinda.

Los umbrales de abajo son **provisionales**. Están puestos con criterio pero sin
calibrar contra fotos reales anotadas; `cli.py` existe precisamente para medir
una tanda de fotos y ajustarlos con datos. Cualquier número aquí que se presente
como definitivo sería inventado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from .checks import medir_todo


class Veredicto(str, Enum):
    ACEPTADA = "aceptada"
    DUDOSA = "dudosa"
    RECHAZADA = "rechazada"


@dataclass
class Umbrales:
    """Límites de aceptación. PROVISIONALES: calibrar con datos reales."""

    # Resolución mínima útil del lado corto
    lado_corto_min: int = 300

    # Nitidez normalizada (varianza del laplaciano / varianza de intensidad)
    nitidez_min: float = 0.010
    nitidez_dudosa: float = 0.020

    # Exposición
    brillo_min: float = 0.18
    brillo_max: float = 0.88
    quemada_max: float = 0.10
    apagada_max: float = 0.10
    rango_dinamico_min: float = 0.25

    # Reflejo del flash: mancha compacta saturada
    reflejo_max: float = 0.04

    # Encuadre
    estructura_min: float = 0.04
    ratio_centro_marco_min: float = 0.55

    # Tinte de color: bloquea solo lo extremo; lo moderado se avisa
    tinte_max: float = 0.45
    tinte_dudoso: float = 0.22


@dataclass
class Informe:
    veredicto: Veredicto
    medidas: dict
    problemas: list[dict] = field(default_factory=list)

    @property
    def aceptable(self) -> bool:
        return self.veredicto is not Veredicto.RECHAZADA

    def mensaje(self) -> str:
        """Un solo mensaje para la persona: el problema más importante."""
        if not self.problemas:
            return "Foto válida."
        return self.problemas[0]["consejo"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["veredicto"] = self.veredicto.value
        return d


# Cada regla: (clave, condición de fallo, gravedad, qué decirle a la persona)
def _reglas(m: dict, u: Umbrales) -> list[dict]:
    reglas = [
        (
            "resolucion",
            m["lado_corto"] < u.lado_corto_min,
            "rechazada",
            "La foto es demasiado pequeña. Usa la cámara del móvil en vez de una "
            "captura de pantalla o una imagen reenviada por mensajería.",
        ),
        (
            "nitidez",
            m["nitidez_normalizada"] < u.nitidez_min,
            "rechazada",
            "La foto está desenfocada. Apoya la mano o el codo en algo firme, toca "
            "la pantalla sobre la zona para que enfoque ahí, y no te acerques más "
            "de un palmo.",
        ),
        (
            "nitidez_justa",
            u.nitidez_min <= m["nitidez_normalizada"] < u.nitidez_dudosa,
            "dudosa",
            "La foto está algo blanda de foco. Si puedes, repítela con mejor pulso.",
        ),
        (
            "oscura",
            m["brillo_medio"] < u.brillo_min or m["fraccion_apagada"] > u.apagada_max,
            "rechazada",
            "Falta luz. Acércate a una ventana o enciende una lámpara, pero sin "
            "usar el flash directo.",
        ),
        (
            "quemada",
            m["brillo_medio"] > u.brillo_max or m["fraccion_quemada"] > u.quemada_max,
            "rechazada",
            "La foto está sobreexpuesta y se ha perdido el detalle. Apaga el flash "
            "y evita la luz directa del sol.",
        ),
        (
            "reflejo",
            m["fraccion_reflejo_mayor"] > u.reflejo_max,
            "rechazada",
            "Hay un reflejo brillante tapando parte de la zona. Apaga el flash e "
            "inclina un poco el móvil para que la luz no rebote hacia la cámara.",
        ),
        (
            "plana",
            m["rango_dinamico"] < u.rango_dinamico_min or m["estructura"] < u.estructura_min,
            "rechazada",
            "La imagen apenas tiene detalle: puede estar velada, muy comprimida o no "
            "contener la zona que quieres analizar. Repite la foto acercándote.",
        ),
        (
            "descentrada",
            m["ratio_centro_marco"] < u.ratio_centro_marco_min,
            "dudosa",
            "La zona de interés parece estar fuera del centro. Encuádrala en el "
            "medio y que ocupe la mayor parte de la foto.",
        ),
        (
            "tinte_extremo",
            m["tinte"] > u.tinte_max,
            "rechazada",
            "La luz tiñe demasiado los colores. Evita bombillas de colores o pantallas "
            "como única fuente de luz; la luz de día es la mejor.",
        ),
        (
            "tinte_moderado",
            u.tinte_dudoso < m["tinte"] <= u.tinte_max,
            "dudosa",
            "Los colores salen algo desviados por la luz del ambiente. El resultado "
            "puede ser menos fiable.",
        ),
    ]
    return [
        {"regla": clave, "gravedad": gravedad, "consejo": consejo}
        for clave, falla, gravedad, consejo in reglas
        if falla
    ]


def evaluar(img: np.ndarray, umbrales: Umbrales | None = None) -> Informe:
    """Mide una foto y decide si sirve para analizarla."""
    u = umbrales or Umbrales()
    medidas = medir_todo(img)
    problemas = _reglas(medidas, u)

    # Los rechazos primero: `mensaje()` devuelve el problema más grave
    problemas.sort(key=lambda p: 0 if p["gravedad"] == "rechazada" else 1)

    if any(p["gravedad"] == "rechazada" for p in problemas):
        veredicto = Veredicto.RECHAZADA
    elif problemas:
        veredicto = Veredicto.DUDOSA
    else:
        veredicto = Veredicto.ACEPTADA

    return Informe(veredicto=veredicto, medidas=medidas, problemas=problemas)
