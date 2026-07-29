"""Pruebas del portero con imágenes sintéticas que rompen cada control a propósito.

Cada caso construye el defecto concreto y comprueba que se detecta *ese* y no
otro. Sin esto, los umbrales serían números sin evidencia de que hagan nada.

    python tono/tests/test_calidad.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calidad import (  # noqa: E402
    Veredicto,
    concordancia_predicciones,
    corregir_con_referencia,
    evaluar,
    mismo_sujeto,
)

RNG = np.random.default_rng(7)
FALLOS: list[str] = []


def comprobar(nombre: str, condicion: bool, detalle: str = "") -> None:
    if condicion:
        print(f"  OK   {nombre}")
    else:
        print(f"  FALLA {nombre} {detalle}")
        FALLOS.append(nombre)


def foto_buena(lado: int = 800) -> np.ndarray:
    """Base sintética con textura y una 'lesión' central: pasa todos los controles."""
    yy, xx = np.mgrid[0:lado, 0:lado].astype(np.float32)
    # Piel con degradado suave y grano
    base = 150 + 18 * np.sin(xx / 90.0) + 12 * np.cos(yy / 70.0)
    base += RNG.normal(0, 9, base.shape)
    # Lesión central más oscura y con borde irregular
    r = np.hypot(xx - lado / 2, yy - lado / 2)
    lesion = 70 * np.exp(-(r ** 2) / (2 * (lado * 0.09) ** 2))
    gris = np.clip(base - lesion, 0, 255).astype(np.uint8)
    # Estructura fina extra, para que la nitidez sea claramente alta
    ruido = RNG.normal(0, 6, gris.shape)
    gris = np.clip(gris.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    # Tono de piel: más rojo que azul, sin llegar a tinte extremo
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.10, 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] * 0.93, 0, 255)
    return img


def main() -> None:
    print("\n--- referencia: foto correcta ---")
    buena = foto_buena()
    inf = evaluar(buena)
    comprobar("una foto correcta se acepta",
              inf.veredicto is Veredicto.ACEPTADA,
              f"-> {inf.veredicto.value}: {inf.problemas}")

    print("\n--- cada defecto por separado ---")

    desenfocada = cv2.GaussianBlur(buena, (0, 0), 9)
    inf = evaluar(desenfocada)
    comprobar("desenfoque se rechaza", inf.veredicto is Veredicto.RECHAZADA)
    comprobar("y el motivo es la nitidez",
              any("nitidez" in p["regla"] for p in inf.problemas),
              f"-> {[p['regla'] for p in inf.problemas]}")

    oscura = (buena.astype(np.float32) * 0.16).astype(np.uint8)
    inf = evaluar(oscura)
    comprobar("falta de luz se rechaza", inf.veredicto is Veredicto.RECHAZADA)
    comprobar("y el motivo es la exposición",
              any(p["regla"] in {"oscura", "plana"} for p in inf.problemas),
              f"-> {[p['regla'] for p in inf.problemas]}")

    quemada = np.clip(buena.astype(np.float32) * 2.6, 0, 255).astype(np.uint8)
    inf = evaluar(quemada)
    comprobar("sobreexposición se rechaza", inf.veredicto is Veredicto.RECHAZADA,
              f"-> {[p['regla'] for p in inf.problemas]}")

    con_reflejo = buena.copy()
    cv2.circle(con_reflejo, (400, 400), 130, (255, 255, 255), -1)
    inf = evaluar(con_reflejo)
    comprobar("reflejo de flash se rechaza", inf.veredicto is Veredicto.RECHAZADA)
    comprobar("y el motivo es el reflejo",
              any(p["regla"] == "reflejo" for p in inf.problemas),
              f"-> {[p['regla'] for p in inf.problemas]}")

    plana = np.full_like(buena, 128)
    inf = evaluar(plana)
    comprobar("imagen sin detalle se rechaza", inf.veredicto is Veredicto.RECHAZADA,
              f"-> {[p['regla'] for p in inf.problemas]}")

    pequena = cv2.resize(buena, (180, 180), interpolation=cv2.INTER_AREA)
    inf = evaluar(pequena)
    comprobar("resolución insuficiente se rechaza",
              any(p["regla"] == "resolucion" for p in inf.problemas),
              f"-> {[p['regla'] for p in inf.problemas]}")

    tenido = buena.copy().astype(np.float32)
    tenido[:, :, 0] *= 2.1  # tinte azul fuerte
    tenido = np.clip(tenido, 0, 255).astype(np.uint8)
    inf = evaluar(tenido)
    comprobar("tinte de color extremo se detecta",
              any("tinte" in p["regla"] for p in inf.problemas),
              f"-> {[p['regla'] for p in inf.problemas]}")

    print("\n--- el consejo es accionable ---")
    inf = evaluar(desenfocada)
    comprobar("el mensaje explica qué hacer",
              "enfoc" in inf.mensaje().lower() or "pulso" in inf.mensaje().lower()
              or "apoya" in inf.mensaje().lower(),
              f"-> {inf.mensaje()!r}")

    print("\n--- varias fotos ---")
    c = concordancia_predicciones([0.71, 0.74, 0.69])
    comprobar("predicciones parecidas se consideran coherentes", c.coherente)
    c = concordancia_predicciones([0.10, 0.55, 0.93])
    comprobar("predicciones dispares NO se consideran coherentes", not c.coherente)
    comprobar("y el mensaje evita dar resultado",
              "no se da un resultado" in c.mensaje, f"-> {c.mensaje!r}")

    variantes = [buena, cv2.GaussianBlur(buena, (0, 0), 1.2),
                 np.clip(buena.astype(np.float32) * 1.06, 0, 255).astype(np.uint8)]
    r = mismo_sujeto(variantes)
    comprobar("tres tomas de la misma zona se reconocen como tal", r["coherente"],
              f"-> similitud {r['similitud_minima']}")

    otra_cosa = RNG.integers(0, 255, buena.shape, dtype=np.uint8)
    r = mismo_sujeto([buena, otra_cosa])
    comprobar("fotos de zonas distintas se detectan", not r["coherente"],
              f"-> similitud {r['similitud_minima']}")

    print("\n--- corrección de color con referencia ---")
    con_tarjeta = buena.copy()
    con_tarjeta[10:90, 10:90] = (200, 200, 200)          # parche neutro
    con_tarjeta = np.clip(con_tarjeta.astype(np.float32) * [1.35, 1.0, 0.8],
                          0, 255).astype(np.uint8)        # tinte azul del ambiente
    corregida = corregir_con_referencia(con_tarjeta, (10, 10, 80, 80))
    parche = corregida[10:90, 10:90].reshape(-1, 3).mean(axis=0)
    comprobar("el parche de referencia queda neutro tras corregir",
              float(parche.max() - parche.min()) < 3.0,
              f"-> medias BGR {parche.round(1)}")

    print("\n" + "=" * 66)
    if FALLOS:
        print(f"FALLARON {len(FALLOS)} comprobaciones: {', '.join(FALLOS)}")
        raise SystemExit(1)
    print("TODAS LAS COMPROBACIONES PASAN")
    print("=" * 66)


if __name__ == "__main__":
    main()
