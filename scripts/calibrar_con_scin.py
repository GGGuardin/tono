"""Calibra el portero de calidad contra el juicio de dermatólogos, usando SCIN.

Hasta ahora los umbrales estaban puestos con criterio pero sin datos. Aquí se
miden fotos reales de voluntarios y se compara el veredicto del portero con la
etiqueta de un dermatólogo sobre si la imagen tenía **calidad suficiente para
evaluarla**.

Las imágenes se descargan y se decodifican **en memoria**, sin tocar el disco en
ningún momento: el equipo no acumula nada. Cada descarga lleva tiempo de espera y
reintento, y el CSV se guarda cada 25 medidas para que un corte de red no tire por
la borda el trabajo hecho.

    python scripts/calibrar_con_scin.py --n 1200
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from calidad import evaluar  # noqa: E402
from calidad.checks import medir_todo  # noqa: E402
from datos.scin import construir_manifiesto  # noqa: E402


def medir_una(url: str, timeout: float = 20.0, intentos: int = 2) -> dict | None:
    """Descarga en memoria, mide, y devuelve None si no hay forma.

    La imagen se decodifica desde el búfer sin tocar el disco. Y **con tiempo de
    espera explícito**: la primera versión usaba `urlretrieve`, que no lo tiene,
    y una sola conexión colgada dejó el proceso 34 minutos parado sin consumir
    CPU. Un fallo de red no debe poder detener una tanda de mil imágenes.
    """
    for intento in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as respuesta:
                datos = respuesta.read()
            img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            medidas = medir_todo(img)
            informe = evaluar(img)
            return {**medidas, "veredicto": informe.veredicto.value,
                    "reglas": "|".join(p["regla"] for p in informe.problemas)}
        except Exception:
            if intento == intentos - 1:
                return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="Imágenes a medir (mitad de cada clase)")
    ap.add_argument("--out", default="results/calibracion_scin.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--solo-unanimes", action="store_true",
                    help="Usa solo casos donde varios dermatólogos coinciden")
    args = ap.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    manifiesto = construir_manifiesto(RAIZ.parent / ".scin")
    if args.solo_unanimes:
        manifiesto = manifiesto[manifiesto["unanime"]]
        print(f"Solo casos unánimes: {len(manifiesto)}")

    # Muestra equilibrada: con 39% de no evaluables, una muestra al azar bastaría,
    # pero equilibrar da más potencia en la clase minoritaria sin coste extra.
    rng = np.random.default_rng(args.seed)
    por_clase = args.n // 2
    partes = []
    for clase in (0, 1):
        sub = manifiesto[manifiesto["evaluable"] == clase]
        idx = rng.permutation(len(sub))[:min(por_clase, len(sub))]
        partes.append(sub.iloc[idx])
    muestra = pd.concat(partes).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    (RAIZ / args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Midiendo {len(muestra)} imágenes en streaming (pico de disco: 1 imagen)\n")

    filas, fallos = [], 0
    for i, fila in muestra.iterrows():
        m = medir_una(fila["image_url"])
        if m is None:
            fallos += 1
        else:
            filas.append({
                "case_id": fila["case_id"],
                "evaluable_derm": int(fila["evaluable"]),
                "fitzpatrick_derm": fila["fitzpatrick_derm"],
                "shot_type": fila["shot_type"],
                **m,
            })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(muestra)}  medidas={len(filas)}  fallos={fallos}", flush=True)
            pd.DataFrame(filas).to_csv(RAIZ / args.out, index=False)

    df = pd.DataFrame(filas)
    out = RAIZ / args.out
    df.to_csv(out, index=False)
    print(f"\n{len(df)} medidas guardadas en {out} ({fallos} imágenes ilegibles)")


if __name__ == "__main__":
    main()
