"""Pasa el portero por una carpeta de fotos y saca una tabla de medidas.

Es la herramienta para **calibrar los umbrales con datos en vez de con intuición**:
mides una tanda de fotos, miras la distribución de cada medida, y colocas los
cortes donde de verdad separan lo usable de lo inservible.

    python -m calidad.cli --carpeta fotos/ --csv medidas.csv
    python -m calidad.cli --carpeta fotos/ --umbral nitidez_min=0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .gate import Umbrales, evaluar

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalúa la calidad de una carpeta de fotos")
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--csv", default=None, help="Guarda las medidas para calibrar umbrales")
    ap.add_argument("--json", default=None, help="Guarda los informes completos")
    ap.add_argument("--umbral", action="append", default=[],
                    help="Sobrescribe un umbral, p.ej. --umbral nitidez_min=0.02")
    args = ap.parse_args()

    umbrales = Umbrales()
    for asignacion in args.umbral:
        clave, _, valor = asignacion.partition("=")
        if not hasattr(umbrales, clave):
            ap.error(f"Umbral desconocido: {clave}. Disponibles: "
                     f"{', '.join(vars(umbrales))}")
        actual = getattr(umbrales, clave)
        setattr(umbrales, clave, type(actual)(valor))

    carpeta = Path(args.carpeta)
    rutas = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not rutas:
        raise SystemExit(f"No encuentro imágenes en {carpeta}")

    filas, informes = [], []
    conteo = {"aceptada": 0, "dudosa": 0, "rechazada": 0}

    for ruta in rutas:
        img = cv2.imread(str(ruta))
        if img is None:
            print(f"  [ilegible] {ruta.name}")
            continue
        inf = evaluar(img, umbrales)
        conteo[inf.veredicto.value] += 1
        filas.append({"archivo": ruta.name, "veredicto": inf.veredicto.value, **inf.medidas})
        informes.append({"archivo": str(ruta), **inf.to_dict()})
        marca = {"aceptada": "OK ", "dudosa": "?? ", "rechazada": "NO "}[inf.veredicto.value]
        print(f"  {marca} {ruta.name}: {inf.mensaje()}")

    total = sum(conteo.values())
    print(f"\n{total} fotos: {conteo['aceptada']} aceptadas, "
          f"{conteo['dudosa']} dudosas, {conteo['rechazada']} rechazadas")
    if total:
        print(f"Tasa de rechazo: {conteo['rechazada'] / total:.1%}")
        print("Si esa tasa es altísima el portero es inusable; si es cero, es un adorno.")

    if args.csv and filas:
        import csv as _csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            escritor = _csv.DictWriter(f, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
        print(f"Medidas en {args.csv}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(informes, f, indent=2, ensure_ascii=False)
        print(f"Informes en {args.json}")


if __name__ == "__main__":
    main()
