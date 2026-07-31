"""Une varios manifiestos en uno, rehaciendo el split por paciente.

Fitzpatrick17k son imágenes de atlas; PAD-UFES-20 son fotos de móvil. Entrenar
con ambas obliga al modelo a no depender del estilo de imagen de una sola fuente,
que es justo el fallo que se midió cuando el modelo de PAD-UFES se estrelló al
aplicarlo a Fitzpatrick17k (AUROC 0,899 -> 0,684).

La tarea binaria coincide en ambos —maligno frente a benigno— así que las
etiquetas son compatibles sin traducción. El tono de piel se normaliza a las
mismas tres bandas.

El split se rehace sobre el conjunto unido para que ningún paciente cruce, y se
conserva la columna `source` para poder evaluar por origen por separado: un
promedio entre dos dominios distintos esconde si uno de los dos va mal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("tono.combinar")

BANDAS = {"1": "1-2 clara", "2": "1-2 clara", "3": "3-4 media",
          "4": "3-4 media", "5": "5-6 oscura", "6": "5-6 oscura"}


def normalizar_tono(valor) -> str:
    """Lleva cualquier codificación de Fitzpatrick a las tres bandas."""
    s = str(valor).strip()
    if s in BANDAS:
        return BANDAS[s]
    if "clara" in s or "media" in s or "oscura" in s:
        return s
    try:
        return BANDAS.get(str(int(float(s))), "UNKNOWN")
    except (ValueError, TypeError):
        return "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser(description="Une manifiestos y rehace el split")
    ap.add_argument("--entradas", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    partes = []
    for ruta in args.entradas:
        d = pd.read_csv(ruta)
        d["fitzpatrick"] = d["fitzpatrick"].map(normalizar_tono)
        # El identificador de paciente se prefija con el origen: dos datasets
        # distintos pueden reutilizar el mismo número y fundirlos crearía
        # pacientes falsos que agruparían imágenes sin relación.
        d["patient_id"] = d["source"].astype(str) + "_" + d["patient_id"].astype(str)
        log.info("%s: %d imagenes, %.1f%% malignas", Path(ruta).name, len(d),
                 100 * d["label"].mean())
        partes.append(d)

    columnas = ["image_path", "patient_id", "label", "source", "view", "sex", "age",
                "fitzpatrick"]
    unido = pd.concat([p[columnas] for p in partes], ignore_index=True)

    from datos.split import make_splits, split_summary

    unido = make_splits(unido, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    unido.to_csv(args.out, index=False)

    log.info("Unido: %d imagenes, %d pacientes", len(unido), unido["patient_id"].nunique())
    print(split_summary(unido).to_string())
    print("\nOrigen por split:")
    print(pd.crosstab(unido["source"], unido["split"]).to_string())
    print("\nBanda de tono por split:")
    print(pd.crosstab(unido["fitzpatrick"], unido["split"]).to_string())


if __name__ == "__main__":
    main()
