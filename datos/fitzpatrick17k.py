"""Lector de Fitzpatrick17k hacia el manifiesto estándar.

Es el dataset que desbloquea la pregunta central del proyecto. Comparación cruda
de representación de piel oscura (tonos 4-6):

| Dataset          | Imágenes | Malignas |
|------------------|----------|----------|
| PAD-UFES-20      |        6 |        6 |
| **Fitzpatrick17k** | **1.079** | **509** |

Tarea binaria sobre el subconjunto neoplásico: **maligno** frente a **benigno**,
4.497 imágenes casi equilibradas. Se descartan las 12.080 no neoplásicas
(inflamatorias, genodermatosis) porque la pregunta clínica es la misma que en
PAD-UFES: ¿esto hay que biopsiarlo?

## El tono se agrupa en bandas, y no es un capricho

El dataset trae dos anotaciones independientes de Fitzpatrick, y **coinciden
exactamente solo el 47,9% de las veces**; ±1 tono, el 91% (correlación 0,810).
Tratarlo como seis clases exactas sería medir sobre todo el ruido del anotador.
Agrupar en tres bandas —clara, media, oscura— se apoya en lo que la anotación sí
resuelve de forma fiable.

## Salvedades que hay que decir al reportar

- Las imágenes vienen de dos atlas dermatológicos en línea, así que el
  diagnóstico es **clínico o de libro, no confirmado por biopsia**. Es una
  verdad de referencia más débil que la de DDI.
- Los tonos los anotó personal **no dermatólogo**, con el desacuerdo citado.
- Son fotos de atlas, no fotos de móvil: el dominio no coincide con el caso de
  uso real, a diferencia de PAD-UFES y SCIN.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("tono.datos.fitz")

BANDAS = {1: "1-2 clara", 2: "1-2 clara", 3: "3-4 media", 4: "3-4 media",
          5: "5-6 oscura", 6: "5-6 oscura"}


def construir_manifiesto(
    root: str | Path,
    solo_neoplasico: bool = True,
    excluir_mal_etiquetadas: bool = True,
) -> pd.DataFrame:
    """Manifiesto normalizado a partir de Fitzpatrick17k."""
    root = Path(root)
    csvs = [p for p in root.rglob("*.csv") if "fitzpatrick" in p.name.lower()]
    if not csvs:
        raise FileNotFoundError(f"No encuentro el CSV de Fitzpatrick17k bajo {root}")
    meta = pd.read_csv(csvs[0])
    log.info("Metadatos: %s (%d filas)", csvs[0].name, len(meta))

    if solo_neoplasico:
        antes = len(meta)
        meta = meta[meta["three_partition_label"].isin(["malignant", "benign"])]
        log.info("Subconjunto neoplasico: %d -> %d", antes, len(meta))

    if excluir_mal_etiquetadas and "qc" in meta.columns:
        malas = meta["qc"].astype(str).str.contains("Wrongly", case=False, na=False)
        if malas.any():
            log.info("Descartadas %d marcadas por un dermatologo como mal etiquetadas",
                     int(malas.sum()))
            meta = meta[~malas]

    log.info("Indexando imagenes bajo %s ...", root)
    rutas = {p.stem: str(p) for p in root.rglob("*.jpg")}
    rutas.update({p.stem: str(p) for p in root.rglob("*.png")})
    log.info("%d imagenes localizadas", len(rutas))

    antes = len(meta)
    meta = meta[meta["md5hash"].isin(rutas)].copy()
    if len(meta) < antes:
        log.warning("%d filas sin imagen: descartadas", antes - len(meta))
    if meta.empty:
        raise FileNotFoundError("Ninguna fila casa con las imagenes encontradas.")

    tono = pd.to_numeric(meta["fitzpatrick_scale"], errors="coerce")
    banda = tono.map(BANDAS).fillna("UNKNOWN")

    df = pd.DataFrame({
        "image_path": meta["md5hash"].map(rutas).values,
        # No hay identificador de paciente: son imagenes de atlas, cada una de un
        # caso distinto. Se usa el hash, de modo que el split por "paciente"
        # equivale al split por imagen. Es una limitacion real, no un descuido.
        "patient_id": meta["md5hash"].astype(str).values,
        "label": (meta["three_partition_label"] == "malignant").astype(int).values,
        "source": "fitzpatrick17k",
        "view": meta["nine_partition_label"].astype(str).values,
        "sex": "UNKNOWN",
        "age": np.nan,
        "fitzpatrick": banda.values,
        "fitzpatrick_exacto": tono.fillna(-1).astype(int).astype(str).values,
        "diagnostico": meta["label"].astype(str).values,
    })

    log.info("Manifiesto: %d imagenes, %.1f%% malignas", len(df), 100 * df["label"].mean())
    log.info("Por banda de tono:\n%s",
             pd.crosstab(df["fitzpatrick"], df["label"]).to_string())
    return df.sort_values("image_path").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fitzpatrick17k -> manifiesto + split")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--todas", action="store_true",
                    help="No restringir al subconjunto neoplasico")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    df = construir_manifiesto(args.root, solo_neoplasico=not args.todas)

    from .split import make_splits, split_summary

    df = make_splits(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    salida = Path(args.out)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    log.info("Guardado en %s", salida)
    print(split_summary(df).to_string())
    print("\nBanda de tono por split:")
    print(pd.crosstab(df["fitzpatrick"], df["split"]).to_string())


if __name__ == "__main__":
    main()
