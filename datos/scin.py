"""Lector de SCIN (Skin Condition Image Network, Google + Stanford, 2024).

~5.000 casos aportados por voluntarios **con su propio teléfono**, en un bucket
público de Google Cloud: sin registro, sin acuerdo que firmar.

Lo que lo hace valioso aquí no es el diagnóstico —SCIN es dermatología general y
apenas contiene cáncer (21 basocelulares, 7 melanomas)— sino dos etiquetas que
casi ningún dataset publica:

1. **Evaluabilidad juzgada por dermatólogo**: 1.925 casos marcados
   explícitamente como «calidad de imagen insuficiente». Es la verdad de
   referencia que necesita el portero de calidad, que hasta ahora tenía umbrales
   puestos a ojo.
2. **Tono de piel valorado por dermatólogo** (Fitzpatrick, 3 valoraciones
   independientes) además del autodeclarado por el paciente. Con 1.147 casos en
   tonos 4-6, permite lo que PAD-UFES-20 no permitía: comprobar si el portero
   rechaza más fotos de piel oscura.

Ese segundo punto importa: un filtro de calidad sesgado excluiría a un grupo
antes de que el modelo llegue a opinar, y sería un daño invisible en cualquier
métrica del clasificador.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("tono.datos.scin")

BASE = "https://storage.googleapis.com/dx-scin-public-data/"
CASES = BASE + "dataset/scin_cases.csv"
LABELS = BASE + "dataset/scin_labels.csv"

# El valor "DEFAULT_YES" indica que el dermatólogo no marcó la imagen como
# insuficiente, no que la aprobara de forma activa. La etiqueta informativa es la
# negativa: cuando dice NO, lo dice a propósito.
GRADABLE = {
    "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT": 1,
    "YES_IMAGE_QUALITY_SUFFICIENT_NO_DISCERNIBLE_PATHOLOGY": 1,
    "NO_IMAGE_QUALITY_INSUFFICIENT": 0,
}


def descargar_metadatos(destino: str | Path = ".scin") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga (o reutiliza) los dos CSV de SCIN."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre, url in [("scin_cases.csv", CASES), ("scin_labels.csv", LABELS)]:
        ruta = destino / nombre
        if not ruta.exists():
            import urllib.request

            log.info("Descargando %s ...", nombre)
            urllib.request.urlretrieve(url, ruta)
        rutas[nombre] = ruta
    return pd.read_csv(rutas["scin_cases.csv"]), pd.read_csv(rutas["scin_labels.csv"])


def construir_manifiesto(destino: str | Path = ".scin") -> pd.DataFrame:
    """Manifiesto con URL de imagen, evaluabilidad y tono de piel.

    La imagen no se descarga aquí: se guarda su URL para poder medirla en
    streaming sin acumular gigabytes en disco.
    """
    casos, etiquetas = descargar_metadatos(destino)
    df = casos.merge(etiquetas, on="case_id", how="inner", suffixes=("", "_lab"))

    grad = df["dermatologist_gradable_for_skin_condition_1"].map(GRADABLE)

    # Acuerdo entre valoradores, donde hay más de uno: sirve para quedarse solo
    # con los casos en que los dermatólogos coinciden, que son los que permiten
    # juzgar al portero sin discutir con el propio desacuerdo humano.
    otros = [df[f"dermatologist_gradable_for_skin_condition_{i}"].map(GRADABLE) for i in (2, 3)]
    n_valoraciones = grad.notna().astype(int) + sum(o.notna().astype(int) for o in otros)
    suma = grad.fillna(0) + sum(o.fillna(0) for o in otros)
    unanime = (n_valoraciones > 1) & ((suma == 0) | (suma == n_valoraciones))

    # Fitzpatrick: se prefiere la valoración del dermatólogo al autodeclarado,
    # porque es la que se usa en la literatura de equidad dermatológica.
    fitz_derm = (df["dermatologist_fitzpatrick_skin_type_label_1"]
                 .astype(str).str.extract(r"FST(\d)")[0])
    fitz_self = df["fitzpatrick_skin_type"].astype(str).str.extract(r"FST(\d)")[0]

    salida = pd.DataFrame({
        "case_id": df["case_id"].astype(str),
        "image_url": BASE + df["image_1_path"].astype(str),
        "evaluable": grad,
        "n_valoraciones": n_valoraciones,
        "unanime": unanime,
        "fitzpatrick_derm": fitz_derm,
        "fitzpatrick_self": fitz_self,
        "shot_type": df["image_1_shot_type"].astype(str),
    })
    salida = salida[salida["image_url"].str.endswith((".png", ".jpg", ".jpeg"))]
    salida = salida[salida["evaluable"].notna()].copy()
    salida["evaluable"] = salida["evaluable"].astype(int)

    log.info("SCIN: %d casos con etiqueta de evaluabilidad", len(salida))
    log.info("  evaluables: %d | no evaluables: %d",
             int(salida.evaluable.sum()), int((1 - salida.evaluable).sum()))
    log.info("  con valoración unánime de >1 dermatólogo: %d", int(salida.unanime.sum()))
    log.info("Fitzpatrick (dermatólogo):\n%s",
             salida.fitzpatrick_derm.value_counts(dropna=False).sort_index().to_string())
    return salida.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="SCIN -> manifiesto de evaluabilidad y tono")
    ap.add_argument("--destino", default=".scin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    df = construir_manifiesto(args.destino)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Guardado en %s", args.out)


if __name__ == "__main__":
    main()
