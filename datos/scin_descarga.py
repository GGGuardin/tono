"""Descarga SCIN en paralelo y produce el manifiesto estándar del pipeline.

Descargar 5.000 imágenes de una en una tarda más de una hora; con un grupo de
hilos baja a minutos, porque el cuello de botella es la latencia de red y no la
CPU. Cada imagen se reescala a 512 px de lado corto y se guarda como JPEG: pasa
de ~1 MB a ~60 KB sin perder nada relevante para juzgar calidad, y el
entrenamiento posterior lee mucho más rápido.

Etiqueta: **1 = el dermatólogo la consideró NO evaluable**. Se orienta así a
propósito, para que el modelo prediga «hay que rechazar esta foto» y las métricas
de equidad se lean directamente:

- **FNR** = fotos malas que el portero deja pasar.
- **FPR** = fotos buenas que el portero descarta. **Esta es la que importa para
  la equidad**: descartar fotos buenas de un grupo lo excluye del sistema.

    python -m datos.scin_descarga --out-dir /tmp/scin --manifiesto manifiesto_scin.csv
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger("tono.datos.scin")


def _descargar_una(url: str, destino: Path, lado: int, timeout: float,
                   intentos: int) -> bool:
    if destino.exists():
        return True
    for intento in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                datos = r.read()
            img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False
            h, w = img.shape[:2]
            corto = min(h, w)
            if corto > lado:
                f = lado / corto
                img = cv2.resize(img, (int(round(w * f)), int(round(h * f))),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(destino), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return True
        except Exception:
            if intento == intentos - 1:
                return False
    return False


def descargar(df: pd.DataFrame, out_dir: str | Path, lado: int = 512,
              hilos: int = 16, timeout: float = 20.0, intentos: int = 3) -> pd.DataFrame:
    """Descarga en paralelo y devuelve el df con la ruta local de cada imagen."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tareas = {}
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        for _, fila in df.iterrows():
            destino = out_dir / f"{fila['case_id']}.jpg"
            tareas[pool.submit(_descargar_una, fila["image_url"], destino, lado,
                               timeout, intentos)] = (fila["case_id"], destino)

        ok, fallos = 0, 0
        for i, futuro in enumerate(as_completed(tareas), 1):
            if futuro.result():
                ok += 1
            else:
                fallos += 1
            if i % 250 == 0:
                log.info("  %d/%d descargadas (%d fallos)", i, len(tareas), fallos)

    log.info("Descarga terminada: %d correctas, %d fallidas", ok, fallos)
    rutas = {cid: str(dest) for cid, dest in tareas.values() if Path(dest).exists()}
    df = df[df["case_id"].isin(rutas)].copy()
    df["image_path"] = df["case_id"].map(rutas)
    return df


def a_manifiesto(df: pd.DataFrame, seed: int = 42,
                 val_size: float = 0.15, test_size: float = 0.20) -> pd.DataFrame:
    """Convierte al formato que consumen train/evaluate/fairness."""
    from .split import make_splits

    salida = pd.DataFrame({
        "image_path": df["image_path"].values,
        # Un caso puede tener varias fotos: agrupar por caso evita que la misma
        # lesión y la misma cámara caigan a ambos lados del split.
        "patient_id": df["case_id"].astype(str).values,
        "label": (1 - df["evaluable"]).astype(int).values,   # 1 = NO evaluable
        "source": "scin",
        "view": df["shot_type"].astype(str).values,
        "sex": "UNKNOWN",
        "age": np.nan,
        "fitzpatrick": df["fitzpatrick_derm"].fillna("UNKNOWN").astype(str).values,
    })
    return make_splits(salida, val_size=val_size, test_size=test_size, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga SCIN y construye el manifiesto")
    ap.add_argument("--out-dir", default="/tmp/scin_imagenes")
    ap.add_argument("--manifiesto", required=True)
    ap.add_argument("--metadatos", default=".scin")
    ap.add_argument("--lado", type=int, default=512)
    ap.add_argument("--hilos", type=int, default=16)
    ap.add_argument("--limite", type=int, default=0, help="0 = todos")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    from .scin import construir_manifiesto

    df = construir_manifiesto(args.metadatos)
    if args.limite:
        df = df.sample(n=min(args.limite, len(df)), random_state=42)
    log.info("Descargando %d imágenes con %d hilos ...", len(df), args.hilos)

    df = descargar(df, args.out_dir, lado=args.lado, hilos=args.hilos)
    manifiesto = a_manifiesto(df)

    salida = Path(args.manifiesto)
    salida.parent.mkdir(parents=True, exist_ok=True)
    manifiesto.to_csv(salida, index=False)

    from .split import split_summary

    log.info("Guardado en %s", salida)
    print(split_summary(manifiesto).to_string())
    print("\nFitzpatrick por split:")
    print(pd.crosstab(manifiesto["fitzpatrick"], manifiesto["split"]).to_string())


if __name__ == "__main__":
    main()
