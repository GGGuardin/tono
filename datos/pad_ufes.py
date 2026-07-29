"""Lector de PAD-UFES-20 hacia el manifiesto estándar del proyecto.

PAD-UFES-20 (Pacheco et al., *Data in Brief* 2020) es la pieza que hace viable la
Fase 2, y por cuatro razones que rara vez coinciden:

1. **Son fotos de móvil**, no imágenes de dermatoscopio: el dato coincide con el
   caso de uso real.
2. Trae **tipo de Fitzpatrick** por paciente, que es la variable sin la cual no
   hay auditoría de equidad posible.
3. Trae **identificador de paciente**, con varias lesiones por persona: el split
   por paciente deja de ser un formalismo.
4. Marca qué casos están **confirmados por biopsia**, así que se puede evaluar
   sobre verdad histológica en lugar de sobre impresión clínica.

Salida: el mismo manifiesto plano que consumen `train.py`, `evaluate.py` y
`fairness.py` del proyecto de tórax, más las columnas propias de dermatología.
Esa compatibilidad es deliberada: el pipeline ya está probado y no se reescribe.

    python -m datos.pad_ufes --root /kaggle/input/skin-cancer --out manifiesto.csv
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("tono.datos")

# Los seis diagnósticos de PAD-UFES-20. Tres son cáncer y tres no, lo que da una
# tarea binaria limpia y clínicamente significativa: ¿esto hay que biopsiarlo?
MALIGNOS = {"BCC", "SCC", "MEL"}          # carcinoma basocelular, espinocelular, melanoma
BENIGNOS = {"ACK", "NEV", "SEK"}          # queratosis actínica, nevo, queratosis seborreica

NOMBRES = {
    "BCC": "carcinoma basocelular",
    "SCC": "carcinoma espinocelular",
    "MEL": "melanoma",
    "ACK": "queratosis actínica",
    "NEV": "nevo",
    "SEK": "queratosis seborreica",
}


def _localizar_metadatos(root: Path) -> Path:
    """Busca el CSV de metadatos por su contenido, no por su nombre.

    Las copias que circulan renombran el fichero, así que se identifica por tener
    la columna de diagnóstico. Codificar la ruta a mano ya salió mal una vez.
    """
    candidatos = sorted(root.rglob("*.csv"))
    for ruta in candidatos:
        try:
            cabecera = pd.read_csv(ruta, nrows=0).columns.str.lower()
        except Exception:
            continue
        if "diagnostic" in cabecera and any("img" in c for c in cabecera):
            return ruta
    raise FileNotFoundError(
        f"No encuentro el CSV de metadatos bajo {root}. "
        f"CSV vistos: {[c.name for c in candidatos[:10]]}"
    )


def _paciente_de(nombre: str) -> str:
    """PAT_100_393_595.png -> PAT_100 (el paciente, no la lesión)."""
    m = re.match(r"(PAT_\d+)", nombre, flags=re.IGNORECASE)
    return m.group(1).upper() if m else nombre


def construir_manifiesto(root: str | Path, solo_biopsiados: bool = False) -> pd.DataFrame:
    """Manifiesto normalizado a partir de PAD-UFES-20."""
    root = Path(root)
    csv = _localizar_metadatos(root)
    meta = pd.read_csv(csv)
    meta.columns = [c.strip().lower() for c in meta.columns]
    log.info("Metadatos: %s (%d filas, %d columnas)", csv.name, len(meta), len(meta.columns))

    col_img = next((c for c in meta.columns if c in {"img_id", "image", "img"}), None)
    if col_img is None:
        raise KeyError(f"No hay columna de imagen en {list(meta.columns)}")

    log.info("Indexando imágenes bajo %s ...", root)
    rutas = {p.name: str(p) for p in root.rglob("*.png")}
    rutas.update({p.name: str(p) for p in root.rglob("*.jpg")})
    log.info("%d imágenes localizadas", len(rutas))

    antes = len(meta)
    meta = meta[meta[col_img].isin(rutas)].copy()
    if len(meta) < antes:
        log.warning("%d filas del CSV sin imagen correspondiente: descartadas", antes - len(meta))
    if meta.empty:
        raise FileNotFoundError("Ninguna fila del CSV casa con las imágenes encontradas.")

    diagnostico = meta["diagnostic"].astype(str).str.strip().str.upper()
    desconocidos = set(diagnostico) - MALIGNOS - BENIGNOS
    if desconocidos:
        log.warning("Diagnósticos no clasificados, descartados: %s", desconocidos)
        mantener = diagnostico.isin(MALIGNOS | BENIGNOS)
        meta, diagnostico = meta[mantener], diagnostico[mantener]

    if solo_biopsiados and "biopsed" in meta.columns:
        biopsiado = meta["biopsed"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
        log.info("Solo biopsiados: %d -> %d", len(meta), int(biopsiado.sum()))
        meta, diagnostico = meta[biopsiado], diagnostico[biopsiado]

    # El campo viene mal escrito en el dataset original ("fitspatrick")
    col_fitz = next((c for c in meta.columns if "fitspatrick" in c or "fitzpatrick" in c), None)
    if col_fitz is None:
        log.warning("Sin columna de Fitzpatrick: NO se podrá auditar por tono de piel")
        fitz = pd.Series("UNKNOWN", index=meta.index)
    else:
        n = pd.to_numeric(meta[col_fitz], errors="coerce")
        fitz = n.map(lambda v: f"{int(v)}" if pd.notna(v) and 1 <= v <= 6 else "UNKNOWN")

    col_pac = next((c for c in meta.columns if c in {"patient_id", "patient"}), None)
    pacientes = (meta[col_pac].astype(str).values if col_pac
                 else [_paciente_de(n) for n in meta[col_img]])

    edad = pd.to_numeric(meta.get("age"), errors="coerce") if "age" in meta.columns else np.nan
    sexo = (meta["gender"].astype(str).str.upper().str[0]
            if "gender" in meta.columns else "UNKNOWN")

    df = pd.DataFrame({
        "image_path": meta[col_img].map(rutas).values,
        "patient_id": pacientes,
        "label": diagnostico.isin(MALIGNOS).astype(int).values,
        "source": "pad_ufes_20",
        # `view` no aplica en dermatología; se conserva la columna por compatibilidad
        # con el pipeline y se usa para la región corporal, que sí es informativa.
        "view": (meta["region"].astype(str).values if "region" in meta.columns else "UNKNOWN"),
        "sex": sexo,
        "age": edad,
        "fitzpatrick": fitz.values,
        "diagnostico": diagnostico.values,
        "biopsiado": (meta["biopsed"].astype(str).values if "biopsed" in meta.columns
                      else "UNKNOWN"),
    })

    log.info("Manifiesto: %d imágenes, %d pacientes, %.1f%% malignos",
             len(df), df["patient_id"].nunique(), 100 * df["label"].mean())
    log.info("Por diagnóstico:\n%s", df["diagnostico"].value_counts().to_string())
    log.info("Por Fitzpatrick:\n%s", df["fitzpatrick"].value_counts().sort_index().to_string())
    return df.sort_values("image_path").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PAD-UFES-20 -> manifiesto + split por paciente")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--solo-biopsiados", action="store_true",
                    help="Restringe a casos con confirmación histológica")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    df = construir_manifiesto(args.root, args.solo_biopsiados)

    from .split import make_splits, split_summary

    df = make_splits(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Guardado en %s", out)
    print(split_summary(df).to_string())
    print("\nFitzpatrick por split:")
    print(pd.crosstab(df["fitzpatrick"], df["split"]).to_string())


if __name__ == "__main__":
    main()
