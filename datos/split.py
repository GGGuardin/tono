"""División train/val/test agrupando POR PACIENTE.

En PAD-UFES-20 una misma persona aporta varias lesiones. Dividir por imagen
metería fotos del mismo paciente —misma piel, misma cámara, misma iluminación— a
ambos lados, y el modelo memorizaría a la persona en lugar de aprender la lesión.
La aserción de abajo aborta la ejecución si eso ocurre: es preferible un error
ruidoso a un resultado inflado que nadie detecta.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("tono.datos.split")


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Añade la columna `split`, agrupando por paciente y estratificando por clase."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()

    def _holdout(frame: pd.DataFrame, frac: float, rs: int) -> np.ndarray:
        n_splits = max(2, int(round(1.0 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        _, hold = next(sgkf.split(frame, frame["label"].values, frame["patient_id"].values))
        return hold

    test_pos = _holdout(df, test_size, seed)
    pacientes_test = set(df.iloc[test_pos]["patient_id"])

    resto = df[~df["patient_id"].isin(pacientes_test)].reset_index(drop=True)
    val_frac = val_size / max(1e-9, 1.0 - test_size)
    val_pos = _holdout(resto, val_frac, seed + 1)
    pacientes_val = set(resto.iloc[val_pos]["patient_id"])

    df["split"] = "train"
    df.loc[df["patient_id"].isin(pacientes_val), "split"] = "val"
    df.loc[df["patient_id"].isin(pacientes_test), "split"] = "test"

    assert_no_patient_leakage(df)
    return df


def assert_no_patient_leakage(df: pd.DataFrame) -> None:
    """Falla ruidosamente si algún paciente aparece en más de un split."""
    por_paciente = df.groupby("patient_id")["split"].nunique()
    infractores = por_paciente[por_paciente > 1]
    if len(infractores):
        raise AssertionError(
            f"FUGA DE DATOS: {len(infractores)} pacientes en más de un split "
            f"(p.ej. {list(infractores.index[:5])})."
        )


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split").agg(
        imagenes=("image_path", "count"),
        pacientes=("patient_id", "nunique"),
        malignos=("label", "sum"),
    )
    g["prevalencia"] = (g["malignos"] / g["imagenes"]).round(4)
    return g.loc[[s for s in ["train", "val", "test"] if s in g.index]]
