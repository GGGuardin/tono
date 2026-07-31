"""Calibra el umbral por subgrupo, eligiéndolo en validación y aplicándolo en test.

El hallazgo a corregir: el modelo discrimina igual de bien en piel oscura
(AUROC 0,910 frente a 0,900 en clara) pero al umbral compartido se pierde el
23,7% de los cánceres frente al 10-15% en las otras bandas. Mismo poder, peor
punto de operación.

**La regla que no se puede romper**: el umbral de cada grupo se deriva del
conjunto de **validación** y se aplica tal cual al de test. Ajustarlo sobre test
daría una mejora que no existiría al desplegarlo — sería medir sobre la respuesta.

Se comparan tres estrategias, todas calibradas en validación:

1. **Único**: un umbral para todos (lo que había).
2. **Por grupo**: un umbral por banda de tono, por índice de Youden.
3. **Sensibilidad fija**: el umbral de cada banda que alcanza una sensibilidad
   objetivo. En cribado de cáncer es lo defendible: se fija cuántos casos estás
   dispuesto a perder, igual para todos, y se acepta la especificidad que salga.

La tercera es la más honesta de las tres para este problema, porque **iguala el
daño** en lugar de igualar un número abstracto.

    python scripts/calibrar_por_grupo.py --val val.csv --test test.csv --grupo fitzpatrick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from scripts._rutas import anadir_al_path  # noqa: E402
anadir_al_path()

from src.metrics import binary_metrics, youden_threshold  # noqa: E402


def umbral_para_sensibilidad(y: np.ndarray, p: np.ndarray, objetivo: float) -> float:
    """Umbral más alto que aún alcanza la sensibilidad pedida."""
    from sklearn.metrics import roc_curve

    if len(np.unique(y)) < 2:
        return 0.5
    _, tpr, umbrales = roc_curve(y, p)
    ok = np.where(tpr >= objetivo)[0]
    return float(umbrales[ok[0]]) if len(ok) else float(umbrales[-1])


def _metricas_por_grupo(df: pd.DataFrame, grupo: str, umbrales: dict | float,
                        min_n: int) -> pd.DataFrame:
    filas = []
    for valor, parte in df.groupby(grupo):
        if len(parte) < min_n or parte["label"].nunique() < 2:
            continue
        u = umbrales[valor] if isinstance(umbrales, dict) else umbrales
        m = binary_metrics(parte["label"].values, parte["prob"].values, u)
        filas.append({"grupo": str(valor), "n": m["n"], "umbral": round(float(u), 4),
                      "auroc": round(m["auroc"], 4), "sensibilidad": round(m["sensibilidad"], 4),
                      "especificidad": round(m["especificidad"], 4),
                      "fnr": round(m["fnr"], 4)})
    return pd.DataFrame(filas).sort_values("grupo")


def calibrar(val: pd.DataFrame, test: pd.DataFrame, grupo: str = "fitzpatrick",
             sensibilidad_objetivo: float = 0.85, min_n: int = 25) -> dict:
    """Deriva umbrales en validación y los aplica a test."""
    # --- 1. Único, derivado en validación ---
    u_global = float(youden_threshold(val["label"].values, val["prob"].values))

    # --- 2 y 3. Por grupo, también derivados en validación ---
    u_grupo, u_sens = {}, {}
    for valor, parte in val.groupby(grupo):
        if len(parte) < min_n or parte["label"].nunique() < 2:
            # Sin datos suficientes en validación se cae al umbral global: es
            # preferible a inventarse un corte con cuatro casos.
            u_grupo[valor] = u_global
            u_sens[valor] = u_global
            continue
        u_grupo[valor] = float(youden_threshold(parte["label"].values, parte["prob"].values))
        u_sens[valor] = umbral_para_sensibilidad(parte["label"].values, parte["prob"].values,
                                                 sensibilidad_objetivo)

    # Grupos presentes en test pero no en validación
    for valor in test[grupo].unique():
        u_grupo.setdefault(valor, u_global)
        u_sens.setdefault(valor, u_global)

    estrategias = {
        "umbral_unico": u_global,
        "umbral_por_grupo": u_grupo,
        f"sensibilidad_{sensibilidad_objetivo:.2f}_por_grupo": u_sens,
    }

    salida = {"umbral_global_validacion": round(u_global, 4),
              "sensibilidad_objetivo": sensibilidad_objetivo, "estrategias": {}}

    for nombre, umbrales in estrategias.items():
        tabla = _metricas_por_grupo(test, grupo, umbrales, min_n)
        if tabla.empty:
            continue
        brecha = float(tabla["fnr"].max() - tabla["fnr"].min())
        salida["estrategias"][nombre] = {
            "brecha_fnr": round(brecha, 4),
            "fnr_maximo": round(float(tabla["fnr"].max()), 4),
            "peor_grupo": str(tabla.loc[tabla["fnr"].idxmax(), "grupo"]),
            "sensibilidad_global": round(float(
                binary_metrics(test["label"].values, test["prob"].values,
                               umbrales if isinstance(umbrales, float) else u_global
                               )["sensibilidad"]), 4),
            "tabla": tabla.to_dict("records"),
        }
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description="Umbral por subgrupo, calibrado en validación")
    ap.add_argument("--val", required=True, help="predictions.csv del split de validación")
    ap.add_argument("--test", required=True, help="predictions.csv del split de test")
    ap.add_argument("--grupo", default="fitzpatrick")
    ap.add_argument("--sensibilidad", type=float, default=0.85)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    val, test = pd.read_csv(args.val), pd.read_csv(args.test)
    r = calibrar(val, test, args.grupo, args.sensibilidad, args.min_n)

    print(f"Umbral global derivado en validacion: {r['umbral_global_validacion']}\n")
    for nombre, d in r["estrategias"].items():
        print("=" * 66)
        print(nombre.upper().replace("_", " "))
        print("=" * 66)
        print(pd.DataFrame(d["tabla"]).to_string(index=False))
        print(f"  -> brecha de FNR entre grupos: {d['brecha_fnr']:.4f}"
              f"   (peor: {d['peor_grupo']}, FNR {d['fnr_maximo']:.4f})\n")

    mejor = min(r["estrategias"].items(), key=lambda kv: kv[1]["brecha_fnr"])
    print(f">>> Menor brecha: {mejor[0]} ({mejor[1]['brecha_fnr']:.4f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"\nInforme en {args.out}")


if __name__ == "__main__":
    main()
