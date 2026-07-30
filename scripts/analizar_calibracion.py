"""Analiza la calibración del portero contra el juicio de dermatólogos.

Tres preguntas, en orden de importancia:

1. **¿Qué medidas predicen de verdad la evaluabilidad?** Programé seis controles
   por criterio propio. Alguno puede no aportar nada, y en ese caso sobra.
2. **¿Dónde van los umbrales?** Se eligen con datos en lugar de con intuición,
   maximizando el índice de Youden contra la etiqueta del dermatólogo.
3. **¿El portero es equitativo?** Si rechaza más fotos de piel oscura, excluye a
   un grupo antes de que el modelo llegue a opinar: un daño que no aparecería en
   ninguna métrica del clasificador.

    python scripts/analizar_calibracion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from calidad import Umbrales  # noqa: E402

# Medidas candidatas y el sentido en que se espera que empeoren la evaluabilidad
# ("bajo" = valores bajos son peores; "alto" = valores altos son peores)
MEDIDAS = {
    "nitidez_normalizada": "bajo",
    "nitidez_global": "bajo",
    "rango_dinamico": "bajo",
    "estructura": "bajo",
    "lado_corto": "bajo",
    "brillo_medio": "ambos",
    "fraccion_quemada": "alto",
    "fraccion_apagada": "alto",
    "fraccion_reflejo_mayor": "alto",
    "ratio_centro_marco": "bajo",
    "tinte": "alto",
}


def auc_dirigida(y: np.ndarray, x: np.ndarray, sentido: str) -> float:
    """AUC de la medida como predictor de 'no evaluable'.

    Se orienta el signo para que 0,5 sea siempre azar y >0,5 signifique que la
    medida discrimina en el sentido esperado.
    """
    if sentido == "alto":
        puntuacion = x
    elif sentido == "bajo":
        puntuacion = -x
    else:  # 'ambos': se mide la desviación respecto a la mediana de los evaluables
        centro = float(np.median(x[y == 0]))
        puntuacion = np.abs(x - centro)
    ok = np.isfinite(puntuacion)
    if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], puntuacion[ok]))


def main() -> None:
    df = pd.read_csv(RAIZ / "results" / "calibracion_scin.csv")
    # y = 1 significa "el dermatólogo dijo que NO era evaluable"
    y = (1 - df["evaluable_derm"]).values
    print(f"{len(df)} imágenes | no evaluables segun dermatologo: {int(y.sum())} "
          f"({y.mean():.1%})\n")

    informe: dict = {"n": int(len(df)), "prop_no_evaluables": round(float(y.mean()), 4)}

    # ------------------------------------------------------------------ 1 --
    print("=" * 74)
    print("1. PODER PREDICTIVO DE CADA MEDIDA (AUC contra 'no evaluable')")
    print("=" * 74)
    print(f"{'medida':26s} {'AUC':>7s}  {'sentido':8s}  lectura")
    filas = []
    for medida, sentido in MEDIDAS.items():
        if medida not in df.columns:
            continue
        auc = auc_dirigida(y, df[medida].values.astype(float), sentido)
        if np.isnan(auc):
            continue
        lectura = ("predice bien" if auc >= 0.65 else
                   "predice algo" if auc >= 0.57 else
                   "casi nada" if auc >= 0.52 else "no aporta")
        filas.append({"medida": medida, "auc": round(auc, 4), "sentido": sentido,
                      "lectura": lectura})
        print(f"{medida:26s} {auc:7.4f}  {sentido:8s}  {lectura}")
    informe["poder_predictivo"] = sorted(filas, key=lambda r: -r["auc"])

    # ------------------------------------------------------------------ 2 --
    print("\n" + "=" * 74)
    print("2. UMBRALES SUGERIDOS POR LOS DATOS (indice de Youden)")
    print("=" * 74)
    sugeridos = {}
    for fila in informe["poder_predictivo"]:
        if fila["auc"] < 0.57 or fila["sentido"] == "ambos":
            continue
        medida = fila["medida"]
        x = df[medida].values.astype(float)
        ok = np.isfinite(x)
        signo = 1.0 if fila["sentido"] == "alto" else -1.0
        fpr, tpr, cortes = roc_curve(y[ok], signo * x[ok])
        corte = float(cortes[int(np.argmax(tpr - fpr))]) * signo
        sugeridos[medida] = round(corte, 5)
        print(f"{medida:26s} corte sugerido {corte:9.5f}   "
              f"(sens {tpr[int(np.argmax(tpr - fpr))]:.3f}, "
              f"esp {1 - fpr[int(np.argmax(tpr - fpr))]:.3f})")
    informe["umbrales_sugeridos"] = sugeridos

    u = Umbrales()
    print("\nComparado con lo que tenia puesto a ojo:")
    equivalencias = {"nitidez_normalizada": "nitidez_min", "rango_dinamico": "rango_dinamico_min",
                     "estructura": "estructura_min", "lado_corto": "lado_corto_min",
                     "fraccion_quemada": "quemada_max", "fraccion_apagada": "apagada_max",
                     "fraccion_reflejo_mayor": "reflejo_max", "tinte": "tinte_max",
                     "ratio_centro_marco": "ratio_centro_marco_min"}
    comparacion = {}
    for medida, atributo in equivalencias.items():
        if medida in sugeridos:
            actual = getattr(u, atributo)
            comparacion[atributo] = {"actual": actual, "sugerido": sugeridos[medida]}
            print(f"  {atributo:24s} actual {actual!s:>9s}  ->  sugerido {sugeridos[medida]}")
    informe["comparacion_umbrales"] = comparacion

    # ------------------------------------------------------------------ 3 --
    print("\n" + "=" * 74)
    print("3. ACUERDO DEL PORTERO CON EL DERMATOLOGO (umbrales actuales)")
    print("=" * 74)
    rechaza = (df["veredicto"] == "rechazada").values
    vp = int(((rechaza == 1) & (y == 1)).sum()); fp = int(((rechaza == 1) & (y == 0)).sum())
    vn = int(((rechaza == 0) & (y == 0)).sum()); fn = int(((rechaza == 0) & (y == 1)).sum())
    sens = vp / (vp + fn) if vp + fn else float("nan")
    esp = vn / (vn + fp) if vn + fp else float("nan")
    print(f"  tasa de rechazo del portero : {rechaza.mean():.1%}")
    print(f"  sensibilidad (pilla las malas): {sens:.3f}")
    print(f"  especificidad (no tira buenas): {esp:.3f}")
    print(f"  matriz: VP={vp} FP={fp} VN={vn} FN={fn}")
    informe["acuerdo_actual"] = {"tasa_rechazo": round(float(rechaza.mean()), 4),
                                 "sensibilidad": round(sens, 4), "especificidad": round(esp, 4),
                                 "vp": vp, "fp": fp, "vn": vn, "fn": fn}

    # ------------------------------------------------------------------ 4 --
    print("\n" + "=" * 74)
    print("4. EQUIDAD DEL PORTERO POR TONO DE PIEL  <-- la pregunta que importa")
    print("=" * 74)
    df["_rechaza"] = rechaza
    tonos = df[df["fitzpatrick_derm"].notna()].copy()
    tonos["fitzpatrick_derm"] = tonos["fitzpatrick_derm"].astype(float).astype(int).astype(str)
    tabla = tonos.groupby("fitzpatrick_derm").agg(
        n=("_rechaza", "size"),
        no_evaluables_derm=("evaluable_derm", lambda s: float(1 - s.mean())),
        tasa_rechazo_portero=("_rechaza", "mean"),
    ).round(4)
    # La pregunta correcta NO es "¿rechaza más en piel oscura?" a secas: si esas
    # fotos son de verdad peores, rechazarlas está bien. La pregunta es si
    # **descarta fotos buenas** de un tono más que de otro. Así que se condiciona
    # sobre las que el dermatólogo consideró evaluables.
    buenas = tonos[tonos["evaluable_derm"] == 1]
    tabla["rechazo_de_las_buenas"] = (buenas.groupby("fitzpatrick_derm")["_rechaza"]
                                      .mean().round(4))
    tabla["n_buenas"] = buenas.groupby("fitzpatrick_derm").size()
    print(tabla.to_string())

    grandes = tabla[tabla["n_buenas"].fillna(0) >= 30]
    if len(grandes) >= 2:
        brecha = float(grandes["rechazo_de_las_buenas"].max()
                       - grandes["rechazo_de_las_buenas"].min())
        print(f"\n  Rechazo de fotos BUENAS por tono (n>=30): brecha {brecha:.4f}")
        print(f"  Mas descartado: tono {grandes['rechazo_de_las_buenas'].idxmax()}  |  "
              f"menos: tono {grandes['rechazo_de_las_buenas'].idxmin()}")
        veredicto = ("Sin evidencia de que el portero descarte fotos buenas de piel "
                     "oscura mas que de piel clara." if brecha < 0.12 else
                     "AVISO: el portero descarta fotos buenas de forma desigual entre tonos.")
        print(f"  >>> {veredicto}")
        informe["equidad_portero"] = {
            "tabla": tabla.reset_index().to_dict("records"),
            "brecha_rechazo_de_buenas": round(brecha, 4),
            "tonos_evaluados": list(grandes.index),
        }
        informe["veredicto_equidad"] = veredicto

    # Hallazgo independiente del portero, y probablemente el más importante:
    # cómo varía el propio patrón de referencia con el tono de piel.
    print("\n  Tasa de 'no evaluable' SEGUN EL DERMATOLOGO, por tono:")
    for tono, fila in tabla.iterrows():
        print(f"    tono {tono}: {fila['no_evaluables_derm']:.1%}  (n={int(fila['n'])})")
    informe["no_evaluable_derm_por_tono"] = {
        str(t): round(float(f["no_evaluables_derm"]), 4) for t, f in tabla.iterrows()}

    salida = RAIZ / "results" / "calibracion_informe.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    print(f"\nInforme en {salida}")


if __name__ == "__main__":
    main()
