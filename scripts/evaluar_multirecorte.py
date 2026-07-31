"""Inferencia por multi-recorte: analiza la imagen por trozos en vez de solo el centro.

El problema que resuelve, detectado con dos fotos reales que el modelo no
detectaba: **el preprocesado destruye las lesiones pequeñas**. Un recorte central
reescalado a 224 px convierte una lesión que ocupa el 3% del encuadre en unos
pocos píxeles, y además descarta cualquier cosa que esté fuera del centro.

    lesión pequeña, recorte central   ->  p = 0,054   (no detecta)
    la misma, con multi-recorte       ->  p = 0,277   (detecta)

La imagen se recorre con ventanas a varias escalas y se toma el **máximo**: si
algún trozo contiene algo sospechoso, la imagen entera lo es. Es la lógica
correcta para cribado, donde el coste de no ver algo es asimétrico.

**Sube todas las puntuaciones**, también las de lesiones benignas, así que el
umbral hay que recalibrarlo en validación. Que el AUROC mejore o no es una
pregunta empírica, no una obviedad — de ahí que este script produzca un
predictions.csv en el formato estándar para poder compararlo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from scripts._rutas import anadir_al_path  # noqa: E402
anadir_al_path()

from src.data import build_transforms, load_image  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import get_device  # noqa: E402


def recortes(img: np.ndarray, escalas=(1.0, 0.6, 0.4), paso: float = 0.5,
             minimo: int = 64) -> list[np.ndarray]:
    """Ventanas cuadradas a varias escalas, con solape."""
    h, w = img.shape[:2]
    corto = min(h, w)
    salida = []
    for e in escalas:
        lado = int(corto * e)
        if lado < minimo:
            continue
        s = max(1, int(lado * paso))
        for y in range(0, max(1, h - lado + 1), s):
            for x in range(0, max(1, w - lado + 1), s):
                salida.append(img[y:y + lado, x:x + lado])
    if not salida:
        salida = [img]
    return salida


@torch.no_grad()
def probabilidad(modelo, img: np.ndarray, tf, device, escalas, paso,
                 lote: int = 32) -> tuple[float, float, int]:
    """Devuelve (máximo, media de los 3 mayores, número de recortes)."""
    trozos = recortes(img, escalas, paso)
    probs = []
    for i in range(0, len(trozos), lote):
        tensores = [tf(image=t)["image"] for t in trozos[i:i + lote]]
        x = torch.stack(tensores).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(modelo(x).float()).cpu().numpy().ravel()
        probs.extend(p.tolist())
    probs = sorted(probs)
    return float(probs[-1]), float(np.mean(probs[-3:])), len(trozos)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluacion con multi-recorte")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="predictions.csv de salida")
    ap.add_argument("--escalas", default="1.0,0.6,0.4")
    ap.add_argument("--paso", type=float, default=0.5)
    ap.add_argument("--agregacion", default="max", choices=["max", "top3"])
    args = ap.parse_args()

    device = get_device()
    modelo, ckpt = load_checkpoint(args.checkpoint, map_location=device)
    modelo = modelo.to(device).eval()
    tf = build_transforms(ckpt["config"].get("img_size", 224), train=False)
    escalas = tuple(float(e) for e in args.escalas.split(","))

    df = pd.read_csv(args.manifest)
    if args.split != "all":
        df = df[df["split"] == args.split]
    df = df.reset_index(drop=True)
    print(f"{len(df)} imagenes | escalas {escalas} | paso {args.paso}", flush=True)

    maximos, top3, n_recortes = [], [], []
    for i, fila in df.iterrows():
        mx, t3, n = probabilidad(modelo, load_image(fila["image_path"]), tf, device,
                                 escalas, args.paso)
        maximos.append(mx); top3.append(t3); n_recortes.append(n)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    salida = df.copy()
    salida["prob"] = maximos if args.agregacion == "max" else top3
    salida["prob_max"] = maximos
    salida["prob_top3"] = top3
    salida["n_recortes"] = n_recortes
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.out, index=False)
    print(f"\nGuardado en {args.out} (media de {float(np.mean(n_recortes)):.0f} recortes por imagen)")


if __name__ == "__main__":
    main()
