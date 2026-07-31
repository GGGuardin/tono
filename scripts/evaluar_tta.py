"""Evaluación promediando volteos (test-time augmentation).

Una lesión cutánea no tiene orientación privilegiada: da igual verla reflejada.
Así que promediar la predicción sobre la imagen original y sus tres reflejos
—horizontal, vertical y ambos— **reduce la varianza sin introducir sesgo**.

Es deliberadamente distinto del multi-recorte, que se probó y salió mal:

| | Multi-recorte | Volteos |
|---|---|---|
| Cambia la escala | sí | no |
| Cambia el encuadre | sí | no |
| Agregación | máximo | media |
| Efecto medido | sube todas las puntuaciones, baja el AUROC | pendiente de medir |

El máximo sobre recortes distintos acaba fijado por el trozo más ruidoso, y por
eso inflaba las puntuaciones. La media sobre transformaciones que preservan el
contenido no tiene ese problema: es el mismo objeto visto de cuatro maneras.

Que ayude o no sigue siendo una pregunta empírica, y por eso esto produce un
predictions.csv comparable en lugar de aplicarse sin medir.
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


def variantes(x: torch.Tensor) -> torch.Tensor:
    """Apila la imagen y sus tres reflejos en un solo lote."""
    return torch.cat([x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)], dim=0)


@torch.no_grad()
def predecir(modelo, imagenes: list[np.ndarray], tf, device, lote: int = 8) -> np.ndarray:
    """Media de las cuatro variantes por imagen."""
    salida = []
    for i in range(0, len(imagenes), lote):
        tensores = torch.stack([tf(image=im)["image"] for im in imagenes[i:i + lote]])
        x = variantes(tensores).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(modelo(x).float()).cpu().numpy().ravel()
        # p viene ordenado por variante: [orig..., h..., v..., hv...]
        salida.extend(p.reshape(4, -1).mean(axis=0).tolist())
    return np.asarray(salida)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluacion promediando volteos")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lote", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    modelo, ckpt = load_checkpoint(args.checkpoint, map_location=device)
    modelo = modelo.to(device).eval()
    tf = build_transforms(ckpt["config"].get("img_size", 224), train=False)

    df = pd.read_csv(args.manifest)
    if args.split != "all":
        df = df[df["split"] == args.split]
    df = df.reset_index(drop=True)
    print(f"{len(df)} imagenes | 4 variantes cada una", flush=True)

    probs = []
    for i in range(0, len(df), args.lote):
        trozo = df.iloc[i:i + args.lote]
        imgs = [load_image(r) for r in trozo["image_path"]]
        probs.extend(predecir(modelo, imgs, tf, device, args.lote).tolist())
        if (i + args.lote) % 200 < args.lote:
            print(f"  {min(i + args.lote, len(df))}/{len(df)}", flush=True)

    salida = df.copy()
    salida["prob"] = probs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.out, index=False)
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
