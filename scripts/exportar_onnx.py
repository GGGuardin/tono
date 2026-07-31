"""Exporta el modelo a ONNX cuantizado, con mapa de activación incluido.

Objetivo: que la inferencia ocurra **dentro del navegador del móvil**, de modo
que la foto no salga del dispositivo. No es una promesa de privacidad que haya
que creerse: no hay servidor al que enviarla.

## Cómo se obtiene el mapa de calor sin gradientes

Grad-CAM necesita retropropagación, y ONNX Runtime Web no la hace. Pero
DenseNet-121 termina en *global average pooling* seguido de una capa lineal, y en
esa topología el **CAM clásico** (Zhou et al., 2016) es exactamente equivalente y
se calcula hacia delante:

    CAM = Σ_k  w_k · A_k

donde `A_k` son los mapas de características y `w_k` los pesos del clasificador.
Eso es literalmente una convolución 1×1 con los pesos del clasificador, así que se
mete en el propio grafo ONNX como una segunda salida. El navegador no calcula
nada: recibe el mapa ya hecho.

    python scripts/exportar_onnx.py --checkpoint best.pth --out web/modelo.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ.parent))

from src.model import load_checkpoint  # noqa: E402


class ModeloConCAM(nn.Module):
    """DenseNet-121 que devuelve el logit y el mapa de activación de clase."""

    def __init__(self, densenet: nn.Module):
        super().__init__()
        self.features = densenet.features
        clasificador = densenet.classifier
        # El clasificador puede ser Linear o Sequential(Dropout, Linear)
        lineal = clasificador if isinstance(clasificador, nn.Linear) else clasificador[-1]
        if not isinstance(lineal, nn.Linear):
            raise TypeError(f"No encuentro la capa lineal final: {type(lineal)}")
        self.classifier = lineal

        # Convolución 1x1 con los pesos del clasificador: produce el CAM.
        # El sesgo se omite a propósito — es una constante que solo desplaza el
        # mapa y desaparece al normalizarlo.
        self.cam = nn.Conv2d(lineal.in_features, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.cam.weight.copy_(lineal.weight.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = F.relu(self.features(x))
        logit = self.classifier(torch.flatten(F.adaptive_avg_pool2d(f, 1), 1))
        return torch.sigmoid(logit), self.cam(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch -> ONNX cuantizado, con CAM")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sin-cuantizar", action="store_true")
    ap.add_argument("--muestras", default=None,
                    help="Carpeta con imagenes REALES para verificar la exportacion")
    ap.add_argument("--error-max", type=float, default=0.02,
                    help="Diferencia maxima tolerada frente a PyTorch")
    args = ap.parse_args()

    modelo, ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    envoltorio = ModeloConCAM(modelo).eval()

    ejemplo = torch.randn(1, 3, ckpt["config"].get("img_size", 224),
                          ckpt["config"].get("img_size", 224))
    with torch.no_grad():
        prob_ref, cam_ref = envoltorio(ejemplo)
    print(f"salida de prueba: prob {prob_ref.shape}, cam {cam_ref.shape}")

    salida = Path(args.out)
    salida.parent.mkdir(parents=True, exist_ok=True)
    sin_cuantizar = salida.with_name(salida.stem + "_fp32.onnx")

    torch.onnx.export(
        envoltorio, ejemplo, str(sin_cuantizar),
        input_names=["imagen"], output_names=["probabilidad", "cam"],
        opset_version=13, dynamo=False,
    )
    mb = sin_cuantizar.stat().st_size / 1e6
    print(f"ONNX sin cuantizar: {mb:.1f} MB")

    if args.sin_cuantizar:
        # `replace` y no `rename`: en Windows renombrar falla si el destino existe,
        # que es justo el caso al reexportar sobre un modelo ya desplegado.
        sin_cuantizar.replace(salida)
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Cuantización dinámica a int8: reduce ~4x el peso sin datos de
        # calibración. La pérdida se verifica abajo contra PyTorch.
        quantize_dynamic(str(sin_cuantizar), str(salida), weight_type=QuantType.QUInt8)
        print(f"ONNX cuantizado:   {salida.stat().st_size / 1e6:.1f} MB "
              f"({mb / (salida.stat().st_size / 1e6):.1f}x mas pequeno)")

    # --- Verificación contra PyTorch ---
    #
    # ATENCIÓN: la primera versión de esto verificaba con `torch.randn`, es decir
    # ruido gaussiano. Daba una diferencia de 0,003 y parecía perfecto. Sobre
    # imágenes REALES el error de la cuantización int8 resultó ser de 0,29 y 0,48
    # —cien veces mayor— porque el ruido no activa la red como lo hace una foto.
    # Verificar con entradas que no se parecen a las reales no verifica nada.
    import numpy as np
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(salida), providers=["CPUExecutionProvider"])
    lado = ckpt["config"].get("img_size", 224)

    entradas = []
    if args.muestras:
        import cv2

        from src.data import build_transforms

        tf = build_transforms(lado, train=False)
        for ruta in sorted(Path(args.muestras).glob("*"))[:12]:
            if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            L = min(h, w)
            rec = img[(h - L) // 2:(h - L) // 2 + L, (w - L) // 2:(w - L) // 2 + L]
            entradas.append(tf(image=rec)["image"].unsqueeze(0))
    if not entradas:
        print("AVISO: sin imagenes de muestra, la verificacion usa ruido y NO es fiable.")
        entradas = [torch.randn(1, 3, lado, lado) for _ in range(8)]

    diferencias = []
    for x in entradas:
        with torch.no_grad():
            p_torch = float(envoltorio(x)[0].item())
        p_onnx = float(sesion.run(None, {"imagen": x.numpy()})[0].ravel()[0])
        diferencias.append(abs(p_torch - p_onnx))
    peor = max(diferencias)
    print(f"verificado sobre {len(entradas)} entradas | diferencia maxima: {peor:.5f}")
    if peor > args.error_max:
        raise SystemExit(
            "EXPORTACION RECHAZADA: la diferencia maxima "
            f"({peor:.4f}) supera el limite ({args.error_max}). "
            "Reexporta con --sin-cuantizar."
        )

    meta = {
        "arch": ckpt["config"].get("arch"),
        "img_size": ckpt["config"].get("img_size", 224),
        "umbral": round(float(ckpt.get("threshold", 0.5)), 4),
        "prevalencia_entrenamiento": round(float(ckpt.get("train_prevalence", 0.5)), 4),
        "auroc_val": round(float(ckpt.get("val_auroc", 0)), 4),
        "media": [0.485, 0.456, 0.406],
        "desv": [0.229, 0.224, 0.225],
        "diferencia_max_vs_pytorch": round(max(diferencias), 5),
    }
    with open(salida.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    sin_cuantizar.unlink(missing_ok=True)
    print(f"\nModelo en {salida}\nMetadatos en {salida.with_suffix('.json')}")


if __name__ == "__main__":
    main()
