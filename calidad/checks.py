"""Medidas individuales de calidad de una foto tomada con móvil.

Cada función devuelve un número interpretable, no un booleano: la decisión de
aceptar o rechazar vive en `gate.py`, y los umbrales son configurables porque
**todavía no están calibrados sobre datos reales**. Separar medición de decisión
permite recalibrar sin reescribir nada.

Todas las medidas se calculan sobre la imagen reescalada a un lado corto
canónico. Sin eso, la misma foto daría puntuaciones distintas según el móvil que
la tomó, y el umbral sería inútil.
"""

from __future__ import annotations

import cv2
import numpy as np

LADO_CANONICO = 512


def normalizar_escala(img: np.ndarray, lado: int = LADO_CANONICO) -> np.ndarray:
    """Reescala manteniendo proporción para que las medidas sean comparables."""
    h, w = img.shape[:2]
    corto = min(h, w)
    if corto == lado:
        return img
    factor = lado / corto
    nuevo = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, nuevo, interpolation=interp)


def a_gris(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# Nitidez
# --------------------------------------------------------------------------- #
def _nitidez_de(g: np.ndarray) -> float:
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap.var()) / (float(g.var()) + 1e-6)


def nitidez(img: np.ndarray, margen: float = 0.2) -> dict:
    """Nitidez por varianza del laplaciano, normalizada por el contraste.

    La varianza del laplaciano a secas es una medida clásica pero engañosa: sube
    con la resolución y con el contraste, así que una foto oscura y nítida puede
    puntuar peor que una clara y borrosa. Dividir por la varianza de intensidad
    la vuelve adimensional y comparable entre fotos.

    Se mide **en la zona central**, no en toda la imagen, por un fallo detectado
    probando con fotos reales: cualquier elemento nítido irrelevante del borde
    —texto, marco, interfaz de una pantalla fotografiada— infla la puntuación
    global y deja pasar fotos cuyo sujeto está borroso. La nitidez global se
    conserva como referencia, pero la decisión se toma con la central.
    """
    g = a_gris(normalizar_escala(img)).astype(np.float64)
    h, w = g.shape
    mh, mw = int(h * margen), int(w * margen)
    centro = g[mh:h - mh, mw:w - mw]

    global_ = _nitidez_de(g)
    central = _nitidez_de(centro) if centro.size > 64 else global_
    return {
        "varianza_laplaciano": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2),
        "nitidez_global": round(global_, 5),
        "nitidez_normalizada": round(central, 5),
    }


# --------------------------------------------------------------------------- #
# Exposición
# --------------------------------------------------------------------------- #
def exposicion(img: np.ndarray) -> dict:
    """Brillo medio, recorte en negros y blancos, y rango dinámico usado.

    El recorte importa más que el brillo medio: un píxel a 0 o a 255 ha perdido
    la información, y ningún realce posterior la recupera.
    """
    g = a_gris(normalizar_escala(img))
    total = g.size
    quemados = float(np.count_nonzero(g >= 250) / total)
    apagados = float(np.count_nonzero(g <= 5) / total)
    p1, p99 = np.percentile(g, [1, 99])
    return {
        "brillo_medio": round(float(g.mean()) / 255.0, 4),
        "fraccion_quemada": round(quemados, 4),
        "fraccion_apagada": round(apagados, 4),
        "rango_dinamico": round(float(p99 - p1) / 255.0, 4),
    }


# --------------------------------------------------------------------------- #
# Reflejos especulares (flash)
# --------------------------------------------------------------------------- #
def reflejos(img: np.ndarray) -> dict:
    """Mancha de reflejo del flash: píxeles saturados agrupados.

    Se distingue de una sobreexposición general porque el reflejo forma una
    mancha compacta. Si cae sobre la lesión, la tapa por completo.
    """
    g = a_gris(normalizar_escala(img))
    mascara = (g >= 250).astype(np.uint8)
    if not mascara.any():
        return {"fraccion_reflejo_mayor": 0.0, "n_manchas": 0}

    n, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    # La etiqueta 0 es el fondo
    areas = estadisticas[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    mayor = float(areas.max()) / g.size if areas.size else 0.0
    return {
        "fraccion_reflejo_mayor": round(mayor, 4),
        "n_manchas": int((areas > g.size * 0.0005).sum()),
    }


# --------------------------------------------------------------------------- #
# Encuadre
# --------------------------------------------------------------------------- #
def encuadre(img: np.ndarray, margen: float = 0.2) -> dict:
    """¿El sujeto está en el centro, y hay algo que mirar?

    Deliberadamente **no** se detecta piel por color. Los detectores clásicos
    por umbrales en HSV o YCrCb están construidos alrededor de tonos claros y
    fallan de forma sistemática en piel oscura: usarlos como filtro de entrada
    rechazaría más fotos justo del grupo al que este proyecto quiere servir, y
    reintroduciría por la puerta de atrás el sesgo que trata de medir.

    En su lugar se usan dos señales sin color: densidad de bordes del centro
    frente al marco, y si la imagen tiene estructura alguna.
    """
    g = a_gris(normalizar_escala(img)).astype(np.float64)
    h, w = g.shape

    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.hypot(gx, gy)

    mh, mw = int(h * margen), int(w * margen)
    centro = magnitud[mh:h - mh, mw:w - mw]
    mascara = np.ones_like(magnitud, dtype=bool)
    mascara[mh:h - mh, mw:w - mw] = False
    marco = magnitud[mascara]

    media_centro = float(centro.mean()) if centro.size else 0.0
    media_marco = float(marco.mean()) if marco.size else 0.0

    return {
        "estructura": round(float(g.std()) / 255.0, 4),
        "bordes_centro": round(media_centro, 3),
        "ratio_centro_marco": round(media_centro / (media_marco + 1e-6), 3),
    }


# --------------------------------------------------------------------------- #
# Color
# --------------------------------------------------------------------------- #
def dominante_color(img: np.ndarray) -> dict:
    """Desviación del iluminante estimada por mundo-gris.

    Cada móvil interpreta el balance de blancos a su manera, y cualquier medida
    que dependa del color —tono de piel, ictericia, palidez— queda inservible si
    hay un tinte fuerte. Esto no lo corrige, lo declara.
    """
    if img.ndim == 2:
        return {"tinte": 0.0, "medias_bgr": [0.0, 0.0, 0.0]}

    peq = normalizar_escala(img).astype(np.float64)
    medias = [float(peq[:, :, c].mean()) for c in range(3)]
    m = float(np.mean(medias)) + 1e-6
    # Desviación máxima relativa entre canales: 0 = neutro
    tinte = float(max(abs(v - m) for v in medias) / m)
    return {"tinte": round(tinte, 4), "medias_bgr": [round(v, 2) for v in medias]}


def corregir_balance_gris(img: np.ndarray) -> np.ndarray:
    """Normaliza el balance de blancos por mundo-gris.

    Suposición fuerte: que el promedio de la escena es neutro. En un primer
    plano de piel eso es falso, así que **solo debe usarse cuando la foto
    incluye una referencia neutra**; para eso está `corregir_con_referencia`.
    """
    if img.ndim == 2:
        return img.copy()
    salida = img.astype(np.float64)
    medias = [salida[:, :, c].mean() + 1e-6 for c in range(3)]
    objetivo = float(np.mean(medias))
    for c in range(3):
        salida[:, :, c] *= objetivo / medias[c]
    return np.clip(salida, 0, 255).astype(np.uint8)


def corregir_con_referencia(img: np.ndarray, caja: tuple[int, int, int, int]) -> np.ndarray:
    """Normaliza usando un parche que el usuario sabe que es neutro.

    `caja` es (x, y, ancho, alto) sobre una zona blanca o gris incluida en la
    toma —una tarjeta de color o una hoja de papel—. Es la forma fiable de
    comparar color entre fotos de móviles distintos.
    """
    if img.ndim == 2:
        return img.copy()
    x, y, aw, ah = caja
    parche = img[y:y + ah, x:x + aw].astype(np.float64)
    if parche.size == 0:
        raise ValueError("La caja de referencia queda fuera de la imagen.")
    medias = [parche[:, :, c].mean() + 1e-6 for c in range(3)]
    objetivo = float(np.mean(medias))
    salida = img.astype(np.float64)
    for c in range(3):
        salida[:, :, c] *= objetivo / medias[c]
    return np.clip(salida, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Resolución
# --------------------------------------------------------------------------- #
def resolucion(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    return {"ancho": int(w), "alto": int(h), "megapixeles": round(w * h / 1e6, 2),
            "lado_corto": int(min(h, w))}


def medir_todo(img: np.ndarray) -> dict:
    """Todas las medidas de una imagen, en un único diccionario plano."""
    return {
        **resolucion(img),
        **nitidez(img),
        **exposicion(img),
        **reflejos(img),
        **encuadre(img),
        **dominante_color(img),
    }
