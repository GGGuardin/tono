# Portero de calidad para fotos de móvil

> ## ⚠️ NO ES UNA HERRAMIENTA DIAGNÓSTICA
> Componente de un proyecto **educativo y experimental**. No es un dispositivo
> médico y no interviene en ninguna decisión clínica.

Proyecto **Tono**. La Fase 1 (este portero) está terminada; los resultados de la
Fase 2 están en [`RESULTADOS.md`](RESULTADOS.md).

Fase 1: Decide si una foto tomada con la cámara de un móvil
sirve para analizarse y, cuando no sirve, **dice qué hacer para arreglarla**.

Es independiente del modelo y del dominio: vale igual para piel, ojo externo o
boca. Se puede usar suelto.

## Por qué existe

Un clasificador alimentado con una foto mala **no falla: responde con la misma
confianza aparente que con una buena**. Es la forma más eficiente de engañar a
alguien. La mayoría de las herramientas de imagen médica por móvil aceptan
cualquier entrada, y ahí se pierde la fiabilidad antes de que el modelo intervenga.

## Qué mide

| Control | Medida | Por qué así |
|---|---|---|
| **Nitidez** | Varianza del laplaciano normalizada por el contraste, **en la zona central** | La varianza a secas sube con la resolución y el contraste. Y medida globalmente, cualquier borde nítido irrelevante la infla — ver la nota de abajo. |
| **Exposición** | Brillo medio, fracción de píxeles quemados y apagados, rango dinámico | El recorte importa más que el brillo: un píxel a 0 o a 255 perdió la información y no se recupera. |
| **Reflejos** | Área de la mancha saturada más grande, por componentes conexas | Distingue el reflejo compacto del flash de una sobreexposición general. Si cae sobre la lesión, la tapa. |
| **Encuadre** | Densidad de bordes del centro frente al marco, y estructura global | Sin usar color — ver la decisión de diseño de abajo. |
| **Color** | Desviación del iluminante por mundo-gris | Cualquier medida que dependa del color queda inservible con un tinte fuerte. Esto no lo corrige: lo declara. |
| **Resolución** | Lado corto en píxeles | Detecta capturas de pantalla y reenvíos por mensajería, que llegan diezmados. |

Y para varias fotos de la misma zona:

| Control | Qué responde |
|---|---|
| **Concordancia de predicciones** | ¿Las tres fotos dicen lo mismo? Si no, **no se da resultado.** |
| **Mismo sujeto** | ¿Son de la misma zona, o alguien fotografió cosas distintas? |

## Dos decisiones de diseño que conviene explicar

### No se detecta piel por color, a propósito

Lo natural sería filtrar por color de piel con umbrales en HSV o YCrCb. **Se
descartó deliberadamente.** Esos detectores clásicos están construidos alrededor
de tonos claros y fallan de forma sistemática en piel oscura. Usarlos como filtro
de entrada rechazaría más fotos justo del grupo al que este proyecto quiere
servir, y reintroduciría por la puerta de atrás el sesgo que trata de medir.

En su lugar, el encuadre se juzga con señales sin color: densidad de bordes del
centro frente al marco, y si la imagen tiene estructura alguna.

### La nitidez se mide en el centro, y eso salió de una prueba real

La primera versión medía la nitidez sobre la imagen completa. Al probarla con una
foto real de una pantalla de visor médico, el resultado fue revelador:

| Foto | Nitidez global | Nitidez central |
|---|---|---|
| Pantalla completa, con la interfaz del visor | **0,1411** | 0,0178 |
| Recortada a la imagen | 0,0073 | 0,0060 |

La foto completa parecía **ocho veces más nítida** de lo que era. Toda esa nitidez
venía del texto y el marco de la interfaz, no del sujeto. El control habría dado
por buena una foto cuyo contenido relevante estaba blando.

Es exactamente un atajo espurio, pero dentro de la métrica de calidad en lugar
del modelo. Medir en la zona central lo corrige.

## Uso

```python
import cv2
from calidad import evaluar

informe = evaluar(cv2.imread("foto.jpg"))
print(informe.veredicto)   # aceptada / dudosa / rechazada
print(informe.mensaje())   # qué hacer si no vale
print(informe.medidas)     # todas las medidas, para calibrar
```

Varias fotos:

```python
from calidad import concordancia_predicciones, mismo_sujeto

mismo_sujeto([img1, img2, img3])                       # ¿son de la misma zona?
concordancia_predicciones([0.71, 0.74, 0.69])          # ¿coinciden las predicciones?
```

Corrección de color fiable, con una referencia neutra en la toma:

```python
from calidad import corregir_con_referencia
corregida = corregir_con_referencia(img, caja=(10, 10, 80, 80))
```

## Calibrar los umbrales

**Los umbrales de `Umbrales` son provisionales.** Están puestos con criterio pero
sin ajustar contra fotos reales anotadas; presentarlos como definitivos sería
inventarse un número. La herramienta para arreglarlo:

```bash
python -m calidad.cli --carpeta fotos/ --csv medidas.csv
python -m calidad.cli --carpeta fotos/ --umbral nitidez_min=0.02
```

Mides una tanda, miras la distribución de cada medida y colocas los cortes donde
de verdad separan lo usable de lo inservible. La CLI reporta además la **tasa de
rechazo**, que es el indicador que importa: si rechaza casi todo es inusable, y si
no rechaza nada es un adorno.

## Pruebas

```bash
python tests/test_calidad.py
```

18 comprobaciones sobre imágenes sintéticas que rompen **cada control por
separado**, verificando que salta el que debe y no otro: desenfoque, falta de luz,
sobreexposición, reflejo de flash, imagen plana, resolución insuficiente, tinte
extremo, predicciones dispares, fotos de zonas distintas y corrección de color con
referencia.

## Instalación

```bash
pip install -r requirements.txt
```

Solo `numpy`, `opencv-python-headless` y `pillow`. Sin dependencias de aprendizaje
automático: el portero es anterior al modelo.

## Licencia

MIT. NOT FOR MEDICAL USE.
