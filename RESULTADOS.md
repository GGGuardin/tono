# Fase 2 — resultados sobre PAD-UFES-20

> ⚠️ **NO ES UNA HERRAMIENTA DIAGNÓSTICA.** Proyecto educativo y experimental.

DenseNet-121 ajustada sobre ~2.300 fotos de móvil de PAD-UFES-20 (Brasil).
Tarea binaria: **maligno** (BCC, SCC, MEL) frente a **benigno** (ACK, NEV, SEK).
10 minutos en una T4. Artefactos en [`results/pad_ufes/`](results/pad_ufes).

---

## El resultado que importa: este dataset no puede responder la pregunta del proyecto

El proyecto existe para medir si el modelo funciona igual de bien en piel oscura.
**Con PAD-UFES-20 eso no se puede medir.** Positivos en el conjunto de test, por
tipo de Fitzpatrick:

| Fitzpatrick | Positivos en test | ¿Evaluable? |
|---|---|---|
| 1 (más clara) | 23 | apenas |
| 2 | 130 | sí |
| 3 | 58 | sí |
| 4 | **4** | no |
| 5 | **2** | no |
| 6 (más oscura) | **0** | no |

Los tres tonos oscuros suman **6 casos positivos**. La auditoría de equidad que
justifica el proyecto entero **es imposible con estos datos**, y presentarla como
hecha sería el tipo de afirmación que este repositorio existe para no hacer.

El resumen automático generó este titular:

> «Brecha de FNR entre tonos de piel: 0,1174 (peor tono: 1)»

**Ese titular no se sostiene y hay que descartarlo.** Está calculado sobre los
tonos 1, 2 y 3 —los tres claros—, y el peor de ellos, el tono 1, tiene 23
positivos: la diferencia entre 5 y 4 falsos negativos mueve la brecha entera. Su
AUROC en ese subgrupo es **0,5435**, indistinguible del azar, sobre n=29.

Lo que sí es un resultado legítimo: **PAD-UFES-20, pese a traer etiqueta de
Fitzpatrick, procede de una población demasiado sesgada hacia piel clara como para
auditar equidad.** Para eso hace falta **DDI** —656 imágenes con biopsia y tonos
deliberadamente diversos, construido por Stanford precisamente por este motivo— o
**SCIN**. Este hallazgo explica por qué DDI tuvo que existir.

---

## Rendimiento global

| Métrica | Valor |
|---|---|
| **AUROC** | **0,895** (IC95% 0,866–0,922) |
| AUPRC | 0,865 (prevalencia 0,476) |
| Sensibilidad / Especificidad | 0,880 / 0,762 |
| FNR (cáncer que se deja pasar) | 0,120 |
| n | 456 imágenes, split por paciente |

Buen número. Pero la siguiente sección lo pone en su sitio.

## El modelo parece mucho mejor contra impresión clínica que contra biopsia

PAD-UFES-20 marca qué casos tienen confirmación histológica. Separando:

| Subconjunto | n | Prevalencia | AUROC | Especificidad |
|---|---|---|---|---|
| Todos | 456 | 0,476 | **0,895** (0,866–0,922) | 0,762 |
| **Solo con biopsia** | 270 | 0,804 | **0,724** (0,640–0,796) | **0,396** |

**El AUROC cae 0,17 al exigir verdad histológica**, y la especificidad se hunde de
0,76 a 0,40. Los intervalos de confianza no se solapan.

La explicación es que los dos subconjuntos no son comparables en dificultad: el
clínico biopsia lo que le resulta sospechoso, así que el subconjunto biopsiado
tiene un 80% de malignidad y concentra los casos difíciles. Los no biopsiados
llevan una etiqueta que **es también un juicio visual**, de modo que el modelo
puede acertarla imitando la mirada del clínico en lugar de la biología.

Es el mismo patrón que apareció en el proyecto de tórax con las etiquetas de NIH
extraídas por NLP: **cuando la etiqueta es barata, el modelo luce; cuando la
etiqueta es dura, el modelo se desnuda.** Dos dominios distintos, misma lección.

## Sobreajuste, y no disimulado

| Época | Train AUROC | Val AUROC | Val loss |
|---|---|---|---|
| 3 (mejor) | 0,957 | **0,908** | 0,422 |
| 8 (parada) | **0,999** | 0,894 | 0,599 |

Con ~1.500 imágenes de entrenamiento, el modelo memoriza: train AUROC 0,999 y
loss 0,06 mientras la pérdida de validación sube un 42%. El *early stopping*
conservó la época 3, así que el checkpoint es sano — pero `dropout: 0.3` no fue
suficiente. Lo siguiente a probar es augmentación más agresiva, congelar parte del
backbone o un modelo más pequeño.

## Grad-CAM: mira la lesión

| Grupo | n | Energía en bordes |
|---|---|---|
| Detecciones positivas | 106 | **0,140** |
| Resto de casos | 29 | 0,475 |
| Mapas nulos | 65 | — |
| *Baseline uniforme* | — | *0,510* |

0,140 frente a 0,510 sobre 106 detecciones: la atención se concentra con fuerza en
el centro, bastante mejor que el 0,266 del modelo de tórax. Tiene sentido — quien
fotografía un lunar lo encuadra en el medio, así que el sujeto ya está donde el
modelo mira.

## Sesgos que sí se pudieron medir

| Atributo | Brecha de FNR | Peor subgrupo |
|---|---|---|
| Región del cuerpo | 0,214 | antebrazo |
| Grupo de edad | 0,178 | 75+ |
| Fitzpatrick 1-3 | 0,117 | *no interpretable, ver arriba* |
| Sexo | 0,021 | despreciable |

Por sexo no hay brecha, igual que en tórax. La mayor es por **región del cuerpo**,
que es el análogo dermatológico del sesgo AP/PA que encontramos en radiografía:
una variable técnica o contextual, no demográfica, resultando ser la que más pesa.
Con los tamaños de subgrupo de este test, tómalo como una pista a confirmar, no
como una medida firme.

---

## Qué hace falta para cerrar la Fase 2 de verdad

1. **DDI o SCIN**, sin los cuales la pregunta central sigue sin respuesta. DDI
   requiere registro en Stanford AIMI; SCIN está en un bucket público de Google.
2. **Contener el sobreajuste** — augmentación más fuerte, backbone parcialmente
   congelado, o menos parámetros.
3. **Reportar siempre por separado** el subconjunto con biopsia. Publicar solo el
   0,895 sería quedarse con la mitad favorable de la historia.
4. **Varias semillas**, para saber cuánto de estas brechas es ruido de
   inicialización. En tórax ese suelo resultó ser 0,004 de AUROC.
