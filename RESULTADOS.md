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

---

# Fase 1 — el portero de calidad **no funciona**, y aquí está la prueba

Los umbrales del portero estaban puestos con criterio pero sin datos, y eso se
dijo desde el principio. SCIN permite arreglarlo: trae **1.925 casos que un
dermatólogo marcó explícitamente como «calidad de imagen insuficiente»** sobre
fotos reales de voluntarios con su propio teléfono.

Se midieron **999 imágenes** (muestra equilibrada 50/50, descargadas y
decodificadas en memoria sin tocar disco). El resultado no es que los umbrales
estuvieran mal colocados. Es peor.

## Ninguna medida predice si un dermatólogo puede evaluar la foto

AUC de cada medida contra «no evaluable»:

| Medida | AUC |
|---|---|
| Nitidez (normalizada, central) | 0,548 |
| Tinte de color | 0,547 |
| Ratio centro/marco | 0,541 |
| Nitidez global | 0,540 |
| Estructura | 0,525 |
| Rango dinámico | 0,523 |
| Brillo medio | 0,522 |
| Reflejo especular | 0,514 |
| Fracción quemada | 0,508 |
| Lado corto (resolución) | 0,501 |
| Fracción apagada | 0,487 |

Todas entre 0,49 y 0,55: **azar**. Y no es que falte combinarlas — una regresión
logística con las once medidas juntas da un AUC de validación cruzada de
**0,5474 ± 0,037**. Tampoco lo explica el tipo de toma: separando por primer plano,
en ángulo y a distancia, el AUC se queda entre 0,526 y 0,550 en los tres.

**La señal no está en estas medidas.** No es un problema de calibración.

Consecuencia práctica, con los umbrales actuales: el portero rechaza el 24,2% de
las fotos y **solo detecta el 27,4%** de las que el dermatólogo consideró
inservibles, tirando por el camino un 21% de las buenas.

## Por qué falla: calidad fotográfica no es adecuación diagnóstica

Es la lección de la tanda, y en retrospectiva era predecible. Mis once medidas
juzgan la **fotografía**: enfoque, exposición, reflejos, tinte. Lo que juzga un
dermatólogo es si **puede diagnosticar con eso**: si la lesión está en el encuadre,
al aumento adecuado, sin vello ni ropa tapándola, con el contexto suficiente.

Una foto nítida, bien expuesta y perfectamente iluminada **del sitio equivocado**
saca sobresaliente en los once controles y es completamente inútil. Y al revés:
una foto mediocre pero centrada en la lesión puede ser diagnosticable.

El diseño «sin dependencias de aprendizaje automático» que presenté como virtud
resulta ser justo la limitación: **la adecuación diagnóstica es semántica, y no se
mide con estadísticos de píxeles.** Un portero que funcione necesita un modelo
—detectar que hay una lesión, y a qué escala—, no más filtros de señal.

## Lo que sí quedó descartado: el portero no discrimina por tono de piel

Era el riesgo que más me preocupaba, y ahora está medido. La pregunta correcta no
es «¿rechaza más en piel oscura?» —si esas fotos son peores, rechazarlas está
bien— sino **¿descarta fotos buenas de un tono más que de otro?**. Condicionando
sobre las que el dermatólogo aprobó:

| Fitzpatrick | Fotos buenas | Rechazadas por el portero |
|---|---|---|
| 1 (más clara) | 54 | **27,8%** |
| 2 | 166 | 26,5% |
| 3 | 132 | 14,4% |
| 4 | 66 | 25,8% |
| 5 | 40 | 12,5% |
| 6 (más oscura) | 7 | *n insuficiente* |

La mayor diferencia es de 15 puntos, **y el tono más castigado es el más claro**.
No hay ninguna tendencia con la oscuridad de la piel, así que la variación parece
ruido de muestreo con estas cantidades (40-166 por tono). **Sin evidencia de sesgo
por tono de piel**, que es lo que se buscaba al descartar la detección de piel por
color.

## El hallazgo que no esperaba: el patrón de referencia también se degrada

Esto es independiente de mi portero y probablemente lo más relevante de la tanda.
Tasa de «no evaluable» **según el dermatólogo**, por tono de piel:

| Fitzpatrick | «No evaluable» | n |
|---|---|---|
| 1 (más clara) | **20,6%** | 68 |
| 2 | 35,2% | 256 |
| 3 | 47,4% | 251 |
| 4 | 56,0% | 150 |
| 5 | 37,5% | 64 |
| 6 (más oscura) | **66,7%** | 21 |

De 20,6% a 66,7%: **más del triple**. El tono 5 rompe la tendencia (n=64) y el
tono 6 tiene solo 21 casos, así que la magnitud exacta no es firme — pero el
gradiente entre los tonos 1 y 4, que suman 725 casos, es difícil de atribuir al
azar.

La implicación va mucho más allá de este proyecto: **cualquier canal que filtre a
imágenes «evaluables» descarta piel oscura unas tres veces más**, y lo hace antes
de que exista ningún modelo. El sesgo entra por el patrón de referencia humano, no
por la red. Un dataset limpiado así llega ya empobrecido en los tonos que más
importan.

No se puede distinguir con estos datos si las fotos de piel oscura son
genuinamente más difíciles de evaluar —menor contraste del eritema, por ejemplo— o
si el juicio de evaluabilidad depende del tono. Ambas explicaciones tienen
consecuencias, y ninguna es tranquilizadora.

## Salvedades honestas

- La etiqueta de evaluabilidad de SCIN es **por caso**, y un caso puede tener
  hasta tres imágenes; aquí se midió la primera. Eso mete ruido. Pero no explica
  el resultado: la separación por tipo de toma no mejora nada y el modelo
  combinado se queda en 0,547.
- El valor `DEFAULT_YES` significa «el dermatólogo no la marcó como insuficiente»,
  no una aprobación activa. La clase informativa es la negativa, y es justo la que
  el portero no encuentra.
- Muestra equilibrada 50/50 por diseño; la proporción real de no evaluables en
  SCIN es del 39%.

## Qué hacer con la Fase 1

1. **No usar el portero como filtro**, tal cual está. Detecta una cuarta parte de
   las fotos malas y descarta una quinta parte de las buenas.
2. Lo que **sí** conserva valor: los mensajes accionables, la comprobación de que
   varias fotos concuerdan, y la corrección de color con referencia. Nada de eso
   depende de los umbrales.
3. Para que el portero sirva hace falta **un modelo de adecuación**, entrenado
   contra estas mismas etiquetas de SCIN. Es factible: 5.033 casos etiquetados y
   accesibles sin registro.
