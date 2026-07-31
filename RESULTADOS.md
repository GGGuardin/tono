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

---

# El portero aprendido tampoco funciona, y ahora sabemos por qué

Si las reglas fotográficas fallaban porque la adecuación diagnóstica es semántica,
lo lógico era entrenar un modelo que la aprendiera. Se hizo: EfficientNet-B0 sobre
los 5.033 casos de SCIN, con división por caso.

| Enfoque | AUROC en test |
|---|---|
| Reglas fotográficas (once medidas) | 0,547 |
| **Modelo entrenado** | **0,544** (IC95% 0,509–0,580) |

**Idéntico. Y ambos son azar.** No hubo mejora alguna.

## El modelo sí aprendió — el problema es que no había nada que generalizar

| Época | Train AUROC | Val AUROC |
|---|---|---|
| 1 | 0,558 | 0,526 |
| 4 | 0,761 | 0,558 |
| 8 | **0,898** | **0,554** |

El AUROC de entrenamiento sube hasta 0,898: el modelo **es perfectamente capaz de
memorizar** esas etiquetas. Pero la validación se queda clavada en 0,55 desde la
primera época hasta la última. Puede aprenderse los casos uno a uno y no extraer
ninguna regla que sirva para casos nuevos.

Eso ya no es un problema de arquitectura, ni de datos insuficientes, ni de
hiperparámetros. Apunta a la etiqueta.

## La causa: los dermatólogos no coinciden entre ellos

SCIN incluye hasta tres valoraciones independientes por caso. En los 716 casos con
más de una, el acuerdo sobre **si la foto es evaluable** es este:

| Comparación | Acuerdo bruto | Kappa de Cohen |
|---|---|---|
| Valorador 1 vs 2 | 69,1% | **0,210** |
| Valorador 1 vs 3 | 70,8% | **0,249** |
| Valorador 2 vs 3 | 62,0% | **0,154** |

Escala habitual: por debajo de 0,20 el acuerdo es *insignificante*; entre 0,21 y
0,40, *débil*. Los tres pares caen ahí.

**Ningún modelo puede predecir una etiqueta que sus propios anotadores no
reproducen.** El 0,544 obtenido no está lejos del techo que impone esa
inconsistencia: estamos midiendo el ruido del criterio humano, no un fallo del
modelo.

Y explica limpiamente por qué los dos enfoques dan el mismo número: **no fallan por
cómo miran la foto, fallan porque la pregunta, tal como está planteada, no tiene
una respuesta estable.**

## Sobre la auditoría de equidad de este modelo

La función de veredicto marcó «CAUTELA» por una brecha de FPR de 0,157 entre tonos.
**Ese aviso no significa nada aquí**, y conviene decirlo antes que presumir de una
comprobación que no aplica:

| Fitzpatrick | n | FPR (fotos buenas descartadas) |
|---|---|---|
| 1 | 75 | 0,485 |
| 2 | 277 | 0,500 |
| 3 | 273 | 0,529 |
| 4 | 149 | 0,495 |
| 5 | 74 | 0,372 |

Todas rondan 0,50 porque **el modelo es esencialmente aleatorio**: descarta la
mitad de todo, sea cual sea el tono. La correlación con la oscuridad de la piel es
incluso negativa (−0,607). Un análisis de sesgo sobre un clasificador que no
clasifica no informa de nada, y presentarlo como «no discrimina» sería vender una
garantía vacía.

La buena noticia colateral: **el temor de que el modelo copiara el sesgo de la
etiqueta no se materializó**. Pero tampoco se puede descartar, porque el modelo
nunca llegó a funcionar.

## Qué significa esto para el proyecto

1. **El control de calidad automático de fotos dermatológicas, tal como está
   planteado, no es un problema resuelto ni fácil.** Dos enfoques independientes
   —reglas y aprendizaje— dan azar, y la causa está medida.
2. **Antes de construir un portero hace falta una definición operativa de
   «evaluable» que dos expertos puedan aplicar igual.** Sin eso no hay etiqueta, y
   sin etiqueta no hay modelo. Es un problema de diseño de anotación, no de redes
   neuronales.
3. **Lo que queda en pie del sistema original**: los mensajes accionables, la
   concordancia entre varias fotos —que estima incertidumbre **sin necesitar
   ninguna etiqueta**— y la corrección de color con referencia.
4. La vía más prometedora sería un objetivo **verificable en lugar de subjetivo**:
   por ejemplo, si la foto permite localizar la lesión, o si dos fotos de la misma
   lesión producen la misma predicción. Ambos se pueden comprobar sin depender de
   que un experto declare «esto se puede evaluar».

---

# La pregunta central, por fin respondida: ¿funciona igual en piel oscura?

Con el portal de Stanford caído y PAD-UFES sin representación (6 casos malignos en
tonos 4-6), **Fitzpatrick17k** desbloqueó la medición: 1.079 imágenes en tonos
oscuros, 509 de ellas malignas.

El tono se agrupa en tres bandas —clara (1-2), media (3-4), oscura (5-6)— porque
las dos anotaciones de Fitzpatrick del propio dataset **coinciden exactamente solo
el 47,9% de las veces** (91% con margen de un tono). Seis clases exactas medirían
sobre todo el ruido del anotador.

## Experimento A — un modelo entrenado con piel clara, aplicado a todo

DenseNet-121 entrenada en PAD-UFES (Brasil, fotos de móvil, piel mayoritariamente
clara) y evaluada sobre Fitzpatrick17k completo.

| Conjunto | n | AUROC |
|---|---|---|
| Test interno (PAD-UFES) | 456 | **0,899** (0,871–0,925) |
| Fitzpatrick17k completo | 4.492 | **0,684** (0,669–0,700) |

Caída de **0,215**. Pero lo interesante es cómo se reparte:

| Banda de tono | n | Prevalencia | AUROC | FNR |
|---|---|---|---|---|
| 1-2 clara | 2.310 | 0,517 | 0,6924 | 0,278 |
| 3-4 media | 1.597 | 0,474 | 0,6579 | **0,361** |
| **5-6 oscura** | **408** | 0,507 | **0,6985** | 0,300 |

**No hay degradación en piel oscura. Es la banda con el AUROC más alto de las
tres**, y la peor tasa de infradiagnóstico está en la banda *media*, no en la
oscura. La prevalencia es plana entre bandas, así que no lo explica un cambio de
composición.

La lectura: **el fallo de transferencia es uniforme, no racial.** Lo que hunde al
modelo es el cambio de dominio —fotos de móvil frente a imágenes de atlas, y
etiquetas de biopsia frente a diagnóstico clínico—, y ese golpe cae por igual sobre
toda la piel.

Es un resultado que contradice la hipótesis de partida del proyecto, y por eso vale
la pena.

## Experimento B — entrenar con representación

Misma arquitectura y configuración, entrenando sobre Fitzpatrick17k.

| Conjunto | n | AUROC |
|---|---|---|
| Test interno | 899 | **0,916** (0,896–0,935) |

| Banda de tono | n | AUROC | Sensibilidad | **FNR** |
|---|---|---|---|---|
| 1-2 clara | 472 | 0,8997 | 0,848 | 0,152 |
| 3-4 media | 317 | 0,9410 | 0,897 | 0,103 |
| **5-6 oscura** | **69** | **0,9100** | **0,763** | **0,237** |

Aquí aparece algo, pero **no es lo que suele denunciarse**. El AUROC en piel oscura
(0,910) es equivalente al de piel clara (0,900): **la capacidad de discriminar es
la misma**. Lo que cambia es el punto de operación — la sensibilidad cae a 0,763 y
la tasa de infradiagnóstico sube a 0,237, entre 1,6 y 2,3 veces la de las otras
bandas.

**Es un problema de calibración, no de discriminación.** Exactamente el mismo
patrón que encontramos en el proyecto de tórax con el conjunto pediátrico: AUROC
0,922 y aun así perdiendo el 64% de los casos, porque el umbral venía de otra
población. Aquí el umbral es único para las tres bandas y encaja peor en la oscura,
que además tiene mayor prevalencia (0,551 frente a 0,46-0,47).

Y eso importa porque **la solución ya está construida**: la corrección de a priori
de `src/calibration.py`, aplicada por subgrupo, o simplemente un umbral por banda.
No hace falta reentrenar nada.

**Salvedad que impide cerrarlo**: la banda oscura del test tiene 69 imágenes, unas
38 positivas, o sea unos 9 falsos negativos. Mover dos casos cambia la cifra de
forma apreciable. Es una señal que hay que confirmar, no un veredicto.

## Qué se puede afirmar y qué no

**Se puede afirmar:** en ninguno de los dos modelos la **capacidad de discriminar**
empeora en piel oscura. En A la banda oscura es incluso la mejor; en B, equivalente
a la clara.

**No se puede afirmar** que el sistema sea equitativo: en B el punto de operación
infradiagnostica más en piel oscura, y aunque la muestra es pequeña, la dirección
coincide con lo documentado en la literatura.

**No se puede comparar A con B directamente** para atribuir la mejora a la
representación: A se evaluó fuera de su dominio y B dentro del suyo. El salto de
0,699 a 0,910 en piel oscura mezcla ambos efectos.

## Salvedades del dataset

- Fitzpatrick17k son **atlas dermatológicos**: diagnóstico clínico o de libro, **no
  confirmado por biopsia**. Verdad de referencia más débil que DDI.
- Son imágenes de atlas, **no fotos de móvil**: el dominio no coincide con el caso
  de uso real.
- Sin identificador de paciente: el split por paciente equivale al split por imagen.
- Ambos modelos **sobreajustan con fuerza** — el de Fitzpatrick17k llega a un AUROC
  de entrenamiento de 0,99998 con la validación en 0,90. El *early stopping* salva
  el checkpoint, pero hay margen claro de mejora con regularización.

## Lo siguiente

1. **Aplicar la corrección de umbral por banda** y volver a medir el FNR. Es una
   línea de código y ataca directamente lo único que salió mal.
2. **Confirmar con DDI** cuando el portal de Stanford vuelva: biopsia como verdad y
   tonos diversos por diseño.
3. **Contener el sobreajuste** antes de sacar conclusiones más finas.
