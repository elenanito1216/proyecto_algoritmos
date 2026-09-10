# Análisis crítico del proyecto

**Detección de Grietas e Inclinación de Elementos Estructurales para la Evaluación de Riesgo en Edificaciones**

Algoritmos y Programación · Ingeniería en Inteligencia Artificial · Universidad Industrial de Santander · 2026-2

---

> **Estado de este documento.** Completo. Todas las cifras proceden de artefactos de `reports/metricas/`, generados por `scripts/train_baseline.py`, `scripts/train_transfer.py`, `scripts/export_tflite.py`, `scripts/evaluate.py` y `scripts/validar_inclinacion.py`; la fuente de cada tabla está indicada para que cualquier número sea trazable hasta el archivo que lo respalda.
>
> **Aviso importante sobre las cifras.** El 15.27 % del conjunto de prueba resultó estar duplicado en el de entrenamiento (§3.7). Donde importa, se reportan las métricas del **conjunto depurado**; las del conjunto completo se conservan para poder comparar contra los artefactos previos y siempre se indican como tales.
>
> Las lagunas que quedan son de **tamaño de muestra**, no de ejecución: la validación de la inclinometría (§3.6) se apoya en una sola fotografía, por las razones que allí se explican, y el conjunto de fotos propias son 40 imágenes (§5.7). Ambas están declaradas donde corresponde en lugar de disimuladas.

---

## 1. Decisiones de arquitectura y su justificación

### 1.1 Por qué MobileNetV2 y no EfficientNet-Lite

El enunciado contemplaba EfficientNet-Lite. **No está disponible en `keras.applications`**: la familia Lite existe únicamente como modelo de TensorFlow Hub o a través de TFLite Model Maker, no como constructor nativo de Keras. Forzarla habría añadido una dependencia externa, un formato de pesos distinto y un camino de exportación a TFLite más frágil, sin una ganancia clara sobre el objetivo del proyecto.

Se usa **MobileNetV2**, que:

- está incluido en Keras, con pesos de ImageNet descargables sin dependencias extra;
- fue diseñado explícitamente para inferencia en dispositivos móviles;
- cuantiza a int8 sin operaciones no soportadas por el intérprete de TFLite;
- es el estándar de facto contra el que se compara la literatura de detección de grietas en dispositivos embebidos.

La decisión está documentada también en `src/models/transfer.py` y se registra en el JSON de cada corrida (`nota_arquitectura`), para que quede trazada en los artefactos y no solo en la memoria.

### 1.2 Por qué `alpha = 0.75`

El *width multiplier* escala el número de canales de cada capa. Con `alpha = 0.75` la base pasa de ~2.26 M a ~1.38 M parámetros (−39 %) y el número de multiplicaciones-acumulaciones cae aproximadamente con `alpha²`, es decir, al ~56 % del original.

El razonamiento es que la tarea es de **textura binaria** —¿esta superficie está fisurada?— y no de reconocimiento fino entre mil categorías. La capacidad sobrante de `alpha = 1.0` no se aprovecha, pero sí se paga en latencia y en tamaño de descarga. En un proyecto cuya restricción explícita es funcionar en un equipo modesto, esa capacidad ociosa es un costo sin contrapartida.

### 1.3 Por qué 160×160 y no 224×224

El costo de una capa convolucional es lineal en el número de píxeles y, por tanto, **cuadrático en el lado de la imagen**:

```
costo(160×160) / costo(224×224) = (160/224)² = 0.51
```

La mitad de cómputo. La pregunta es qué se pierde. Una grieta estructuralmente relevante ocupa entre 2 y 4 píxeles de ancho en el encuadre típico del dataset; a 160×160 esa firma de alta frecuencia sobrevive al submuestreo. A 128×128 se ahorraría un 36 % adicional, pero las fisuras capilares empiezan a difuminarse, y **en este dominio el error caro es el falso negativo**. Ahorrar un tercio de cómputo a cambio de perder recall en la clase peligrosa es un mal negocio.

Restricción adicional: los pesos preentrenados de MobileNetV2 solo existen para resoluciones de {96, 128, 160, 192, 224}. 160 es el valor admisible que mejor equilibra ambas presiones.

### 1.4 Por qué un motor de reglas y no un segundo modelo

Un clasificador de "riesgo" necesitaría etiquetas de riesgo. Esas etiquetas no existen en ningún dataset público y solo un ingeniero estructural matriculado puede emitirlas. Entrenar sobre etiquetas inventadas por el equipo produciría un modelo que aprende las opiniones de cuatro estudiantes de pregrado, presentadas con la autoridad de una red neuronal.

El motor de reglas (`src/risk/reglas.py`) es una **función pura** que:

- es **auditable**: cada regla cita el criterio de ingeniería que la motiva;
- es **explicable**: la salida incluye qué reglas se dispararon y con qué valores;
- es **recalibrable**: todos los umbrales están en `config.yaml`; un ingeniero puede ajustarlos sin tocar una línea de Python ni reentrenar nada;
- es **testeable**: 33 pruebas unitarias cubren su comportamiento, incluidas propiedades de monotonía (más evidencia de grieta nunca puede rebajar el nivel de riesgo).

---

## 2. Análisis de complejidad computacional

### 2.1 Complejidad asintótica del módulo clásico

Sea `n = H × W` el número de píxeles y `m` el número de píxeles de borde tras Canny (típicamente entre el 1 % y el 5 % de `n`).

| Etapa | Complejidad | Nota |
|---|---|---|
| Conversión a escala de grises | `O(n)` | Una combinación lineal por píxel |
| Desenfoque gaussiano, kernel `k` | `O(n·k)` | Separable; con `k = 5` fijo, es `O(n)` |
| Gradientes Sobel | `O(n)` | Kernel fijo 3×3 |
| Supresión de no máximos | `O(n)` | Una comparación local por píxel |
| Histéresis de Canny | `O(n)` amortizado | Cada píxel entra una vez en la pila |
| **HoughLinesP** | `O(m·T)` | `T` = número de valores discretos de θ; con resolución de 1°, `T = 180` |
| Filtrado y mediana ponderada | `O(L log L)` | `L` = segmentos detectados, del orden de decenas |

**Total: `O(n + m·T)`.** Como `m << n` y `T` es una constante de configuración, el pipeline es **lineal en el número de píxeles**.

Consecuencia práctica: el tiempo de pared está dominado por Canny, y **la única optimización que importa es reducir la resolución antes de procesar**, porque `n` cae cuadráticamente con el lado. Por eso la aplicación limita el lado mayor a 1024 px antes de la inclinometría: una foto de móvil de 12 MP multiplicaría por doce el trabajo sin aportar una sola línea recta adicional que fuese informativa.

### 2.2 Complejidad asintótica del pipeline de inferencia

Una capa convolucional con entrada `H×W×C_in`, kernel `k×k` y `C_out` filtros cuesta

```
O(H · W · k² · C_in · C_out)
```

multiplicaciones-acumulaciones. Tres consecuencias:

1. **Cuadrático en la resolución** (justifica 160×160, §1.3).
2. **MobileNetV2 usa convolución separable en profundidad**: descompone la operación en un filtrado `k×k` por canal (`H·W·k²·C_in`) seguido de una mezcla `1×1` entre canales (`H·W·C_in·C_out`). La razón frente a la convolución densa es `1/C_out + 1/k²`, es decir, **entre 8 y 9 veces menos operaciones con `k = 3`**. Ahí está el grueso de su eficiencia, no en tener menos capas.
3. **`alpha` escala `C_in` y `C_out` simultáneamente**, así que el costo cae aproximadamente con `alpha²` (justifica `alpha = 0.75`, §1.2).

La **cuantización a int8 no cambia el orden asintótico**: cambia la constante. Aritmética entera de 8 bits en lugar de coma flotante de 32, y cuatro veces menos ancho de banda de memoria. En CPU móvil eso suele traducirse en una aceleración real de 2× a 3×, que es exactamente lo que la tabla de §3.2 debe confirmar o desmentir con medidas propias.

### 2.3 Metodología de medición

Las cifras de latencia se obtienen con `src/eval/complejidad.py` bajo un protocolo explícito, porque una medida de latencia sin protocolo no significa nada:

- **60 repeticiones** por modelo (el enunciado exige ≥ 50), con **10 de calentamiento descartadas**. La primera llamada incluye construcción del grafo, reserva de buffers y llenado de cachés: incluirla contaminaría la media con un atípico de cientos de milisegundos.
- Se invoca `modelo(x, training=False)` y no `modelo.predict()`. Este último añade la maquinaria de callbacks y el troceado en lotes, que no forma parte del costo de inferir.
- **Lote de 1 imagen**: es el escenario real de la aplicación, no el de máximo rendimiento.
- **TFLite con un solo hilo**, para representar un dispositivo modesto. Medir con todos los hilos de un portátil daría una cifra optimista que no describe al dispositivo objetivo.
- La memoria se mide como **incremento del RSS del proceso** con `psutil`, no con `tracemalloc`: el grueso de la memoria de TensorFlow se reserva en buffers de C++ que el rastreador del asignador de Python no ve.

---

## 3. Resultados

**Configuración de las corridas reportadas.** Dataset de 58 138 imágenes (24 001 + 16 695 en entrenamiento; 41.03 % de la clase positiva), partición estratificada 70/15/15 con semilla 42: 40 696 / 8 721 / 8 721. Entrada 160×160×3, lote 32, pesos de clase balanceados. Hardware: CPU AMD Ryzen (AMD64 Family 23 Model 160), Windows 10, 15.2 GB de RAM, TensorFlow 2.19.0, **sin GPU**.

Todas las cifras salen de artefactos de `reports/metricas/`; la fuente de cada tabla está indicada.

### 3.1 Desempeño: línea base frente a transfer learning

Conjunto de **prueba** (8 721 imágenes: 5 143 sin grieta, 3 578 con grieta), umbral 0.50.
Fuente: `evaluacion.json` → `modelos.<etiqueta>.test.metricas`

> ⚠️ **Estas cifras están infladas.** El 15.27 % del conjunto de prueba resultó estar duplicado en el de entrenamiento, y los modelos aciertan prácticamente el 100 % de esas imágenes. Las cifras honestas son las de la segunda tabla. El análisis completo está en **§3.7**.

**Conjunto de prueba completo (8 721 imágenes) — contaminado:**

| Modelo | Exactitud | Precisión | Recall | F1 | ROC-AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| Línea base (CNN propia) | 0.9490 | 0.9869 | 0.8874 | 0.9345 | 0.9854 | 403 |
| MobileNetV2 fine-tuned | 0.9663 | 0.9943 | 0.9231 | 0.9574 | 0.9921 | 275 |
| MobileNetV2 TFLite int8 | 0.9570 | 0.9969 | 0.8980 | 0.9449 | 0.9816 | 365 |

**Conjunto de prueba depurado (7 389 imágenes) — sin duplicados de entrenamiento:**

| Modelo | Exactitud | Precisión | Recall | F1 | ROC-AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| Línea base (CNN propia) | 0.9400 | 0.9822 | 0.8493 | 0.9109 | 0.9789 | 402 |
| **MobileNetV2 fine-tuned** | **0.9603** | 0.9925 | **0.8969** | **0.9423** | **0.9887** | **275** |
| MobileNetV2 TFLite int8 | 0.9495 | 0.9961 | 0.8635 | 0.9251 | 0.9741 | 364 |

**A partir de aquí, el resto del informe usa las cifras del conjunto completo** cuando compara contra artefactos generados antes de detectar la fuga, y lo indica expresamente. Las conclusiones cualitativas —ordenación de los modelos, ventaja del fine-tuning, costo de la cuantización— **no cambian**: la fuga afecta a los tres modelos por igual y no altera su orden relativo.

Matrices de confusión sobre el conjunto completo (VN · FP · FN · VP): línea base 5101 · 42 · 403 · 3175; MobileNetV2 5124 · 19 · 275 · 3303; TFLite int8 5133 · 10 · 365 · 3213.

El efecto del fine-tuning se mide sobre **validación**, no sobre prueba, porque es ahí donde se decidió adoptarlo.
Fuente: `entrenamiento_mobilenetv2.json` → `metricas_val_congelado` / `metricas_val_finetuned`

| Etapa | Exactitud | Precisión | Recall | F1 | ROC-AUC | FN | FP |
|---|---|---|---|---|---|---|---|
| 1 · base congelada (10 épocas) | 0.9618 | 0.9853 | 0.9206 | 0.9519 | 0.9891 | 284 | 49 |
| 2 · fine-tuning (5 épocas, LR 1e-5) | **0.9678** | 0.9922 | **0.9287** | **0.9594** | **0.9918** | **255** | **26** |

**Respuestas a las tres preguntas del apartado.**

1. **El preentrenamiento aporta +0.0229 de F1** (0.9345 → 0.9574) y, más relevante, **reduce los falsos negativos de 403 a 275: un 32 % menos de grietas que se escapan**. El costo es de **52× más parámetros** (28 145 → 1 464 113) y **8.7× más latencia** en formato `.keras` (17.1 → 148.4 ms). Dicho sin adornos: se compran 2.3 puntos de F1 muy caros. Lo que salva el trade-off no es el transfer learning por sí solo, sino la cuantización posterior (§3.2): el TFLite int8 conserva la mayor parte de la ganancia y además es **3.5× más rápido que la propia línea base**.
2. **El fine-tuning mejora de verdad, no sobreajusta.** Sobre validación mejora las cuatro métricas a la vez —F1 +0.0075, ROC-AUC +0.0027— y reduce simultáneamente falsos negativos (284 → 255) y falsos positivos (49 → 26). Cuando una segunda etapa mejora ambos errores a la vez, no está desplazando el umbral: está aprendiendo. Descongelar 19 capas con LR 1e-5 fue una intervención conservadora y acertada.
3. **La cuantización int8 cuesta 0.0125 de F1** (0.9574 → 0.9449) y **90 falsos negativos más** (275 → 365), a cambio de **8.0× menos tamaño y 30.8× menos latencia**. En el conjunto de prueba el intercambio es claramente favorable. Fuera de distribución **no lo es**, y esa es la advertencia de §3.5.

### 3.2 Costo computacional

Latencia medida con lote de 1 imagen, 60 repeticiones tras 10 de calentamiento; TFLite con **un solo hilo** para representar un dispositivo modesto.
Fuente: `comparativa.csv` y `exportacion_tflite.json`

| Modelo | Parámetros | Tamaño (MB) | Latencia media (ms) | Desv. est. (ms) | p95 (ms) | img/s | Memoria (MB) |
|---|---|---|---|---|---|---|---|
| Línea base | 28 145 | 0.39 | 17.09 | 1.08 | 18.78 | 58.5 | 0.00 |
| MobileNetV2 `.keras` | 1 464 113 | 13.90 | 148.37 | 1.80 | 151.06 | 6.7 | 0.00 |
| MobileNetV2 TFLite float32 | — | 5.46 | no medida | — | — | — | — |
| **MobileNetV2 TFLite int8** | — | **1.74** | **4.82** | 0.77 | 6.35 | **207.5** | 0.18 |

Dos aclaraciones de método, para que las cifras se lean bien:

- La variante **float32 se exportó pero no se perfiló**: `evaluate.py` mide el artefacto declarado en `app.archivo_tflite`, que es el int8. Su tamaño (5.46 MB) sí permite separar las dos fuentes de compresión: **convertir a TFLite aporta 2.5×** (13.90 → 5.46 MB, por eliminar el estado del optimizador y el grafo de entrenamiento) y **cuantizar aporta otros 3.1×** (5.46 → 1.74 MB, por pasar de 32 a 8 bits). Total 8.0×.
- El **incremento de memoria de 0.00 MB** en los modelos Keras no significa que no consuman: significa que TensorFlow ya había reservado sus buffers durante el calentamiento, así que la inferencia en régimen no pide RAM adicional. El consumo absoluto del proceso fue de 990 MB con la línea base y 2 032 MB con MobileNetV2. El TFLite int8 sí muestra un incremento medible (0.18 MB) porque el intérprete reserva sus tensores bajo demanda.

**El resultado que sostiene la viabilidad móvil:** el TFLite int8 corre a **4.82 ms por imagen (207 img/s) ocupando 1.74 MB**, en la CPU de un portátil sin GPU y con un solo hilo. Es **30.8× más rápido que el `.keras`** del que procede y **3.5× más rápido que la línea base**, que tiene 52 veces menos parámetros. Ahí se ve que la eficiencia no viene del tamaño del modelo sino del formato de ejecución.

### 3.3 Costo de entrenamiento

Fuente: `entrenamiento_*.json` → `tiempos` y `dispositivo`

| Experimento | Dispositivo | Épocas | s/época | Total (min) |
|---|---|---|---|---|
| Línea base | CPU | 17 (parada temprana; mejor época: 12) | 435.4 | 123.4 |
| MobileNetV2 etapa 1 (congelado) | CPU | 10 | 473.2 | 78.9 |
| MobileNetV2 etapa 2 (fine-tuning) | CPU | 5 | 507.3 | 42.3 |

**Costo total del proyecto: 244.6 minutos (4 h 5 min) de CPU** para los tres modelos.

Una observación que contradice la expectativa inicial y merece explicarse. Se anticipaba que la etapa 1 sería *sustancialmente* más rápida que la etapa 2, porque con la base congelada solo se retropropaga por la cabeza densa. La medición dice otra cosa: **473.2 frente a 507.3 s/época, apenas un 7 % de diferencia**. La razón es que el paso hacia adelante por las 154 capas de MobileNetV2 hay que darlo igual en ambas etapas, y en CPU ese paso —dominado por convoluciones separables, limitadas por ancho de banda de memoria— es el que manda. La retropropagación por 19 capas adicionales añade poco al lado de eso. En una GPU, donde el cómputo no es el cuello de botella, la diferencia sería mucho más marcada.

Igual de revelador: la línea base, con **52 veces menos parámetros**, tardó **435.4 s/época** frente a los 473.2 de MobileNetV2. Solo un 8 % menos. Con lotes de 32 imágenes a 160×160 en CPU, el tiempo lo consume la lectura y decodificación del dataset, no la aritmética del modelo. Es la misma lección que la de §3.2 vista desde el otro lado: **el cuello de botella casi nunca está donde uno supone antes de medir**.

### 3.4 Generalización: dataset público frente a fotos propias

Conjunto propio: 40 fotografías tomadas por el equipo (20 con grieta, 20 sin grieta), nunca usadas para entrenar ni para elegir hiperparámetros.

> **Nota sobre el tamaño de la muestra.** El conjunto propio se amplió después a **60 fotografías** (30 y 30). Las cifras de §3.4 a §3.8 y de §4.5 son las medidas sobre las 40 originales y se conservan tal cual: reescribirlas con los datos nuevos ocultaría que las decisiones de diseño se tomaron con la información disponible entonces. Las mediciones de **§4.6 usan las 60**, y su tabla incluye la línea base recalculada sobre la misma muestra, de modo que la comparación que sustenta la conclusión es interna y homogénea.
Fuente: `evaluacion.json` → `modelos.<etiqueta>.test` frente a `.propias`

| Modelo | F1 `test` (depurado) | F1 en `propias` | **Brecha** | Recall propias | Precisión propias | FN | FP |
|---|---|---|---|---|---|---|---|
| Línea base | 0.9109 | **0.7500** | −0.1609 | 0.60 | 1.000 | 8/20 | 0 |
| MobileNetV2 | 0.9423 | 0.6667 | −0.2756 | 0.50 | 1.000 | 10/20 | 0 |
| TFLite int8 | 0.9251 | 0.6207 | **−0.3044** | 0.45 | 1.000 | 11/20 | 0 |

*Se usa el F1 del conjunto **depurado** (§3.7): comparar contra el conjunto contaminado exageraría la brecha atribuyendo a desplazamiento de dominio lo que en realidad era memorización. Con el conjunto completo las brechas serían −0.1845, −0.2907 y −0.3242.*

**Este es el resultado más importante de todo el informe, y es incómodo por partida triple.**

**(a) El orden se invierte.** El modelo que gana en el dataset público es el que peor generaliza a fotografías reales. La CNN propia de 28 145 parámetros aguanta mejor (F1 0.750) que MobileNetV2 (0.667). Cuanto más se especializó un modelo en la textura del hormigón del dataset, menos transfirió a paredes reales. Un ranking construido solo sobre el conjunto de prueba habría recomendado exactamente el peor de los tres para uso en campo.

**(b) Se pierde entre el 40 % y el 55 % de las grietas reales.** El recall cae de ~0.90 a 0.45–0.60. En términos operativos: de cada 20 grietas fotografiadas por el equipo, MobileNetV2 encuentra 10.

**(c) Cero falsos positivos, en los tres modelos.** Los 20 negativos se clasifican correctamente siempre. El sistema es **ultraconservador fuera de su distribución**: ante una textura que no reconoce, responde «no hay grieta». Es justamente la dirección peligrosa en seguridad estructural, y es un patrón conocido de los clasificadores ante desplazamiento de dominio: sus salidas se desplazan hacia la clase mayoritaria del entrenamiento en lugar de expresar incertidumbre.

El ROC-AUC sobre fotos propias (0.9100 línea base, 0.8400 MobileNetV2, 0.8288 TFLite) confirma que **sí queda señal discriminante**: el problema no es que los modelos estén ciegos, sino que su umbral está mal calibrado para este dominio. Esa distinción es la que abre la puerta a §3.5.

Con 40 imágenes, los intervalos de confianza son anchos y estas cifras sirven para detectar un fallo grueso de generalización, no para estimarlo con precisión (§5.7). El fallo grueso está detectado.

### 3.5 La cuantización int8 destruye el margen de recalibración

Este resultado no estaba previsto y salió de analizar las probabilidades almacenadas en `evaluacion.json` → `modelos.<etiqueta>.propias.probabilidades`.

Probabilidad que cada modelo asigna a las 20 fotos propias **que sí tienen grieta**:

| Modelo | mínimo | **mediana** | máximo |
|---|---|---|---|
| MobileNetV2 `.keras` | 0.0014 | **0.6383** | 1.0000 |
| MobileNetV2 TFLite int8 | 0.0000 | **0.0176** | 0.9961 |

Y el efecto de bajar el umbral de decisión sobre esas mismas 40 fotos:

| Umbral | `.keras` recall | `.keras` FN | **int8 recall** | **int8 FN** |
|---|---|---|---|---|
| 0.50 | 0.50 | 10 | 0.45 | 11 |
| 0.25 | 0.55 | 9 | 0.45 | 11 |
| 0.10 | **0.70** | **6** | 0.45 | 11 |
| 0.05 | 0.75 | 5 | 0.45 | 11 |

En el modelo `.keras`, bajar el umbral a 0.10 recupera 4 de las 10 grietas perdidas a costa de un solo falso positivo. **En el int8 no cambia absolutamente nada**: de 0.50 a 0.05, el recall permanece clavado en 0.45 y los 11 falsos negativos no se mueven ni uno.

La causa es que la cuantización comprime el rango dinámico de la salida, y ante entradas fuera de distribución ese rango se **satura en cero**. La mediana pasa de 0.6383 a 0.0176: la información que permitía recalibrar el modelo ya no existe en el artefacto.

**La implicación práctica es seria y no aparece en ninguna métrica del conjunto de prueba.** Sobre `test`, el int8 parece una ganga: −0.0125 de F1 a cambio de 8× menos tamaño y 30.8× menos latencia. Lo que esa comparación oculta es que **el modelo cuantizado pierde la capacidad de ser ajustado al dominio de despliegue**. El umbral de decisión —la única herramienta barata que tiene el proyecto para priorizar recall (§4.1)— deja de funcionar justo en el escenario para el que se diseñó.

Para el despliegue real esto sugiere una arquitectura distinta de la que se supone por defecto: usar int8 como **filtro rápido de primer paso** y reservar el `.keras` (o al menos una variante float32) para los casos dudosos, en lugar de sustituir uno por otro sin más.

### 3.6 Validación de la inclinometría

Fuente: `reports/metricas/validacion_inclinacion.json`

**Metodología.** No existe *ground truth* del desaplome real: nadie midió esas paredes con un inclinómetro. Lo que sí se puede medir es la **fidelidad diferencial** del estimador. Se toma una fotografía real, se rota un ángulo exacto conocido `α` y se comprueba que la estimación se desplaza exactamente `−α` respecto a la medida de referencia sin rotar (el signo se justifica en `src/vision/inclinacion.py`: `cv2` rota el contenido en sentido antihorario para `α > 0`, con lo que el tope del elemento se desplaza a la izquierda). Cada imagen rotada se recorta al 80 % central para eliminar los bordes artificiales que la rotación introduce en las esquinas y que Canny detectaría como rectas perfectas inexistentes.

Imagen de referencia: `WhatsApp Image 2026-08-15 at 11.17.17 AM (3).jpeg`, con desaplome medido de **+0.00°** sustentado por **9 segmentos coherentes**.

| Rotación aplicada | Ángulo esperado | Ángulo estimado | Error | Segmentos coherentes |
|---|---|---|---|---|
| +2° | −2.000° | −1.935° | 0.065° | 11 |
| −2° | +2.000° | +1.936° | 0.064° | 8 |
| +5° | −5.000° | −4.970° | 0.030° | 19 |
| −5° | +5.000° | +4.998° | **0.002°** | 16 |
| +10° | −10.000° | −9.945° | 0.055° | 17 |
| −10° | +10.000° | +9.982° | 0.018° | 17 |

**Error medio global: 0.039°. Error máximo: 0.065°. Casos fiables: 6 de 6.**

**Criterio de aceptación: superado con holgura.** El umbral de aceptación era ~0.5° de error medio; el resultado es **trece veces mejor**. Con un error de 0.039°, el método separa sin ambigüedad el umbral de desaplome severo (2.0°) del de atención (1.0°): el margen es dos órdenes de magnitud mayor que la incertidumbre de la medida. Las reglas **R5 y R6 del motor de riesgo quedan respaldadas empíricamente** en cuanto a la precisión del ángulo que consumen.

Obsérvese además que el error no crece con la magnitud de la rotación —0.065° a 2°, 0.055° a 10°— ni muestra sesgo de signo. Eso indica que el residuo es **ruido de discretización de la transformada de Hough** (resolución angular de 1° en el acumulador, refinada por la mediana ponderada de 8-19 segmentos), no un error sistemático del método.

#### La otra cara: la cobertura es del 2.5 %

Un resultado de exactitud tan bueno no puede presentarse sin el dato que lo acompaña: **de las 40 fotografías propias, solo 1 produjo una estimación fiable**. Las otras 39 se descartaron así:

| Motivo del descarte | Fotos |
|---|---|
| No se detectó ningún borde recto | 35 |
| Se detectaron líneas, pero ninguna próxima a la vertical (±35°) | 2 |
| Menos de 3 segmentos coherentes (`min_lineas_confiables`) | 2 |

Ninguna de las 39 produjo un ángulo **erróneo**: el módulo las rechazó, devolviendo `fiable=False` con un mensaje explicativo y sin lanzar excepción. Ese comportamiento es el correcto y estaba diseñado así. Pero el balance operativo hay que decirlo sin adornos:

> **El módulo de inclinometría es muy preciso cuando funciona (0.039° de error) y funciona muy pocas veces (1 de cada 40 fotografías reales).**

La causa es de encuadre, no de algoritmo: el conjunto propio se tomó siguiendo el protocolo de captura de la **clasificación de grietas** —de 0.5 a 1.5 m, perpendicular a la superficie— que encuadra *superficie de pared*, no *elementos verticales completos*. Sin un borde largo y contrastado de arriba abajo, no hay nada que Hough pueda votar.

Esto tiene dos implicaciones prácticas:

1. **Los dos módulos necesitan fotografías distintas.** La detección de grietas quiere un primer plano de la superficie; la inclinometría quiere el elemento completo a media distancia. Una sola foto rara vez sirve para ambos, y el protocolo de captura del proyecto debería pedir explícitamente **dos tomas por elemento**. Es una corrección de método que este análisis destapa y que la aplicación no advierte hoy.
2. **En uso real, la mayoría de las veces el nivel de riesgo se decidirá solo con la fisuración.** No es un fallo silencioso —el motor de reglas emite la advertencia «no se pudo medir la inclinación» y la interfaz la muestra—, pero significa que las reglas R5, R6 y R8 se dispararán con mucha menos frecuencia de lo que su peso en el diseño sugiere.

**Limitación estadística.** Con `n = 1` imagen y 6 rotaciones, este resultado demuestra que el estimador **puede** ser muy preciso sobre una fotografía real bien encuadrada. No establece su precisión media sobre una población de fotografías, ni su comportamiento ante perspectiva marcada, contraluz o superficies texturadas. Para eso harían falta 10-15 imágenes de elementos verticales variados, y es el complemento natural de este apartado.

**Verificación adicional sobre imágenes sintéticas.** `tests/test_inclinacion.py` valida el estimador con ángulos exactos conocidos (0°, ±2°, ±5°, ±10°, 20°) y tolerancia de 1.5°, comprueba la robustez frente a segmentos espurios y verifica la fidelidad diferencial bajo rotación. Las 39 pruebas del módulo pasan. Esa batería cubre la corrección **geométrica**; la tabla de arriba cubre el comportamiento sobre **fotografía real**, que es donde entran el ruido del sensor, la compresión JPEG y la iluminación no controlada.

### 3.7 Fuga de datos: el 15 % del conjunto de prueba estaba en el de entrenamiento

Fuente: `reports/metricas/fuga_datos.json`, generado por `scripts/analizar_fuga_datos.py`

#### Por qué había que comprobarlo

El proyecto contemplaba desde el principio el riesgo de fuga por parches: `datos.split.agrupar_por_origen` reparte por superficie de origen en lugar de por imagen. Pero esa salvaguarda **depende de que el nombre del archivo codifique el origen**, y los archivos de este dataset se llaman `00001.jpg`, `00002.jpg`… No hay nada que agrupar. La protección estaba escrita pero era inerte, y el reparto fue aleatorio.

Como el nombre no dice nada, la fuga había que medirla **sobre los píxeles**.

#### Método: proponer con pHash, confirmar con correlación

1. **Proponer.** Se calcula un hash perceptual de 64 bits (DCT sobre la imagen reducida a 32×32, bloque 8×8 de bajas frecuencias sin el coeficiente DC, umbralizado por su mediana) para las 58 138 imágenes, y se busca para cada imagen de prueba su vecino más cercano en entrenamiento por distancia de Hamming. Son 355 millones de pares, resueltos vectorizados en 28 s.
2. **Confirmar.** Cada candidato se verifica píxel a píxel. **Este segundo paso no es opcional**, y descubrirlo fue parte del hallazgo: el pHash tiene solo 63 bits útiles y los parches de hormigón son extremadamente homogéneos en bajas frecuencias, así que colisiona.

El discriminador correcto resultó ser la **correlación de Pearson**, no el error absoluto medio. En superficies uniformes dos parches sin relación alguna tienen valores de gris parecidos y por tanto un MAE bajo; medido sobre estos datos, el MAE no separa (duplicados reales dan 1.7–2.3, igual que algunos pares no relacionados) mientras que la correlación separa sin ambigüedad:

| Tipo de par | Correlación | MAE |
|---|---|---|
| Duplicado exacto | 1.0000 | 0.00 |
| Misma imagen, otra compresión JPEG | 0.9955 – 0.9978 | 1.7 – 2.3 |
| Imágenes distintas | 0.0151 – 0.1871 | 20 – 27 |

Un MAE de ~2 con correlación 0.996 es la firma de la misma imagen recodificada, no de dos imágenes parecidas.

#### Resultado

| Medida | Valor |
|---|---|
| Candidatos propuestos por el pHash (distancia ≤ 5/64) | 1 370 |
| **Duplicados confirmados píxel a píxel** | **1 332** |
| Precisión del pHash como detector | 97.2 % |
| **Fracción del conjunto de prueba contaminada** | **15.27 %** |
| Contaminación equivalente en validación | 16.13 % (candidatos) |
| Duplicados con **etiqueta discordante** entre train y test | 1 |

#### La prueba de que el modelo memorizaba

Al separar el conjunto de prueba en las 1 332 imágenes duplicadas y las 7 389 limpias, el contraste no deja lugar a dudas:

| Modelo | Recall sobre las **duplicadas** | Recall sobre las **limpias** | Diferencia |
|---|---|---|---|
| Línea base | 0.9989 (910/911) | 0.8493 | **−0.150** |
| **MobileNetV2** | **1.0000 (911/911)** | 0.8969 | **−0.103** |
| TFLite int8 | 0.9989 (910/911) | 0.8635 | **−0.135** |

**MobileNetV2 acierta las 911 imágenes duplicadas sin fallar ni una, y falla el 10.3 % de las limpias.** Eso no es capacidad de generalización: es reconocimiento de imágenes ya vistas durante el entrenamiento.

El detalle que lo confirma: al depurar el conjunto, **los falsos negativos no se mueven** (275 → 275 en MobileNetV2). Se eliminan 911 positivos y los 911 eran aciertos. La fuga no aportaba dificultad, aportaba puntuación gratis.

#### Impacto sobre las métricas

| Modelo | F1 completo | F1 depurado | Δ | Recall completo | Recall depurado | Δ |
|---|---|---|---|---|---|---|
| Línea base | 0.9345 | 0.9109 | −0.0236 | 0.8874 | 0.8493 | −0.0381 |
| MobileNetV2 | 0.9574 | 0.9423 | −0.0151 | 0.9231 | 0.8969 | −0.0263 |
| TFLite int8 | 0.9449 | 0.9251 | −0.0198 | 0.8980 | 0.8635 | −0.0345 |

Tres lecturas:

1. **El impacto es real pero moderado**: entre 1.5 y 2.4 puntos de F1. No invalida el trabajo; lo corrige.
2. **La precisión apenas se mueve** (−0.002 a −0.005) mientras el **recall cae hasta 3.8 puntos**. Coherente con lo anterior: las imágenes filtradas eran positivos correctamente clasificados, así que su eliminación golpea al recall y deja intacta la precisión.
3. **La línea base era la más inflada** (−0.0236) y MobileNetV2 la menos (−0.0151). Tiene sentido: un modelo de 28 145 parámetros no puede aprender el concepto de grieta con la misma solvencia, así que se apoya más en memorizar texturas concretas. La fuga favorecía desproporcionadamente al modelo débil, lo que significa que **la ventaja real del transfer learning es mayor de lo que decía §3.1**, no menor.

#### Lo que esto cambia y lo que no

**No cambia** el orden de los modelos, la conclusión sobre el fine-tuning, el costo de la cuantización ni ninguna de las mediciones de eficiencia de §3.2 y §3.3.

**Sí cambia** la magnitud del contraste con las fotos propias: la brecha de F1 de MobileNetV2 pasa de −0.2907 a **−0.2756** frente al conjunto depurado. Sigue siendo el hallazgo dominante de §3.4, y ahora parte de una base honesta.

**Y cambia el diagnóstico.** Antes de este análisis, la explicación natural de la brecha era «desplazamiento de dominio». Ahora se sabe que **una parte de esa brecha era artificial**: el conjunto de prueba era más fácil de lo que aparentaba porque el modelo ya había visto el 15 % de él.

#### Recomendación

El dataset no permite un reparto sin fuga porque los nombres de archivo no codifican el origen. Con lo que hay, la opción correcta es **deduplicar antes de repartir**: agrupar las imágenes por hash perceptual confirmado y asignar cada grupo entero a una sola partición. Es el equivalente a `agrupar_por_origen`, pero usando los píxeles como identificador de origen en lugar del nombre. Queda propuesto en §8.

**Nota metodológica que merece la pena retener (sobre este apartado).** El primer veredicto de este análisis —basado solo en el pHash— decía 15.71 % de contaminación. Una verificación apresurada sobre 10 pares sugirió después que el 70 % eran colisiones y que la fuga era despreciable. La verificación exhaustiva sobre los 1 370 candidatos dio el número correcto: 97.2 % confirmados. **Ninguna de las dos primeras cifras era fiable, y las dos eran fáciles de creer.** La lección no es sobre hashing: es que una medición sin verificación y una verificación sin muestra suficiente fallan igual de bien.

### 3.8 Robustez sin reentrenar: TTA, ensemble y recalibración del umbral

Fuente: `reports/metricas/robustez.json`, generado por `scripts/evaluar_robustez.py`

Ante la brecha de §3.4 —recall de 0.50 sobre fotografías propias— se evaluaron tres técnicas que **no requieren reentrenar** y operan sobre los modelos ya guardados.

1. **Test-time augmentation (TTA).** Se predice sobre varias vistas del grupo diédrico D4 (volteos y giros de 90°) y se agregan las probabilidades. Las transformaciones preservan la etiqueta en este dominio: una grieta girada sigue siendo una grieta, la misma razón por la que el aumento de entrenamiento incluye volteo vertical. Se probaron dos agregaciones con sentidos opuestos: la **media** reduce la varianza, y el **máximo** dispara si *cualquier* vista ve grieta, lo que privilegia el recall.
2. **Ensemble.** Promedio de la CNN de línea base y de MobileNetV2. La motivación es el resultado de §3.4: el modelo que gana en el dataset público es el que peor generaliza. Si fallan en casos distintos, promediarlos debería batir a ambos fuera de distribución.
3. **Recalibración del umbral.** El umbral se elige **sobre el conjunto de prueba** (garantizando recall ≥ 95 %) y se aplica después a las fotos propias. Elegirlo sobre las fotos propias y medir en ellas sería circular.

#### Resultados

**Conjunto de prueba depurado (7 389 imágenes), umbral 0.50:**

| Variante | F1 | Recall | Precisión |
|---|---|---|---|
| Línea base | 0.9109 | 0.8493 | 0.9822 |
| MobileNetV2 | 0.9423 | 0.8969 | 0.9925 |
| Ensemble (media) | 0.9376 | 0.8879 | 0.9933 |
| MobileNetV2 + TTA4 máximo | 0.9473 | 0.9138 | 0.9835 |
| **MobileNetV2 + TTA8 máximo** | **0.9483** | **0.9220** | 0.9762 |
| Ensemble + TTA8 máximo | 0.9309 | 0.9291 | 0.9326 |

**Fotografías propias (40 imágenes):**

| Variante | F1 | Recall | Precisión | FN | F1 (umbral calibrado) | FN |
|---|---|---|---|---|---|---|
| Línea base | 0.7500 | 0.60 | 1.000 | 8 | 0.8000 | 6 |
| MobileNetV2 | 0.6667 | 0.50 | 1.000 | 10 | 0.6875 | 9 |
| **Ensemble (media)** | **0.8235** | 0.70 | 1.000 | 6 | 0.8000 | 6 |
| MobileNetV2 + TTA8 media | 0.7097 | 0.55 | 1.000 | 9 | 0.7097 | 9 |
| **Ensemble + TTA8 media** | 0.7879 | 0.65 | 1.000 | 7 | **0.8571** | **5** |

#### El hallazgo: la técnica que gana dentro de distribución pierde fuera

| Técnica | Efecto en `test` | Efecto en `propias` |
|---|---|---|
| TTA8 máximo sobre MobileNetV2 | **+0.0060** F1 | +0.0208 F1 |
| **Ensemble (media)** | **−0.0047** F1 | **+0.1568** F1 |

**Un equipo que hubiera elegido la técnica mirando solo el conjunto de prueba habría descartado el ensemble**, porque allí empeora ligeramente. Y el ensemble es, con diferencia, lo que mejor funciona sobre fotografías reales: sube el recall de 0.50 a 0.70, elimina 4 de los 10 falsos negativos y mantiene la precisión en 1.000.

Es la **segunda aparición del mismo patrón**. En §3.4 el modelo que ganaba en el dataset público era el que peor generalizaba; aquí ocurre lo mismo con las técnicas de agregación. Dos evidencias independientes de que **el conjunto de prueba público no sirve para tomar decisiones de despliegue** en este proyecto.

Por qué funciona el ensemble: recupera 4 grietas que MobileNetV2 perdía, aportadas por la CNN de 28 145 parámetros. Un modelo tan pequeño no tiene capacidad para memorizar texturas concretas —de hecho §3.7 mostró que era el más beneficiado por la fuga, es decir, el que más memorizaba de lo poco que podía—, y lo que aprende es más grueso y transfiere mejor. **La mejora vino de aprovechar el modelo débil, no de agrandar el fuerte.**

#### El costo decide la recomendación

| Variante | Latencia (mediana) | Pasadas | F1 en `propias` |
|---|---|---|---|
| MobileNetV2 | 170.8 ms | 1 | 0.6667 |
| **Ensemble** | **189.4 ms** (+11 %) | 2 | **0.8235** |
| MobileNetV2 + TTA4 | 692 ms (+305 %) | 4 | 0.7097 |
| MobileNetV2 + TTA8 | 1 328 ms (+678 %) | 8 | 0.7097 |
| Ensemble + TTA8 | 1 503 ms (+780 %) | 16 | 0.7879 (0.8571 calibrado) |

El ensemble cuesta **+18.6 ms, un 11 %**. El TTA8, un **678 %** para la mitad de mejora.

Un matiz de medición que conviene registrar: una primera pasada dio 183 ms para el ensemble y 187 ms para MobileNetV2 sola —es decir, el ensemble *más rápido* que uno de sus propios componentes, que es imposible—. La causa era un calentamiento insuficiente: MobileNetV2 arrastraba una desviación de ±25 ms por trazado de grafo en las primeras llamadas. Con 20 pasadas de calentamiento para todos los formatos y reportando la **mediana** en lugar de la media, el resultado se estabiliza en +11 % y es reproducible. **Una medida de latencia sin calentamiento suficiente puede invertir el orden de dos sistemas.**

Ese 11 % tiene además una lectura interesante: la línea base es el **1.9 % de los parámetros** del conjunto pero aporta el **11 % de la latencia**. Es la misma desproporción de §3.2, donde con 52 veces menos parámetros era solo 8.7 veces más rápida. **El número de parámetros es un mal predictor del tiempo de ejecución**: pesan más la profundidad, el patrón de acceso a memoria y la sobrecarga fija por invocación.

**Recomendación: ensemble simple por media, con el umbral por defecto.** Es la única de las tres técnicas cuya mejora justifica su costo: +0.157 de F1 y 4 falsos negativos menos por un 11 % de latencia. El TTA pide 7 veces más tiempo para menos de un tercio de la mejora.

`Ensemble + TTA8 media` con umbral calibrado alcanza el mejor resultado absoluto (F1 0.8571, recall 0.75, 5 falsos negativos), pero a 1.5 s por imagen queda descartado para uso interactivo. Es la opción para un procesamiento por lotes sin restricción de tiempo.

#### Limitación

Estos resultados sobre fotos propias se apoyan en **40 imágenes**: la diferencia entre 0.6667 y 0.8235 son 4 grietas más detectadas. La dirección del efecto es consistente en todas las variantes con ensemble y coherente con el mecanismo propuesto, pero la magnitud tiene un intervalo de confianza ancho (§5.7). Sobre el conjunto de prueba, con 7 389 imágenes, el efecto medido es fiable y **es negativo**: ahí el ensemble no ayuda.

---

## 4. Análisis de errores

### 4.1 Falsos negativos: el error costoso

Fuente: `reports/metricas/evaluacion.json` → `modelos.<etiqueta>.test.falsos_negativos`

La asimetría de este dominio es total y hay que decirla sin rodeos:

- Un **falso positivo** cuesta una revisión innecesaria: tiempo de un técnico, molestia, quizá un andamio.
- Un **falso negativo** es un elemento estructural dañado que **nadie va a inspeccionar**. En contexto sísmico, esa omisión puede costar vidas.

Optimizar F1 —que pondera precisión y recall por igual— asume implícitamente que ambos errores cuestan lo mismo. Aquí no es así, y por eso el proyecto:

1. Reporta el **recall de la clase positiva por separado**, no escondido dentro del F1.
2. Incluye `buscar_umbral_para_recall()`, que localiza el **umbral más alto** que aún garantiza el recall objetivo (por defecto 0.95). Se elige el más alto a propósito: se compra recall al menor precio posible en precisión, no se regala precisión sin necesidad.
3. Lista los **peores casos** —los falsos negativos con probabilidad más baja, aquellos en los que el modelo estuvo más seguro de equivocarse— para poder inspeccionarlos visualmente y entender el modo de fallo.
4. Expone el compromiso en la propia aplicación, para que la decisión sea consciente y no un efecto secundario de un valor por defecto.

**Magnitud del problema en el conjunto de prueba** (3 578 grietas reales, umbral 0.50):

| Modelo | Falsos negativos | Tasa de fuga | P(grieta) mediana en los FN | FN a menos de 0.10 del umbral |
|---|---|---|---|---|
| Línea base | 403 | 11.26 % | 0.1923 | 28 |
| MobileNetV2 | **275** | **7.69 %** | 0.1909 | 31 |
| TFLite int8 | 365 | 10.20 % | 0.1797 | 26 |

Un dato que conviene leer con cuidado: solo 26-31 de los falsos negativos están *cerca* del umbral. La gran mayoría recibe una probabilidad mediana en torno a 0.19, muy por debajo de 0.50. **No son casos dudosos: son casos en los que el modelo está razonablemente seguro y se equivoca.** Recuperarlos exige mover el umbral bastante, no un ajuste fino.

**El compromiso, dicho con precisión** (MobileNetV2, fuente `test.umbral_recall`):

Bajar el umbral de **0.50 a 0.2495** sube el recall de **0.9231 a 0.9503** y baja la precisión de **0.9943 a 0.9553**. En términos operativos, sobre las 8 721 imágenes de prueba: se detectan **97 grietas adicionales** (falsos negativos de 275 a 178) a cambio de **140 falsas alarmas más** (falsos positivos de 19 a 159).

Dicho de otro modo: **cada grieta recuperada cuesta 1.44 revisiones innecesarias**. Con la asimetría de este dominio —una revisión de más frente a un elemento dañado sin inspeccionar— esa compra es claramente razonable. Pero es una decisión de ingeniería que debe tomar una persona, no un valor por defecto de scikit-learn.

Los otros dos modelos pagan ese recall mucho más caro: la línea base necesita bajar a 0.1824 y su precisión cae a 0.8682; el TFLite int8 baja a 0.1797 y se desploma a 0.8108. **MobileNetV2 no solo tiene mejor F1: es el que compra recall al mejor precio**, que es el criterio que de verdad importa aquí. Ese argumento no se ve en la tabla de §3.1 y es el que justifica elegirlo pese a sus 52× más parámetros.

### 4.2 Modos de fallo esperados del clasificador

| Modo de fallo | Causa | Mitigación aplicada |
|---|---|---|
| Falso positivo en juntas de dilatación | El dataset no contiene negativos difíciles; una junta es una línea oscura recta | Ninguna en el modelo. Requiere ampliar el dataset con negativos difíciles |
| Falso positivo en sombras duras | El dataset tiene iluminación uniforme | Aumento de brillo y contraste (parcial: las sombras son geométricas, no fotométricas) |
| Falso negativo en superficies pintadas | Bajo contraste entre la fisura y el fondo | Aumento de contraste (parcial) |
| Falso negativo por distancia excesiva | La grieta ocupa pocos píxeles tras redimensionar a 160×160 | Protocolo de captura documentado; no hay solución algorítmica sin más resolución |
| Fallo en ladrillo a la vista o pañete | Fuera de la distribución de entrenamiento | Ninguna. Es una limitación honesta del dataset |

### 4.3 Modos de fallo de la inclinometría

| Modo de fallo | Causa | Mitigación aplicada |
|---|---|---|
| Ángulo arrastrado por líneas espurias | Ventanas, cables, marcos, el borde de la propia foto | **Mediana ponderada por longitud** en vez de promedio simple, más un refinamiento que descarta lo incoherente con la estimación inicial |
| Suelos y dinteles tomados por elementos verticales | Hough no distingue semántica | Filtro de verticalidad: se descarta todo lo que se aleje más de 35° de la vertical |
| «No se detectó nada» | Pared lisa, foto desenfocada, contraluz | Se devuelve `fiable=False` con un mensaje explicativo. **No se lanza excepción**: en la aplicación es un resultado válido, no un error |
| Sesgo por perspectiva | La foto no se tomó perpendicular a la superficie | **Sin mitigación.** Es la limitación fundamental del método (§5.2) |
| Envolvimiento angular al promediar | 179° y 1° son casi la misma dirección | Estadística **axial** (duplicar el ángulo, promediar como vectores, dividir entre dos) en la clasificación de orientación |
| **La orientación detectada es la del elemento, no la de la grieta** | Hough no distingue semánticamente una fisura de la arista de una columna | Documentado en la interfaz y en §4.4. Sin solución dentro de este enfoque |
| **La grieta tomada como eje del elemento** (reverso del anterior) | En un primer plano de pared sin aristas, la única recta casi vertical es la propia fisura | **Tres guardarraíles de verosimilitud** (ángulo máximo, dispersión y extensión vertical) que descartan la medida; más auditoría visual del operador (§4.5). Era el modo de fallo más peligroso: disparaba R5 y producía un riesgo Alto falso |

### 4.4 La orientación de la grieta se confunde con la del elemento

Detectada durante la validación de extremo a extremo, y conviene explicarla porque es una limitación **de diseño**, no un fallo de implementación.

`clasificar_orientacion_grieta()` calcula la dirección dominante de todos los segmentos que devuelve Hough. En un recorte de la zona agrietada, esos segmentos son la grieta y el resultado es correcto. Pero en una fotografía de **una columna completa**, los segmentos más largos y numerosos son las **aristas verticales de la propia columna**, no la fisura: la orientación devuelta será «vertical» aunque la grieta sea claramente diagonal.

En una prueba controlada con una columna sintética inclinada 2.5° y una grieta diagonal dibujada explícitamente, el clasificador devolvió `vertical (88.0°)` — la orientación de los bordes de la columna, exactamente como predice el análisis.

La causa de raíz es que **Hough no tiene semántica**: encuentra rectas, no sabe cuáles son grietas. Distinguirlas exigiría segmentar la fisura primero (véase §8, punto 3), que es un cambio de enfoque, no un ajuste de parámetros.

Mitigaciones aplicadas:

- La aplicación advierte explícitamente de este comportamiento junto a la métrica de orientación y recomienda encuadrar de cerca la zona agrietada.
- El motor de reglas solo usa la orientación **cuando ya hay grieta detectada** (`R3`/`R4` exigen `hay_grieta`), lo que limita el daño: sobre una pared sana la orientación no influye en el nivel.
- El efecto es en general **conservador**: «vertical» está catalogada como no grave en muros, de modo que el fallo típico rebaja la regla `R3` a `R4` en vez de generar una alarma falsa. Pero en una viga, donde «vertical» sí es grave, podría ir en la dirección contraria. No es una mitigación completa y no se presenta como tal.

### 4.5 El caso simétrico: la grieta tomada como eje del elemento

Detectado durante el uso de la aplicación, ajustando los controles de sensibilidad sobre una fotografía propia. Es el **reverso exacto de §4.4** y, a diferencia de aquel, produce una **falsa alarma de riesgo Alto**.

#### El caso observado

Fotografía en primer plano de un muro liso pintado, con una fisura diagonal sinuosa. No hay ninguna columna, arista ni elemento vertical en el encuadre: solo superficie de pared.

Con los controles en posición permisiva (Canny 20/85, longitud mínima 10 px, votos 35, separación 25), el módulo devolvió:

> **Desaplome: +34.90° | FIABLE (25 líneas)**

y dibujó las líneas verdes **sobre la propia grieta**. El sistema midió la inclinación de la fisura y la reportó como desviación del elemento respecto a la vertical.

#### Por qué es más grave que §4.4

La cadena de consecuencias es directa y automática:

| Paso | Valor | Efecto |
|---|---|---|
| `confianza = 25` | ≥ `min_confianza_inclinacion` (3) | La medida se acepta como fiable |
| `angulo = 34.90°` | ≥ `desaplome_severo` (2.0°) | **Dispara R5, «Desaplome severo»** |
| Severidad de R5 | ALTA | **Nivel final: Riesgo Alto** |

El sistema declararía riesgo Alto sobre un muro que está perfectamente a plomo, y lo justificaría citando el límite de deriva de la NSR-10. En §4.4 el fallo era mayoritariamente conservador; **aquí es una falsa alarma con apariencia de dictamen fundamentado**, que es peor.

Un detalle que ilustra lo frágil de la frontera: **34.90° queda a una décima de grado de los 35° de `tolerancia_vertical_grados`**. Si la fisura hubiera sido medio grado más tendida, el filtro la habría rechazado y el módulo habría respondido correctamente «no se detectó ningún elemento vertical». El resultado erróneo dependió de una décima de grado.

#### Por qué ningún ajuste de parámetros lo resuelve

Se barrieron **40 combinaciones** de Canny (4 valores), longitud mínima (5) y votos (2) sobre las 40 fotografías propias, replicando el redimensionado a 1024 px que aplica la aplicación. Se contabilizó como **medida implausible** todo desaplome mayor de 10°: una edificación en pie está a pocos grados de plomo, de modo que un ángulo así solo puede venir de medir otra cosa.

| Canny | Long. mín. | Votos | Cobertura | \|ángulo\| mediano | Medidas > 10° | Dispersión |
|---|---|---|---|---|---|---|
| 30/100 | 10 | 35 | 6/40 | 21.2° | **5** | 1.50 |
| 50/150 | 10 | 35 | 5/40 | 23.2° | **4** | 1.34 |
| 50/150 | 40 | 35 | 3/40 | 18.6° | **2** | 0.26 |
| 50/150 | 40 | 60 | 2/40 | 17.2° | **1** | 0.51 |
| 50/150 | 80 | 60 | 1/40 | 0.0° | **0** | 0.61 |
| **50/150** | **120** | **60** | 1/40 | 0.0° | **0** | **0.00** |
| 50/150 | 180 | 60 | 1/40 | 0.0° | **0** | 0.00 |

**Cada fotografía adicional que se gana relajando los parámetros es una medida falsa.** Al pasar de longitud mínima 120 a 10, la cobertura sube de 1 a 6 imágenes y **5 de esas 6 reportan ángulos de 16° a 23°**. No se gana sensibilidad: se fabrica ruido con formato de resultado.

Dos observaciones más del barrido:

- **Los umbrales de Canny apenas influyen.** Las cuatro configuraciones probadas (20/85 a 70/200) dan el mismo patrón. El parámetro que gobierna la fiabilidad es la **longitud mínima de línea**, porque es el único que separa la arista estructural (segmentos de cientos de píxeles) de la fisura sinuosa (segmentos de decenas).
- Con longitud mínima ≥ 120 la **dispersión cae a 0.00°**: los segmentos aceptados coinciden entre sí de forma exacta. Es la firma de estar midiendo una recta real y no un conjunto de tramos con orientaciones dispares.

#### Conclusión y mitigación

La causa raíz es la misma que en §4.4 —**la transformada de Hough encuentra rectas, no sabe qué representa cada una**— manifestándose en la dirección contraria. Confirma que el problema no es de calibración sino de **ausencia de semántica**, y que su solución real es segmentar la fisura antes de medir (§8, punto 3).

#### Mitigación implementada: tres guardarraíles de verosimilitud

A raíz de este hallazgo se añadieron a `estimar_inclinacion()` tres filtros que **descartan la medida en lugar de emitir un número sin sentido**. Ninguno corrige el ángulo: marcan `fiable=False` con su motivo, que es lo que el motor de reglas necesita para no disparar R5 ni R6. Los tres umbrales viven en `config.yaml`.

| Guardarraíl | Umbral | Qué detecta |
|---|---|---|
| **Ángulo máximo plausible** | 10° | Una edificación en pie está a pocos grados de plomo; 10° equivalen a un 17.6 % de deriva. Más que eso solo puede venir de medir otra cosa |
| **Dispersión máxima** | 2.0° (MAD) | Una arista real da segmentos que coinciden (dispersión ~0); una grieta sinuosa da tramos dispares. Medido: 0.00–0.61 midiendo aristas, 1.34–1.50 midiendo grietas |
| **Extensión vertical mínima** | 30 % del alto | El eje de una columna recorre casi todo el encuadre; un grupo de segmentos cortos y agrupados, no |

**Efecto medido sobre las 40 fotografías propias**, con los mismos controles permisivos que produjeron el caso original (Canny 20/85, longitud 10, votos 35):

| | Antes | Después |
|---|---|---|
| Medidas declaradas fiables | 6 | **1** |
| De ellas, implausibles (> 10°) | 5 | **0** |

Sobrevive exactamente la medida legítima. Los tres casos rechazados que se inspeccionaron fueron **+34.90°** (el original, por ángulo inverosímil), **+17.82°** (por lo mismo) y **+3.28° con dispersión ±3.28°** (por incoherencia entre segmentos) — este último es interesante porque el ángulo sí era plausible y aun así la medida no se sostenía.

**Un conflicto que hubo que resolver.** Los guardarraíles rompían la propia validación del módulo: `validar_con_rotaciones()` gira las fotos ±10° a propósito, generando desaplomes que en una foto de inspección se rechazarían. Aplicarlos allí habría dejado §3.6 sin datos. La solución es el parámetro `aplicar_guardarrailes`, que la validación desactiva de forma explícita y documentada: **el guardarraíl codifica una suposición sobre la escena, no sobre la corrección del algoritmo**, y confundir ambas cosas habría impedido medir el error. Hay una prueba de regresión que lo fija.

Mitigaciones que siguen dependiendo del operador:

1. **Criterio de lectura.** Antes de aceptar un desaplome hay que mirar la imagen procesada y comprobar si las líneas verdes caen sobre el borde del elemento o sobre la grieta. Si caen sobre la grieta, el número no significa nada. Este es el propósito real de la comparación lado a lado de la interfaz: permitir la **auditoría visual de la decisión del algoritmo**. La aplicación muestra ahora un aviso destacado con el motivo concreto cuando un guardarraíl se activa.
2. **Los guardarraíles reducen el daño, no eliminan la causa.** Una grieta que resultara estar a menos de 10° de la vertical, con tramos coherentes y recorriendo el encuadre, seguiría midiéndose como desaplome. La solución de fondo sigue siendo segmentar la fisura antes de medir (§8, punto 4).

**Este hallazgo apareció usando la aplicación, no ejecutando pruebas.** Es un argumento a favor de haber construido una interfaz que muestra el trabajo intermedio del algoritmo en lugar de solo su conclusión: un panel que hubiera mostrado únicamente «Riesgo Alto · desaplome 34.90°» habría ocultado el error por completo.

### 4.6 El error no era del modelo: era nuestro, y estaba en el redimensionado

Esta sección corrige un diagnóstico anterior de esta misma memoria. Es el hallazgo más importante del proyecto y también el más incómodo, porque durante varias fases se atribuyó a la naturaleza del problema un fallo que habíamos introducido nosotros.

#### El diagnóstico que dimos por bueno

El modelo alcanza F1 0.9423 sobre el conjunto de prueba depurado y colapsaba a 0.6667 sobre nuestras propias fotografías. La §6 lo explicaba como **desplazamiento de dominio**: otro material, otra iluminación, otra cámara, otro país. La explicación era plausible, encajaba con la literatura y estaba respaldada por los sesgos reales del dataset. Su problema es que nunca se puso a prueba: se aceptó porque sonaba razonable.

De ella se derivaron dos líneas de trabajo, ambas costosas: recolectar más fotografías propias y recalibrar el umbral por dominio.

#### La medición que lo desmonta

Antes de recolectar nada, medimos algo que no habíamos mirado nunca: **el tamaño de las imágenes**.

| Origen | n | Tamaño mediano | Reducción hasta 160 px |
|---|---|---|---|
| Parches de entrenamiento | 400 | 227 × 227 | **1.4×** |
| Fotografías propias | 60 | 1200 × 1600 | **7.5×** |

Una fisura de 3 px de ancho en la imagen original llega al modelo así:

- desde un parche de entrenamiento: 3 / 1.4 = **2.1 px** — sobrevive;
- desde una foto de teléfono: 3 / 7.5 = **0.4 px** — no sobrevive.

Por debajo de un píxel, la interpolación bilineal promedia la fisura con la pared que la rodea hasta hacerla indistinguible del ruido de textura. **La grieta no se le escapa al modelo: la borramos antes de enseñársela.**

Y hay un detalle que confirma que el fallo era de procedimiento y no de dominio: las 227 × 227 del entrenamiento son parches *recortados* de fotografías grandes, no fotografías reducidas. El dataset nunca contuvo una imagen que hubiera sufrido una reducción de 7.5×. Estábamos evaluando el modelo en un régimen de escala que jamás había visto, y llamando a eso «otro dominio».

#### La corrección: analizar por mosaicos

Si la hipótesis es correcta, trocear la fotografía en ventanas y evaluarlas por separado debe recuperar el recall **sin reentrenar nada**. Es una predicción falsable: si el problema fuese el material o la iluminación, el troceado no cambiaría el resultado.

Medido con `scripts/evaluar_mosaicos.py` sobre las 60 fotografías propias, con el ensemble y umbral 0.5:

| Estrategia | F1 | Recall | Precisión | FN | FP | Ventanas | s/foto |
|---|---|---|---|---|---|---|---|
| Imagen entera | 0.8889 | 0.80 | 1.000 | **6** | 0 | 1 | 0.32 |
| Mosaicos 227 px | 0.9180 | 0.93 | 0.903 | 2 | 3 | 140 | 2.27 |
| Mosaicos 320 px | 0.9355 | 0.97 | 0.906 | 1 | 3 | 63 | 1.33 |
| **Mosaicos 480 px** | **0.9508** | **0.97** | 0.935 | **1** | 2 | 24 | 0.78 |
| Mosaicos 640 px | 0.9508 | 0.97 | 0.935 | 1 | 2 | 12 | 0.55 |

La predicción se cumple: **el recall pasa de 0.80 a 0.97** y los falsos negativos de 6 a 1, sin tocar los pesos. El F1 supera además el 0.8889 que alcanzaba la recalibración del umbral con el óptimo elegido *a posteriori* sobre las propias fotos (§3.8), es decir, bate a una cota que ya era optimista.

#### El óptimo no está donde la teoría ingenua lo pondría

El resultado interesante no es que el troceado funcione, sino **dónde deja de funcionar**. Si la única causa fuese la escala, el mejor mosaico sería el de 227 px, que reproduce exactamente el tamaño de los parches de entrenamiento. Es la peor de las cuatro filas de troceado.

Hay dos efectos en sentidos opuestos:

1. **Mosaicos grandes** → reducción excesiva → la fisura desaparece (el caso extremo es la imagen entera);
2. **Mosaicos pequeños** → la fisura pierde su contexto. Una grieta se reconoce por cómo interrumpe la superficie que la rodea; un recorte de 227 px de una foto de 1600 px muestra una franja de pared con una línea, sin la continuidad que la identifica. Además multiplica por seis el número de ventanas y, con agregación por máximo, **por seis las ocasiones de dar una falsa alarma**: los falsos positivos suben de 2 a 3 mientras el recall baja.

El óptimo es interior, en 480–640 px, y hay que encontrarlo midiendo. Se eligió 480 px: empata con 640 en métricas y falla en las mismas tres fotos, pero deja el único falso negativo en p = 0.222 en lugar de p = 0.071 —tres veces más cerca de detectarse— y con 24 ventanas en vez de 12 localiza la grieta con el doble de resolución.

#### El precio, dicho sin adornos

La precisión baja de **1.000 a 0.935**: aparecen 2 falsas alarmas donde antes no había ninguna. Es aritmética directa de la agregación por máximo: 24 ventanas son 24 oportunidades de equivocarse, frente a una sola. Y el coste por fotografía sube de 0.32 s a 0.78 s, un factor 2.4× para 24 veces más inferencias —la diferencia la absorbe el procesamiento por lotes, que amortiza el coste fijo de cada llamada al modelo.

Ambos precios son los que §4.1 argumenta que hay que pagar: en evaluación estructural, una falsa alarma cuesta una inspección; un falso negativo puede costar una vida.

#### Un beneficio no buscado: localización

Analizar por ventanas responde a una pregunta que la inferencia sobre la imagen entera no puede responder: **dónde**. La aplicación dibuja las ventanas que superan el umbral y resalta la más alta. Para un inspector, la diferencia entre «hay grieta en esta foto» y «hay grieta aquí» es la diferencia entre un aviso y una indicación accionable. Además ofrece una comprobación de coherencia gratuita: la evidencia concentrada en ventanas contiguas es más creíble que la dispersa, que sugiere ruido.

#### Lo que este hallazgo obliga a corregir

- La §6 mantenía «Escala: recortes a distancia aproximadamente constante → degradación con encuadres amplios» como un sesgo del *dataset*. Es más que eso: es un sesgo que **nuestro preprocesado convertía en un fallo evitable**. La fila queda corregida.
- La recomendación de recolectar más fotografías propias pierde prioridad. El modelo no necesitaba más ejemplos; necesitaba verlos a la escala correcta.
- La recalibración del umbral por dominio (§3.8) pasa de ser el arreglo principal a un ajuste fino: con mosaicos, el recall ya es 0.97 en el umbral por defecto.

#### Consecuencia inesperada: el ensemble casi deja de hacer falta

Al repetir la medición con MobileNetV2 sola aparece un resultado que reordena una decisión anterior del proyecto. Las cuatro celdas, sobre las mismas 60 fotografías:

| | Imagen entera | Mosaicos 480 px | Ganancia del troceado |
|---|---|---|---|
| **MobileNetV2 sola** | F1 0.8000 · recall 0.67 · 10 FN | F1 0.9355 · recall 0.97 · 1 FN | **+0.1355** |
| **Ensemble** | F1 0.8889 · recall 0.80 · 6 FN | F1 0.9508 · recall 0.97 · 1 FN | +0.0619 |
| **Ventaja del ensemble** | **+0.0889** | **+0.0153** | |

El troceado ayuda **más al modelo solo que al ensemble**, y el motivo es coherente con todo lo anterior: la ventaja del ensemble consistía en gran parte en **compensar un error que ahora está corregido en su origen**. Promediar dos modelos rescataba algunas de las grietas que el redimensionado había casi borrado; cuando la grieta llega íntegra, hay mucho menos que rescatar.

Corregida la escala, el ensemble aporta **0.0153 de F1 —un solo falso positivo de sesenta— a cambio de un 26 % más de tiempo** (0.78 s frente a 0.62 s por fotografía). En la decisión original (§3.8) el ensemble compraba 0.0889 de F1 por un 11 % de latencia, y era una compra evidente. Ahora la relación es mucho peor, y con `n = 60` esa diferencia de un solo caso no es distinguible del ruido.

Esto no invalida el ensemble, pero sí obliga a revisar por qué está: **se mantiene por su comportamiento en el modo de vídeo**, donde no hay troceado y la ventaja de +0.0889 sigue vigente. Para el análisis de fotografías, MobileNetV2 sola con mosaicos es la opción defendible, y la aplicación permite elegirla.

Es también una advertencia general sobre cómo se acumulan las mejoras en un proyecto: una técnica que se justificó midiendo contra una línea base defectuosa puede dejar de estar justificada cuando el defecto se arregla. La ganancia de +0.0889 nunca fue del ensemble; era del error que compensaba.

#### La lección metodológica

Es la tercera vez en este proyecto que un fallo atribuido al modelo resultó estar en cómo le entregábamos los datos: el barajado que producía lotes de una sola clase, la fuga de datos entre particiones y ahora la escala del redimensionado. Las tres veces la explicación sofisticada —el modelo no converge, el dominio es distinto— era más cómoda que la simple, y las tres veces era falsa.

El patrón que las une: **medir la entrada antes de culpar al modelo**. Ninguno de los tres fallos requirió técnicas avanzadas para encontrarse; los tres requirieron mirar una estadística elemental de los datos que nadie había mirado. En este caso, la mediana de dos números que llevaban meses en el disco.


### 4.7 Calibrar el umbral: un resultado negativo, medido en serio

Con el análisis por mosaicos funcionando, quedaba pendiente la última mejora «gratuita» del plan: mover el umbral de decisión. La §3.8 ya lo había intentado, pero eligiendo el umbral sobre las mismas fotos en las que después reportaba el resultado —circular, y allí se etiquetó como cota superior—. Aquí se hace bien, con `scripts/calibrar_umbral.py`.

#### El protocolo

**Validación cruzada estratificada de 5 pliegues.** La regla se elige en 4 pliegues y se aplica al quinto, que no participó en la elección. Las predicciones fuera de pliegue se acumulan y se miden juntas. Es la diferencia entre *«existe una regla que funciona en estas 60 fotos»* y *«elegir la regla así funcionará en la foto 61»*.

Se compararon dos familias de reglas, porque la agregación por máximo deja dos formas distintas de decidir:

1. **Umbral sobre el máximo** de las 24 ventanas — lo habitual.
2. **Fracción mínima de ventanas** que ven grieta — exigir que la fisura aparezca en varias ventanas, no en una sola. A priori parecía la más robusta: cuenta evidencias en lugar de fiarse de un único valor extremo, y se explica sin hablar de probabilidades.

#### Resultado

| Regla | F1 | Recall | Precisión | FN | FP | Corte |
|---|---|---|---|---|---|---|
| **Umbral fijo 0.5 (actual)** | 0.9355 | 0.97 | 0.906 | 1 | 3 | 0.5000 |
| CV · umbral sobre el máximo | 0.9508 | 0.97 | 0.935 | 1 | 2 | 0.9977 |
| CV · fracción de ventanas | **0.8814** | 0.87 | 0.897 | 4 | 3 | 0.1250 |
| *Oráculo sobre el máximo* | *0.9831* | *0.97* | *1.000* | *1* | *0* | *0.9977* |
| *Oráculo sobre la fracción* | *0.9508* | *0.97* | *0.935* | *1* | *2* | *0.1250* |

Dos sorpresas, y ninguna en la dirección esperada.

#### Sorpresa 1: el umbral óptimo es 0.998, no un valor bajo

Todo el proyecto venía razonando que bajar el umbral aumenta el recall (§4.1). Con mosaicos ocurre lo contrario: el corte útil está **altísimo**. El motivo está en la distribución que produce la agregación por máximo:

| | mín. | p25 | mediana | máx. |
|---|---|---|---|---|
| Fotos **sin** grieta | 0.0131 | 0.0196 | 0.0253 | 0.9972 |
| Fotos **con** grieta | 0.3466 | 1.0000 | 1.0000 | 1.0000 |

**19 de las 30 fotos con grieta dan exactamente 1.0.** El máximo de 24 ventanas satura: basta con que una ventana esté segura para que el resultado se pegue al techo. La decisión ya no se toma entre «poco probable» y «muy probable», sino entre «saturado» y «no saturado», y ese límite vive en las milésimas superiores.

#### Sorpresa 2: ese umbral es una casualidad, no un aprendizaje

El oráculo alcanza F1 0.9831 con 0 falsos positivos, que sería el mejor número de toda la memoria. No se puede usar, y la razón la da la propia herramienta:

```
Margen de separacion del corte del oraculo
  maximo     al negativo mas alto 0.0006 · al positivo mas bajo 0.0006
  fraccion   al negativo mas alto 0.0417 · al positivo mas bajo 0.0417
```

El corte de 0.9977 pasa a **seis diezmilésimas** de la foto sana peor clasificada (0.99716). No ha encontrado una frontera entre dos clases: ha encontrado un hueco entre dos fotografías concretas de esta muestra. Cualquier fotografía nueva con una junta algo más marcada cae del otro lado. Un modelo que separa por 0.0006 no separa.

La validación cruzada lo confirma por otra vía: 4 de los 5 pliegues eligen ~0.9977 y el quinto se va a 0.2124. Ese pliegue no está roto — está empatado: con una distribución tan bimodal, el F1 tiene dos óptimos casi idénticos, uno que lo acepta casi todo y otro que rechaza las falsas alarmas. El criterio no distingue entre extremos opuestos.

#### Sorpresa 3: la regla «robusta» es la peor

La fracción de ventanas parecía la opción sensata, y su oráculo confirma que hay un valor bueno (0.9508). Pero **fuera de pliegue rinde 0.8814, peor que no calibrar nada**, y solo 2 de 5 pliegues coinciden en el corte. Al mirar los datos se ve por qué no puede funcionar:

- fotos sanas: 27 de 30 encienden **0** ventanas; las otras tres encienden 2, 4 y 4;
- fotos con grieta: la mayoría encienden entre 9 y 16, pero cuatro encienden 0, 1, 4 y 4.

Las dos clases **se solapan justo en la zona del corte**. No hay ningún número de ventanas que las separe, así que el valor elegido depende de qué fotos toquen en cada pliegue. Su margen de 0.0417 es setenta veces mayor que el del máximo y aun así no basta: un margen amplio en una frontera que atraviesa el solapamiento no sirve de nada.

#### Conclusión: no se calibra

**Se mantiene el umbral fijo en 0.5.** La ganancia cruzada era +0.0153 de F1 —un falso positivo de sesenta— sostenida sobre un margen de 0.0006, y la alternativa que parecía más sólida resultó peor que no hacer nada.

Es un resultado negativo, y se documenta con el mismo detalle que uno positivo por dos razones. La primera es que **no adoptar una mejora aparente es en sí una decisión de ingeniería**, y sin la medición no habría forma de defenderla frente a quien viera el 0.9831 del oráculo y preguntara por qué no se usa. La segunda es que esa cifra existe, es reproducible y alguien podría reportarla de buena fe: la diferencia entre 0.9831 y 0.9355 no está en el modelo, está en si el umbral se eligió mirando o no las fotos en las que se mide.

Queda además una observación de fondo para §5: que 19 de 30 fotos den exactamente 1.0 significa que **las probabilidades del modelo no están calibradas**. Se pueden usar para ordenar, no para leerlas como grados de confianza. La interfaz muestra «Probabilidad de grieta: 100 %», y ese número no debe interpretarse como certeza — solo como «muy por encima del umbral».


### 4.8 La decisión final: un modelo por modo, y ningún selector

Hasta esta fase la aplicación ofrecía un selector con tres modelos —ensemble, MobileNetV2 y TFLite int8— y dejaba la elección al usuario. Las mediciones de §4.6 y §4.7 permiten cerrar esa decisión, y el resultado es que **el selector desaparece**.

#### Por qué un selector era la respuesta equivocada

Ofrecer la elección parecía flexible, pero trasladaba al usuario una decisión que él no puede tomar bien: para elegir entre tres modelos hay que conocer sus curvas de recall y latencia, que están en esta memoria y no en la pantalla. Un inspector que abre la aplicación no tiene forma de saber que TFLite pierde 0.10 de recall. La flexibilidad aparente era, en la práctica, una forma de no comprometerse.

Y sobre todo: **la respuesta está medida**. No depende del gusto de nadie.

#### Las tres celdas, sobre las 60 fotografías propias

Con mosaicos de 480 px, umbral 0.5:

| Modelo | F1 | Recall | Precisión | FN | FP | s/foto | ms/fotograma |
|---|---|---|---|---|---|---|---|
| **MobileNetV2** | 0.9355 | 0.97 | 0.906 | 1 | 3 | 0.62 | 170.8 |
| Ensemble | 0.9508 | 0.97 | 0.935 | 1 | 2 | 0.78 | 189.4 |
| **TFLite int8** | 0.9123 | 0.87 | 0.963 | 4 | 1 | 0.28 | 4.9 |

#### La decisión, y sus dos criterios opuestos

Fotografía y vídeo imponen restricciones contrarias, y por eso ningún modelo único puede ser el correcto para ambos:

- **Fotografía → MobileNetV2 + mosaicos.** Aquí se admite medio segundo de cálculo, así que manda el acierto. Se descarta TFLite porque pierde 0.10 de recall, es decir, tres grietas reales más sin detectar: exactamente el error que §4.1 argumenta que no hay que cometer. Y se descarta el ensemble porque su ventaja es +0.0153 de F1 —**un falso positivo de sesenta**— por un 26 % más de tiempo y un segundo modelo en memoria. Con n = 60 esa diferencia no es distinguible del ruido.
- **Vídeo → TFLite int8.** Aquí hay 33 ms por fotograma para sostener 30 FPS y ninguno de los otros cabe: MobileNetV2 daría 6 FPS y el ensemble 5. Su menor recall por fotograma se compensa en parte con el suavizado temporal por mediana, que agrega varias observaciones de la misma escena — un lujo que la fotografía única no tiene.

Es la misma pieza de razonamiento que aparece en §4.6 sobre el tamaño del mosaico: no hay un óptimo global, hay un óptimo por régimen, y encontrarlo exige medir en cada uno.

#### Lo que se conserva y por qué

El ensemble **no se borra del código**. Sigue disponible en `config.yaml` y en el comparador de latencias de la aplicación, porque una decisión documentada debe poder reproducirse: quien lea que el ensemble aporta +0.0153 tiene que poder comprobarlo. Lo que se elimina es la *pregunta al usuario*, no la *capacidad de medir*.

La aplicación indica en todo momento qué modelo está usando y por qué, y si el artefacto esperado no está en disco repliega a otro y lo advierte, en lugar de fallar en silencio. El orden de repliegue también es distinto en cada modo, por la misma lógica: sin MobileNetV2, la fotografía prefiere el ensemble (recall 0.97) antes que TFLite (0.87); sin TFLite, el vídeo prefiere MobileNetV2 antes que el ensemble, porque este último solo añadiría latencia a un vídeo que ya va lento.

#### El patrón que cierra las tres secciones

§4.6, §4.7 y §4.8 comparten una forma. En las tres, la conclusión salió de comparar contra una línea base recalculada sobre la misma muestra, y en las tres esa comparación desactivó algo que parecía establecido:

- el troceado reveló que la brecha de dominio era en buena parte un error de escala nuestro;
- la calibración honesta reveló que un F1 de 0.9831 descansaba sobre 0.0006 de margen;
- las dos juntas revelaron que la ventaja del ensemble era el error que compensaba.

Ninguna de las tres necesitó una técnica avanzada. Las tres necesitaron medir la alternativa aburrida antes de aceptar la interesante.


### 4.9 Degradación de escala en entrenamiento: la hipótesis que no se sostuvo

§4.6 corrige el desajuste de escala en **inferencia**, troceando la fotografía, y eso cuesta 2.4× de tiempo. La continuación natural era atacarlo en **entrenamiento**: si el modelo aprende con grietas ya degradadas por una reducción fuerte, debería reconocerlas sin necesidad de trocear, y el coste se recuperaría.

Se implementó (`construir_capa_degradacion`: reduce la imagen por un factor aleatorio de hasta 4× y la restaura, en el 50 % de los lotes) y se puso a prueba. **No funcionó**, y merece la pena contar cómo se comprobó, porque el resultado no era evidente de antemano.

#### El diseño del experimento

Dos entrenamientos **idénticos salvo en la variable de estudio**: misma semilla, mismo 25 % del dataset, mismas 3 + 2 épocas, misma configuración. Lo único distinto es la degradación.

Se eligió deliberadamente una escala reducida —25 % de los datos, 5 épocas, unos 10 minutos por modelo— antes de comprometer las 2 h 10 min del entrenamiento completo. Una comprobación barata que puede descartar la hipótesis vale más que una cara que la confirma tarde.

#### Resultado 1: no cambia nada dentro de distribución

| | Val. accuracy | Val. recall | Val. precisión | Val. F1 | s/época |
|---|---|---|---|---|---|
| Control | 0.9656 | 0.9251 | 0.9904 | 0.9566 | 99.4 |
| Degradado | 0.9674 | 0.9239 | 0.9964 | 0.9588 | 99.9 |

Esto era lo primero que había que descartar: que la degradación mejorase en fotos reales a costa de hundirse en el dataset público habría significado cambiar un sesgo por el contrario. No ocurre — y el sobrecoste de la capa es del 0.5 %, despreciable.

#### Resultado 2: tampoco cambia nada fuera de distribución

Sobre las 60 fotografías propias:

| Estrategia | Control | Degradado |
|---|---|---|
| **Imagen entera** | F1 0.8519 · recall 0.77 · 7 FN · 1 FP | F1 0.8462 · recall 0.73 · 8 FN · 0 FP |
| **Mosaicos 480 px** | F1 0.9524 · recall 1.00 · 0 FN · 3 FP | F1 0.9524 · recall 1.00 · 0 FN · 3 FP |

La fila de la imagen entera es donde debía notarse el efecto, y ahí el modelo degradado queda **ligeramente peor**, no mejor.

#### Por qué esa diferencia no significa nada, y cómo se comprueba

Un F1 de 0.8519 frente a 0.8462 invita a concluir que la degradación perjudica. Sería tan indebido como concluir que ayuda: la comparación correcta no es entre dos números agregados, sino **foto a foto**.

```
imagen_entera
  fallan ambos modelos:                    7
  solo falla el control (gana degradado):  1
  solo falla el degradado (gana control):  1
  McNemar exacto sobre 2 discordancias:    p = 1.000

mosaicos_480px
  ninguna discordancia: los dos modelos aciertan y fallan
  exactamente en las mismas fotos
```

De 60 fotografías, los dos modelos **discrepan en dos**, una a favor de cada uno. Con mosaicos no discrepan en ninguna: emiten predicciones idénticas en las 60. La diferencia de F1 procede íntegramente de que una fotografía cambió de lado en cada dirección.

La prueba de McNemar da p = 1.000, que es literalmente el resultado menos significativo posible. **La degradación de escala no tiene ningún efecto medible**, ni bueno ni malo.

#### Decisión y su alcance

**No se ejecuta el entrenamiento completo.** La comprobación de 20 minutos hizo exactamente su trabajo: ahorró 2 h 10 min de CPU que no habrían cambiado el resultado. El troceado sigue siendo necesario y su coste de 2.4× no se recupera.

Los límites de esta conclusión, dichos con precisión:

- Se probó **un** punto del espacio de diseño (factor 4.0, probabilidad 0.5). Factores mayores o degradación aplicada siempre podrían comportarse de otro modo.
- Se probó con **5 épocas sobre el 25 % de los datos**. Los aumentos de datos a menudo tardan en rendir, porque añaden dificultad antes de añadir robustez. Es concebible que a 15 épocas sobre el dataset completo aparezca un efecto que aquí no se ve.

Lo que sí puede afirmarse es lo que decide la cuestión práctica: **con el presupuesto disponible, esta vía no compite con el troceado**, que da +0.10 de recall de forma inmediata y sin entrenar nada.

#### Un hallazgo lateral que sí merece seguimiento

Ambos modelos del experimento —entrenados con la **cuarta parte** de los datos y **un tercio** de las épocas— alcanzan sobre las fotos propias con mosaicos **F1 0.9524 y recall 1.00 (0 falsos negativos)**, frente al 0.9355 y recall 0.97 del modelo de producción, que vio cuatro veces más datos durante quince épocas.

Es una diferencia de una sola fotografía y no debe leerse como que entrenar menos sea mejor. Pero apunta en la misma dirección que §3.4 y §3.8: **entrenar más ajusta mejor el dominio público y no necesariamente el real**. Queda anotado en §8 como línea de trabajo, con una advertencia: comprobarlo exigiría un criterio de parada que mire a las fotos propias, y usarlas para decidir cuándo parar las convertiría en conjunto de validación — perdiendo la única medida honesta de generalización que tiene el proyecto. El experimento tendría que diseñarse con mucho cuidado.


---

## 5. Limitaciones

### 5.1 No se puede medir el ancho real de la grieta

Es la limitación más grave, y conviene declararla antes que ninguna otra porque afecta a la utilidad del sistema entero.

Sin una **referencia métrica** en la escena —una regla, una moneda, un objeto de dimensión conocida—, la escala de la imagen es desconocida. Una fisura que ocupa 3 píxeles puede tener 0.1 mm o 5 mm de ancho según la distancia a la que se tomó la foto.

El problema es que **el ancho de fisura es exactamente el criterio que usa la NSR-10** para clasificar la severidad del daño. El sistema puede decir «hay una grieta diagonal en esta columna», que es información valiosa para priorizar; **no** puede decir «esta grieta tiene 0.4 mm y por tanto está dentro de lo admisible», que es lo que un ingeniero necesita para dictaminar.

Esta brecha es estructural, no un defecto de implementación. Resolverla exigiría fotogrametría con referencia conocida o un sensor de profundidad, y queda fuera del alcance del proyecto.

### 5.2 La perspectiva sesga la medida del ángulo

El desaplome se mide **en el plano de la imagen**. Si la fotografía no se toma perpendicular a la superficie, la proyección introduce un error sistemático: una columna perfectamente vertical fotografiada desde abajo y de lado aparenta inclinación.

El error crece con el ángulo de la toma y no es aleatorio, así que **promediar varias fotos malas no lo cancela**. La única mitigación real es el protocolo de captura, que está documentado en el README y en el notebook de EDA, y que la propia aplicación recuerda al usuario.

### 5.3 Generalización a superficies no vistas

El modelo solo puede reconocer lo que se parece a lo que vio. El dataset está dominado por hormigón liso; el ladrillo a la vista, el pañete, la piedra, la madera y el bahareque están fuera de su distribución.

Esto tiene una dimensión que no es técnica sino social, y se desarrolla en §7.3.

### 5.4 Dependencia de la cámara

Las imágenes de entrenamiento provienen de un conjunto reducido de cámaras. El ruido de sensor, la compresión JPEG agresiva de algunos móviles, el enfoque automático que se equivoca y el procesamiento computacional que aplican los teléfonos modernos (nitidez artificial, reducción de ruido que borra texturas finas) alteran precisamente la señal de alta frecuencia en la que se apoya la detección de fisuras.

Un teléfono que «mejora» la foto puede estar borrando la grieta.

### 5.5 Clasificación binaria sobre una realidad continua

El sistema responde «hay grieta / no hay grieta». La realidad estructural es un continuo: microfisuración superficial, fisura de retracción, fisura activa, grieta pasante. Estas categorías tienen implicaciones muy distintas y el sistema no las distingue.

### 5.6 Las probabilidades no están calibradas

19 de las 30 fotografías con grieta obtienen exactamente **1.0** (§4.7). Con agregación por máximo sobre 24 ventanas basta con que una esté segura para saturar el resultado. Las probabilidades sirven para **ordenar** casos, no para leerse como grados de confianza: un 100 % significa «muy por encima del umbral», no «certeza». La interfaz las muestra tal cual, y esa es una limitación real de cara a un usuario no técnico.

### 5.7 El módulo de riesgo no ha sido validado por un ingeniero

Los umbrales de `config.yaml` se apoyan en criterios de la NSR-10 y en la práctica de inspección visual post-sismo, pero **no han sido revisados ni avalados por un ingeniero estructural matriculado**. Son una propuesta razonada, no un criterio profesional validado.

El diseño mitiga esto tanto como se puede: los umbrales están fuera del código y cada regla cita el criterio que la motiva, de modo que un profesional puede revisarlos y recalibrarlos en minutos. Pero mientras esa revisión no ocurra, es una limitación abierta y debe presentarse como tal.

### 5.8 El conjunto de fotos propias es pequeño

Con 60 imágenes, los intervalos de confianza de las métricas siguen siendo anchos. La mejora de §4.6 (F1 0.8889 → 0.9508) equivale a detectar 5 grietas más: la dirección es clara y el mecanismo está explicado por una medición independiente —el factor de reducción—, pero la magnitud exacta no debe leerse con tres decimales. Sirve para detectar **fallos gruesos** de generalización, no para estimar el desempeño con precisión.

---

## 6. Sesgos del dataset

| Sesgo | Descripción | Consecuencia |
|---|---|---|
| **Iluminación** | Luz uniforme y difusa, sin sombras duras ni contraluces | Falsos positivos en sombras; falsos negativos en zonas subexpuestas |
| **Material** | Predominio de hormigón liso; ausencia de ladrillo, pañete, bahareque | Fallo sistemático en construcción informal |
| **Escala** | Recortes a distancia aproximadamente constante (mediana 227 × 227 px) | Degradación con encuadres amplios. **Corregido en §4.6**: buena parte del fallo no venía del dataset sino de reducir la foto entera a 160 px; el análisis por mosaicos lo resuelve sin reentrenar |
| **Negativos fáciles** | Los negativos son superficies limpias, sin juntas, cables ni manchas | Falsas alarmas frecuentes en campo |
| **Origen geográfico** | Infraestructura de Corea del Sur y Estados Unidos | Prácticas constructivas y patrones de deterioro distintos a los colombianos |
| **Etiquetado binario** | Sin niveles de severidad | Impide graduar la respuesta |
| **Posible fuga de datos** | Parches recortados de pocas fotografías madre | Métricas infladas si el reparto es aleatorio |

Sobre el último punto: es un sesgo del *procedimiento*, no del dato, y por eso el proyecto lo trata explícitamente. `datos.split.agrupar_por_origen` reparte por superficie de origen en lugar de por imagen, el notebook de EDA incluye una celda que estima el riesgo, y el cargador imprime una advertencia cuando el split es aleatorio. Un equipo que no controle esto reportará un 99.8 % de exactitud y un modelo que falla en la primera foto real.

---

## 7. Implicaciones éticas

### 7.1 Falsos negativos en contexto sísmico

Bucaramanga está en zona de amenaza sísmica intermedia-alta y el Nido Sísmico de Bucaramanga es una de las fuentes de sismicidad más activas del país. El contexto no es hipotético.

Un falso negativo en este sistema no es un error estadístico: es **una persona que mira su pantalla, lee "Riesgo Bajo" y decide no llamar a nadie**. Si esa pared falla en el siguiente evento, la herramienta habrá contribuido activamente a la decisión equivocada.

Esto impone tres obligaciones de diseño que el proyecto cumple:

1. **El sistema debe equivocarse hacia el lado seguro.** El umbral es ajustable y el proyecto documenta explícitamente cómo priorizar recall.
2. **Nunca debe presentarse "Riesgo Bajo" como certificado de seguridad.** El texto de la aplicación dice «Sin indicios relevantes», no «Seguro». La diferencia no es cosmética: describe lo que el sistema realmente sabe.
3. **El aviso legal debe ser permanente y visible**, no una casilla que se acepta una vez y desaparece.

### 7.2 Uso indebido como dictamen técnico

El riesgo más plausible no es el mal uso malicioso, sino el **desplazamiento de responsabilidad**: un arrendador, una constructora o una administración municipal usando una captura de pantalla de la aplicación para justificar que no se contrató una inspección profesional.

Una interfaz que parece profesional aumenta ese riesgo. Es una tensión real del proyecto: se pide un panel que parezca una herramienta de ingeniería seria, y **cuanto mejor se cumple ese requisito, más creíble resulta un dictamen que el sistema no está capacitado para emitir**.

Mitigaciones aplicadas:

- Aviso permanente en la pestaña de análisis, no en un pie de página que nadie lee.
- Aviso repetido en el pie de toda la aplicación y en la pestaña «Acerca del proyecto».
- Las limitaciones se presentan **en la propia interfaz**, no escondidas en un informe aparte.
- El lenguaje de la salida es de tamizaje («requiere revisión profesional»), nunca de dictamen («elemento no apto»).

### 7.3 Sesgo del dataset frente a la construcción informal

Este es el punto ético más importante y el que menos suele analizarse.

El dataset está compuesto por hormigón de infraestructura formal, bien iluminado, de países de renta alta. Una fracción sustancial del parque construido colombiano es **autoconstrucción**: ladrillo a la vista, pañete artesanal, bahareque, mampostería sin confinar.

De ahí se sigue una consecuencia incómoda: **el sistema funcionará peor precisamente donde el riesgo estructural es mayor y donde el acceso a un ingeniero es menor**. Las edificaciones de la construcción informal son las más vulnerables ante un sismo y las que menos probabilidad tienen de recibir una inspección profesional; son, por tanto, las que más se beneficiarían de una herramienta de tamizaje gratuita. Y son las que peor cubre el modelo.

El sesgo del dataset no es neutro: **reproduce y amplifica una desigualdad preexistente**. Una herramienta que funciona bien en el edificio de oficinas y mal en la vivienda autoconstruida está, en la práctica, distribuyendo seguridad de forma desigual.

No se puede corregir con más aumento de datos ni con una arquitectura mejor. Exige **recolectar datos representativos del contexto de uso real**. Reconocerlo es lo mínimo; el proyecto, al menos, no presenta su desempeño en el dataset público como si fuese universal, y por eso separa deliberadamente la evaluación sobre fotos propias.

### 7.4 Privacidad

Las fotografías de edificaciones pueden contener personas, matrículas, interiores de viviendas y datos que permiten geolocalizar. Este prototipo procesa todo **en local** y no sube ninguna imagen a ningún servidor. Cualquier versión futura que use inferencia en la nube tendría que abordar consentimiento, retención y anonimización antes de recibir una sola foto.

### 7.5 Automatización y criterio profesional

El objetivo legítimo de una herramienta así es **ampliar la capacidad de tamizaje** —permitir que un equipo priorice qué revisar primero tras un sismo, cuando hay miles de edificaciones y decenas de inspectores— no sustituir el criterio profesional.

La diferencia entre ambos usos no está en la tecnología: está en cómo se presenta y en quién la opera. Un sistema idéntico puede ser una ayuda valiosa en manos de un equipo de inspección o una excusa peligrosa en manos de quien quiere ahorrarse una revisión.

---

## 8. Trabajo futuro

En orden de impacto esperado sobre la utilidad real del sistema:

1. **Deduplicar antes de repartir.** El hallazgo de §3.7 tiene una solución directa: agrupar las imágenes por hash perceptual **confirmado píxel a píxel** y asignar cada grupo entero a una sola partición. Es lo que hace `datos.split.agrupar_por_origen`, pero usando los píxeles como identificador de origen en lugar del nombre de archivo, que en este dataset no codifica nada. `scripts/analizar_fuga_datos.py` ya calcula los grupos; faltaría conectarlos al cargador. Es la mejora de menor esfuerzo y mayor efecto sobre la honestidad de las métricas.
2. **Menos entrenamiento, no más.** Los dos modelos del experimento de §4.9, entrenados con la cuarta parte de los datos y un tercio de las épocas, igualaron o superaron al modelo de producción sobre las fotografías propias (F1 0.9524, recall 1.00). Es una diferencia de una fotografía y no concluye nada por sí sola, pero apunta a lo mismo que §3.4 y §3.8: entrenar más ajusta el dominio público, no el real. Comprobarlo exige diseñar un criterio de parada que **no** use las 60 fotos propias, so pena de convertirlas en conjunto de validación y perder la única medida honesta de generalización del proyecto.

   *Descartado por medición:* la **degradación de escala en entrenamiento** (§4.9). Estaba implementada y parecía la continuación natural de §4.6, pero un experimento pareado mostró p = 1.000: los modelos con y sin ella emiten predicciones idénticas en las 60 fotografías. El código queda en el repositorio, desactivado, junto al resultado que lo desaconseja.

3. **Ampliar el dataset con construcción local**: ladrillo a la vista, pañete, bahareque, fotografiado en Bucaramanga con teléfonos corrientes. Sigue siendo valioso, pero §4.6 lo desplaza de la primera posición: el modelo no fallaba por falta de ejemplos, sino por verlos a una escala que nunca había encontrado.
4. **Incorporar negativos difíciles**: juntas de dilatación, cables, manchas de humedad, marcas de encofrado. Es lo que más reduciría las falsas alarmas en campo, y §4.6 sube su prioridad: con agregación por máximo sobre 24 ventanas, cada falso positivo del modelo tiene 24 oportunidades de manifestarse.
5. **Segmentación en vez de clasificación**: una U-Net ligera daría la máscara de la grieta y permitiría medir su longitud y trayectoria, no solo su presencia.
6. **Referencia métrica**: un marcador ArUco impreso pegado junto a la grieta resolvería el problema de escala de §5.1 con una impresora y cinco minutos de trabajo. Es la mejora de mayor relación valor/esfuerzo de toda la lista.
7. **Validación con un ingeniero estructural**: revisar y recalibrar los umbrales del motor de reglas. Convertiría §5.7 de limitación abierta en criterio avalado.
8. **Aplicación móvil nativa** con el `.tflite` int8 ya exportado, para inspección en campo sin conectividad.
9. **Estimación de incertidumbre** (*Monte Carlo dropout* o *deep ensembles*), para que el sistema pueda decir «no lo sé» en lugar de emitir una probabilidad sobre una imagen fuera de distribución.

---

## 9. Conclusión

El proyecto entrega un sistema completo y funcional que combina aprendizaje profundo, visión clásica y un motor de reglas explicable, ejecutable en un equipo modesto y exportable a un teléfono.

Lo que **sí** puede afirmarse:

- Detecta grietas en superficies de hormigón similares a las del entrenamiento con **F1 de 0.9423 sobre el conjunto depurado de duplicados** (275 falsos negativos sobre 2 667 grietas reales), frente al 0.9109 de una CNN propia entrenada desde cero (§3.1 y §3.7).
- Corre en CPU, sin GPU, a **4.82 ms por imagen (207 img/s) con un artefacto de 1.74 MB**: 30.8× más rápido y 8.0× más pequeño que el modelo Keras del que procede (§3.2). La viabilidad en un dispositivo modesto está medida, no supuesta.
- Se entrenó por completo en **4 h 5 min de CPU** para los tres modelos (§3.3), lo que lo hace reproducible sin infraestructura especial.
- Produce un juicio de riesgo **explicable, auditable y recalibrable sin reentrenar**, verificado por 33 pruebas unitarias del motor de reglas, dentro de una suite de 145 que cubre también inclinometría, cámara y análisis por mosaicos.
- Permite comprar recall a precio conocido: **97 grietas adicionales por 140 falsas alarmas** al bajar el umbral a 0.2495 (§4.1).
- Mide la desviación respecto a la vertical con un **error medio de 0.039°** sobre fotografía real, validado con rotaciones controladas de ±2°, ±5° y ±10° (§3.6). Es trece veces mejor que el criterio de aceptación y respalda empíricamente las reglas R5 y R6.
- Cierra buena parte de la brecha con fotografías reales **sin reentrenar ni un peso**: analizar la fotografía en mosaicos de 480 px, en lugar de reducirla entera a 160, sube el recall sobre las 60 fotos propias de **0.80 a 0.97** y baja los falsos negativos de 6 a 1 (§4.6). La causa no era el dominio, era una pérdida de escala que introducía nuestro propio preprocesado.
- Localiza la evidencia, no solo la detecta: el troceado indica **en qué región** de la fotografía está la fisura, algo que la inferencia sobre la imagen completa no puede dar.

Lo que **no** puede afirmarse:

- **Que generalice a fotografías reales sin ayuda.** Sobre las fotos propias, analizadas como el dataset público —imagen entera reducida a 160 px—, el F1 cae de 0.9423 a 0.8000 y el recall a 0.67 (§3.4, §4.6). El troceado lo recupera hasta 0.9355, pero eso es una corrección en inferencia, no un modelo que generalice por sí mismo.
- **Que un F1 de 0.9831 sea alcanzable.** Existe un umbral que lo consigue sobre estas 60 fotografías, pero pasa a 0.0006 del peor negativo: no separa clases, separa dos fotos concretas (§4.7). La cifra honesta, con validación cruzada, es 0.9355 con el umbral fijo.
- **Que el dataset público sea un conjunto de prueba honesto.** El 15.27 % de sus imágenes de prueba estaban duplicadas en entrenamiento, y los modelos las acertaban al 100 % (§3.7). Cualquier resultado publicado sobre este dataset sin deduplicar está inflado.
- **Que la variante cuantizada pueda ajustarse al dominio de despliegue.** El int8 satura sus probabilidades fuera de distribución y ningún umbral recupera sus falsos negativos (§3.5).
- **Que el módulo de inclinometría sea aplicable en la práctica.** Es preciso cuando funciona, pero **solo 1 de las 40 fotografías propias produjo una estimación fiable** (§3.6). Su precisión está demostrada; su cobertura, en un 2.5 %, no. Y con `n = 1` no se puede afirmar una precisión media poblacional.
- Que funcione en materiales y contextos constructivos no representados en el dataset.
- Que sus niveles de riesgo equivalgan a un criterio de ingeniería validado.
- Que pueda medir el ancho de fisura, que es lo que la norma exige para dictaminar.
- Que sea seguro usarlo como base para decidir si una edificación es habitable.

La distancia entre ambas listas no es un defecto del trabajo: **es el trabajo**. Un proyecto de aprendizaje automático que solo reporta su exactitud está contando la mitad de la historia, y en un dominio donde el error se paga en vidas, la mitad que falta es la que importa.
