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

### 5.6 El módulo de riesgo no ha sido validado por un ingeniero

Los umbrales de `config.yaml` se apoyan en criterios de la NSR-10 y en la práctica de inspección visual post-sismo, pero **no han sido revisados ni avalados por un ingeniero estructural matriculado**. Son una propuesta razonada, no un criterio profesional validado.

El diseño mitiga esto tanto como se puede: los umbrales están fuera del código y cada regla cita el criterio que la motiva, de modo que un profesional puede revisarlos y recalibrarlos en minutos. Pero mientras esa revisión no ocurra, es una limitación abierta y debe presentarse como tal.

### 5.7 El conjunto de fotos propias es pequeño

Con 30–50 imágenes, los intervalos de confianza de las métricas son anchos. Un F1 de 0.85 sobre 40 imágenes es compatible con un valor real bastante peor. Sirve para detectar **fallos gruesos** de generalización, no para estimar el desempeño con precisión.

---

## 6. Sesgos del dataset

| Sesgo | Descripción | Consecuencia |
|---|---|---|
| **Iluminación** | Luz uniforme y difusa, sin sombras duras ni contraluces | Falsos positivos en sombras; falsos negativos en zonas subexpuestas |
| **Material** | Predominio de hormigón liso; ausencia de ladrillo, pañete, bahareque | Fallo sistemático en construcción informal |
| **Escala** | Recortes a distancia aproximadamente constante | Degradación con encuadres amplios |
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
2. **Ampliar el dataset con construcción local**: ladrillo a la vista, pañete, bahareque, fotografiado en Bucaramanga con teléfonos corrientes. Es lo que más mejoraría la utilidad real, y también lo más laborioso.
3. **Incorporar negativos difíciles**: juntas de dilatación, cables, manchas de humedad, marcas de encofrado. Es lo que más reduciría las falsas alarmas en campo.
4. **Segmentación en vez de clasificación**: una U-Net ligera daría la máscara de la grieta y permitiría medir su longitud y trayectoria, no solo su presencia.
5. **Referencia métrica**: un marcador ArUco impreso pegado junto a la grieta resolvería el problema de escala de §5.1 con una impresora y cinco minutos de trabajo. Es la mejora de mayor relación valor/esfuerzo de toda la lista.
6. **Validación con un ingeniero estructural**: revisar y recalibrar los umbrales del motor de reglas. Convertiría §5.6 de limitación abierta en criterio avalado.
7. **Aplicación móvil nativa** con el `.tflite` int8 ya exportado, para inspección en campo sin conectividad.
8. **Estimación de incertidumbre** (*Monte Carlo dropout* o *deep ensembles*), para que el sistema pueda decir «no lo sé» en lugar de emitir una probabilidad sobre una imagen fuera de distribución.

---

## 9. Conclusión

El proyecto entrega un sistema completo y funcional que combina aprendizaje profundo, visión clásica y un motor de reglas explicable, ejecutable en un equipo modesto y exportable a un teléfono.

Lo que **sí** puede afirmarse:

- Detecta grietas en superficies de hormigón similares a las del entrenamiento con **F1 de 0.9423 sobre el conjunto depurado de duplicados** (275 falsos negativos sobre 2 667 grietas reales), frente al 0.9109 de una CNN propia entrenada desde cero (§3.1 y §3.7).
- Corre en CPU, sin GPU, a **4.82 ms por imagen (207 img/s) con un artefacto de 1.74 MB**: 30.8× más rápido y 8.0× más pequeño que el modelo Keras del que procede (§3.2). La viabilidad en un dispositivo modesto está medida, no supuesta.
- Se entrenó por completo en **4 h 5 min de CPU** para los tres modelos (§3.3), lo que lo hace reproducible sin infraestructura especial.
- Produce un juicio de riesgo **explicable, auditable y recalibrable sin reentrenar**, verificado por 33 pruebas unitarias que incluyen propiedades de monotonía.
- Permite comprar recall a precio conocido: **97 grietas adicionales por 140 falsas alarmas** al bajar el umbral a 0.2495 (§4.1).
- Mide la desviación respecto a la vertical con un **error medio de 0.039°** sobre fotografía real, validado con rotaciones controladas de ±2°, ±5° y ±10° (§3.6). Es trece veces mejor que el criterio de aceptación y respalda empíricamente las reglas R5 y R6.
- Reduce la brecha con fotografías reales **sin reentrenar**: el ensemble de los dos modelos sube el F1 sobre fotos propias de 0.6667 a **0.8235** y elimina 4 de los 10 falsos negativos, por un 11 % de latencia adicional (§3.8).

Lo que **no** puede afirmarse:

- **Que generalice a fotografías reales.** Sobre las 40 fotos propias, el F1 cae a 0.6667 y el recall a 0.50: se pierde la mitad de las grietas (§3.4). Y el modelo que gana en el dataset público es el que peor generaliza.
- **Que el dataset público sea un conjunto de prueba honesto.** El 15.27 % de sus imágenes de prueba estaban duplicadas en entrenamiento, y los modelos las acertaban al 100 % (§3.7). Cualquier resultado publicado sobre este dataset sin deduplicar está inflado.
- **Que la variante cuantizada pueda ajustarse al dominio de despliegue.** El int8 satura sus probabilidades fuera de distribución y ningún umbral recupera sus falsos negativos (§3.5).
- **Que el módulo de inclinometría sea aplicable en la práctica.** Es preciso cuando funciona, pero **solo 1 de las 40 fotografías propias produjo una estimación fiable** (§3.6). Su precisión está demostrada; su cobertura, en un 2.5 %, no. Y con `n = 1` no se puede afirmar una precisión media poblacional.
- Que funcione en materiales y contextos constructivos no representados en el dataset.
- Que sus niveles de riesgo equivalgan a un criterio de ingeniería validado.
- Que pueda medir el ancho de fisura, que es lo que la norma exige para dictaminar.
- Que sea seguro usarlo como base para decidir si una edificación es habitable.

La distancia entre ambas listas no es un defecto del trabajo: **es el trabajo**. Un proyecto de aprendizaje automático que solo reporta su exactitud está contando la mitad de la historia, y en un dominio donde el error se paga en vidas, la mitad que falta es la que importa.
