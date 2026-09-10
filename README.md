# Detección de Grietas e Inclinación de Elementos Estructurales

**Evaluación de riesgo en edificaciones mediante visión por computador**

Proyecto de la asignatura *Algoritmos y Programación* · Ingeniería en Inteligencia Artificial · Universidad Industrial de Santander · 2026-2

---

## Qué hace

A partir de **una sola fotografía** de un elemento estructural, el sistema responde tres preguntas y las combina en un juicio explicable:

| Pregunta | Módulo | Técnica |
|---|---|---|
| ¿Hay grieta? | `src/models/` | CNN — MobileNetV2 por transfer learning |
| ¿Está a plomo? | `src/vision/` | Canny + transformada de Hough |
| ¿Qué riesgo implica? | `src/risk/` | Motor de reglas con criterios NSR-10 |

La salida no es solo un nivel de riesgo: viene con **la lista de reglas que lo dispararon** y el criterio de ingeniería que motiva cada una.

Todo está pensado para **ejecutarse en equipos modestos**: entrada de 160×160, MobileNetV2 con `alpha=0.75` y exportación a TensorFlow Lite int8. La eficiencia se mide y se reporta, no se asume.

---

## Instalación

> **Requisito:** Python **3.10, 3.11 o 3.12**. TensorFlow 2.19 no publica ruedas para 3.13+. Si `python --version` te devuelve 3.13 o superior, usa el lanzador `py -3.11` como se muestra abajo.

### Windows (PowerShell)

```powershell
cd crack-risk-assessment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea el script de activación con *"la ejecución de scripts está deshabilitada"*, habilítalos solo para tu usuario:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux / macOS

```bash
cd crack-risk-assessment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verificar que todo quedó bien

```bash
python scripts/verificar_entorno.py
```

Debe terminar con `ENTORNO CORRECTO. Fase 0 superada.` Comprueba el intérprete, las doce dependencias críticas, la carga del `config.yaml`, la importación de todos los módulos y una prueba funcional de OpenCV, TensorFlow y el motor de reglas.

---

## Configuración en VS Code

El proyecto trae `.vscode/` versionado y listo:

- **`settings.json`** — apunta al intérprete del `.venv`, activa **formateo al guardar con Black** (100 columnas), el linter **Ruff** y el descubrimiento de pruebas con pytest.
- **`launch.json`** — doce configuraciones de depuración (`F5`), una por script, incluyendo **lanzar Streamlit desde el depurador**.
- **`extensions.json`** — VS Code te ofrecerá instalar las extensiones necesarias al abrir la carpeta.

Pasos:

1. `File → Open Folder…` y abre `crack-risk-assessment/`.
2. Acepta la instalación de las extensiones recomendadas.
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → elige el del `.venv`.
4. Pulsa `F5` y selecciona una configuración de la lista.

Todo se ejecuta **en local**. El proyecto no usa Google Colab ni ningún servicio en la nube.

---

## Preparar los datos

### Dataset público → `data/raw/`

El cargador **detecta solo** cuál de las dos estructuras estás usando:

```
data/raw/                        data/raw/
├── Positive/                    ├── labels.csv        (columnas: filename, label)
│   ├── 00001.jpg          o     ├── img_00001.jpg
│   └── ...                      └── ...
└── Negative/
    └── ...
```

Se aceptan variantes de nombre (`positive`, `crack`, `con_grieta`, `1`, …); están declaradas en `datos.alias_clases` de `config.yaml`. Si usas CSV con otros nombres de columna, ajusta `datos.csv`.

### Fotos propias del equipo → `data/propias/`

**Obligatorias.** Misma estructura que `data/raw/`. Son el conjunto de prueba final y **no participan nunca** en el entrenamiento ni en la selección de hiperparámetros.

Protocolo de captura recomendado (detallado en el notebook de EDA):

- 30–50 imágenes mínimo, con ambas clases.
- Distancia de 0.5 a 1.5 m, cámara perpendicular a la superficie.
- Variedad deliberada de luz: sol, sombra, interior.
- Incluir **negativos difíciles** a propósito: juntas, cables, manchas de humedad.
- Para la inclinometría, algunas fotos con la columna o el muro **completo** y su borde vertical visible de arriba abajo.

---

## Flujo completo

Ejecuta todo desde la **raíz del repositorio**, con el entorno activado.

### 0 · Verificar el entorno

```bash
python scripts/verificar_entorno.py
```

### 1 · Análisis exploratorio

Abre `notebooks/01_eda.ipynb` en VS Code, selecciona el kernel `.venv` y ejecuta las celdas. Genera las figuras en `reports/figuras/` y documenta los sesgos del dataset.

### 2 · Ensayo rápido del pipeline (hazlo siempre primero)

Antes de lanzar una corrida de horas en CPU, valida que todo funciona con el 5 % de los datos:

```bash
python scripts/train_transfer.py --config config.yaml --subset 0.05 --epochs 2 --epochs-ft 1
```

Si esto termina sin errores, el pipeline entero está bien y puedes lanzar la corrida larga con confianza.

### 3 · Línea base (CNN propia desde cero)

```bash
python scripts/train_baseline.py --config config.yaml
```

### 4 · Transfer learning con MobileNetV2

```bash
python scripts/train_transfer.py --config config.yaml
```

Ejecuta dos etapas: base congelada y después fine-tuning de los últimos bloques con `lr=1e-5`. Guarda **ambos** modelos y **ambas** métricas para poder comparar cuánto aporta el fine-tuning.

### 5 · Exportar a TensorFlow Lite

```bash
python scripts/export_tflite.py --config config.yaml
```

Produce `_float32.tflite` (control) e `_int8.tflite` (cuantización entera calibrada con imágenes reales del conjunto de entrenamiento).

### 6 · Evaluar todo

```bash
python scripts/evaluate.py --config config.yaml
```

Evalúa línea base, MobileNetV2 y TFLite int8 sobre el conjunto de prueba **y por separado sobre las fotos propias**. Genera el análisis de falsos negativos, la curva precisión-recall, la búsqueda del umbral que garantiza el recall objetivo y la tabla comparativa desempeño-vs-costo. **Todos los artefactos que consume la aplicación salen de aquí.**

### 7 · Validar la inclinometría

```bash
python scripts/validar_inclinacion.py --config config.yaml --imagenes data/propias --guardar-anotadas
```

Rota tus fotos ángulos conocidos (±2°, ±5°, ±10°) y comprueba que la estimación se desplaza exactamente ese ángulo. Es validación sin ground truth instrumentado.

Necesita **al menos una foto con un elemento vertical largo, nítido y contrastado de arriba abajo** (una columna o la esquina de un muro, encuadrada entera). Si todas tus fotos son primeros planos de superficie, el script responderá que ninguna produjo una estimación fiable — y tendrá razón: sin una arista larga no hay nada que la transformada de Hough pueda votar.

El límite `--max-imagenes` vale 200 por defecto. Si tienes más fotos que eso, el script **avisa** de cuántas deja fuera; súbelo para incluirlas todas. Conviene que las cubra todas porque los archivos se recorren en orden alfabético, y un límite corto procesaría solo las primeras subcarpetas.

### 7 bis · Medir el efecto del análisis por mosaicos

```bash
python scripts/evaluar_mosaicos.py --config config.yaml
```

Compara analizar la fotografía entera con trocearla en ventanas del tamaño con el que se entrenó el modelo. Sobre 60 fotos propias sube el recall de **0.80 a 0.97** sin reentrenar nada: una foto de teléfono se reduce 7.5 veces para entrar al modelo, y una fisura fina no sobrevive a esa reducción. El razonamiento completo está en `reports/analisis.md` §4.6.

El troceado viene activado por defecto en la aplicación y se puede desactivar o reajustar desde la barra lateral.

### 7 ter · Comprobar si conviene mover el umbral

```bash
python scripts/calibrar_umbral.py --config config.yaml
```

Elige la regla de decisión con **validación cruzada estratificada**, no sobre las mismas fotos en las que después la mide. El resultado fue negativo y está documentado en §4.7: el umbral que parece óptimo (0.998, F1 0.9831) pasa a 0.0006 del peor negativo, así que **se mantiene el 0.5 fijo**. Un resultado negativo medido bien vale tanto como uno positivo.

### 7 quater · Qué modelo usa cada modo

La aplicación **no tiene selector de modelo**: elige ella, porque la respuesta está medida (§4.8).

| Modo | Modelo | Por qué |
|---|---|---|
| Fotografía | MobileNetV2 + mosaicos | Manda el acierto: F1 0.936, recall 0.97 |
| Vídeo en vivo | TFLite int8 | Manda la latencia: 4.9 ms, lo único que sostiene 30 FPS |

El ensemble sigue en `config.yaml` y en el comparador de latencias, pero ya no se ofrece: con mosaicos solo aporta +0.015 de F1 —un falso positivo de sesenta— por un 26 % más de tiempo. Si falta el artefacto esperado, la app repliega a otro y lo advierte.

### 7 quinquies · Degradación de escala: probada y descartada

El trabajo futuro nº 2 del informe se implementó y **se ejecutó**. El resultado fue negativo y está en §4.9: los modelos entrenados con y sin degradación de escala emiten **predicciones idénticas** en las 60 fotografías propias (McNemar p = 1.000). El troceado de §4.6 sigue siendo necesario.

La capa queda en el repositorio, desactivada, junto al experimento que la desaconseja. Para reproducirlo (unos 20 min, dos entrenamientos pareados):

```bash
python scripts/train_transfer.py --config config.yaml --subset 0.25 --epochs 3 --epochs-ft 2 --nombre escala_control
```

```bash
python scripts/train_transfer.py --config config.yaml --subset 0.25 --epochs 3 --epochs-ft 2 --nombre escala_degradado --degradacion-escala 4.0
```

```bash
python scripts/evaluar_mosaicos.py --config config.yaml --formato keras --modelo models/escala_degradado_finetuned.keras --lados 480
```

`--nombre` cambia el nombre de **todos** los artefactos, así que ningún experimento sobrescribe `models/mobilenetv2_finetuned.keras`.

### 8 · Lanzar la aplicación

```bash
streamlit run app/app.py
```

Se abre en `http://localhost:8501`.

### 9 · Pruebas

```bash
pytest tests -v
```

---

## Entrenar en CPU

El proyecto asume que **puedes no tener GPU**. Windows nativo, además, no soporta GPU en TensorFlow desde la versión 2.11 (solo vía WSL2), así que entrenar en CPU es el caso normal, no la excepción.

Herramientas incluidas para que eso sea viable:

| Herramienta | Cómo se usa |
|---|---|
| **Submuestreo** | `--subset 0.1` valida el pipeline completo en minutos |
| **Checkpoint por época** | Se guarda en `models/checkpoints/` al final de **cada** época |
| **Reanudación** | `--resume` continúa desde la época exacta en que se interrumpió |
| **Resolución modesta** | 160×160 en vez de 224×224: la mitad de cómputo |
| **`alpha=0.75`** | 39 % menos parámetros en la base de MobileNetV2 |
| **Detección de dispositivo** | Cada script informa si usa CPU o GPU al arrancar |
| **Tiempo por época + ETA** | Se imprime en cada época y se guarda en `reports/metricas/` |

Ejemplos:

```bash
# Ensayo de 5 minutos
python scripts/train_transfer.py --config config.yaml --subset 0.05 --epochs 2

# Se cortó la luz a mitad de la corrida larga
python scripts/train_transfer.py --config config.yaml --resume

# Lote más pequeño si la RAM se queda corta
python scripts/train_transfer.py --config config.yaml --batch-size 16
```

**Sobre la resolución.** 160×160 conserva la firma de una grieta fina (bordes de alta frecuencia, 2–4 px de ancho) y cuesta la mitad que 224×224, porque el costo de una convolución es cuadrático en el lado de la imagen: (160/224)² = 0.51. Bajar a 128×128 ahorraría otro 36 %, pero las fisuras capilares empiezan a perderse por submuestreo, y un falso negativo es el error caro de este dominio. 160 es el punto de equilibrio.

---

## Estructura del repositorio

```
crack-risk-assessment/
├── README.md
├── requirements.txt              # dependencias directas, versiones fijadas
├── requirements-lock.txt         # árbol completo resuelto (reproducibilidad exacta)
├── pyproject.toml                # configuración de Black, Ruff y pytest
├── config.yaml                   # ← rutas, hiperparámetros y umbrales de riesgo
├── .gitignore
├── .vscode/
│   ├── settings.json             # intérprete, Black, Ruff, pytest
│   ├── launch.json               # 12 configuraciones de depuración
│   └── extensions.json
├── .streamlit/config.toml        # tema base de la interfaz
├── notebooks/
│   └── 01_eda.ipynb              # único notebook; importa de src/
├── scripts/                      # todo ejecutable por CLI con argparse
│   ├── verificar_entorno.py
│   ├── train_baseline.py
│   ├── train_transfer.py
│   ├── evaluate.py
│   ├── export_tflite.py
│   ├── validar_inclinacion.py
│   ├── analizar_fuga_datos.py    # duplicados entre particiones (pHash + correlación)
│   ├── evaluar_robustez.py       # TTA, ensemble y recalibración del umbral
│   ├── evaluar_mosaicos.py       # imagen entera vs. análisis por ventanas
│   └── calibrar_umbral.py        # umbral con validación cruzada, sin oráculo
├── src/
│   ├── utils/                    # rutas, config, semillas, dispositivo
│   ├── data/loader.py            # carga, split, augmentation
│   ├── models/
│   │   ├── baseline.py           # CNN propia
│   │   ├── transfer.py           # MobileNetV2
│   │   ├── entrenamiento.py      # callbacks y reanudación compartidos
│   │   └── inferencia.py         # interfaz común .keras / .tflite
│   ├── vision/inclinacion.py     # Canny + Hough
│   ├── risk/reglas.py            # motor de reglas (función pura)
│   └── eval/
│       ├── metricas.py           # desempeño
│       └── complejidad.py        # parámetros, tamaño, latencia, memoria
├── app/
│   ├── app.py                    # Streamlit
│   └── estilos.py                # CSS y paleta
├── tests/
│   ├── test_reglas.py
│   └── test_inclinacion.py
├── models/                       # .keras y .tflite (no versionados)
├── data/                         # dataset y fotos propias (no versionado)
└── reports/
    ├── analisis.md               # análisis crítico
    ├── figuras/
    └── metricas/                 # JSON/CSV que consume la app
```

### Reglas de arquitectura que el proyecto respeta

- **Un solo notebook**, y solo para EDA. Importa de `src/`, no define lógica.
- **Todo lo que entrena, evalúa o exporta es un script con `argparse`.**
- **Cero rutas absolutas.** Todo se construye con `pathlib` desde `src/utils/rutas.py`, relativo a la raíz del repositorio.
- **Cero variables globales mutables.** Los parámetros se pasan como argumentos.
- **Cero valores incrustados.** Cada umbral, hiperparámetro y ruta vive en `config.yaml`.
- **Semillas en un único sitio** (`src/utils/semillas.py`), invocadas al inicio de cada script.
- **La app no calcula métricas.** Lee artefactos de `reports/metricas/`.

---

## La aplicación

Tres pestañas:

**🔍 Análisis en vivo** — Carga la foto y ejecuta el pipeline completo con barra de progreso por etapas reales (carga → preprocesado → clasificación → inclinometría → reglas). Muestra probabilidad de grieta, ángulo de desaplome, orientación de la fisura y tiempo de inferencia medido; original y procesada lado a lado en columnas de igual ancho; semáforo de riesgo con las reglas que lo justifican.

**📊 Métricas del modelo** — Matriz de confusión, curvas de entrenamiento, curva precisión-recall y tabla comparativa desempeño-vs-costo, todo con Plotly interactivo y **leído de `reports/metricas/`**.

**ℹ️ Acerca del proyecto** — Problema, enfoque, dataset, limitaciones, equipo y umbrales vigentes.

En la barra lateral: carga de imagen, elemento inspeccionado, **selector `.keras` / `.tflite` con comparación de latencia en vivo**, ficha técnica del modelo y cinco controles de sensibilidad de OpenCV que alteran el resultado en tiempo real, inicializados con los valores de `config.yaml`.

**Accesibilidad:** el semáforo nunca depende solo del color. Cada nivel lleva icono (`✓ ▲ ✕`) y etiqueta de texto — entre el 5 y el 8 % de los hombres tiene alguna deficiencia en la visión del rojo y el verde, y en un panel de seguridad eso no es un detalle estético.

---

## Ajustar la configuración

Todo vive en `config.yaml`. Los ajustes que más se tocan:

```yaml
preproceso:
  alto: 160            # ↓ para entrenar más rápido, ↑ para detectar fisuras más finas
  ancho: 160
  barajar_buffer: 2000 # ventana de barajado de tf.data (ver nota abajo)
  cache: false         # true solo si el dataset cacheado cabe en RAM (ver nota abajo)

entrenamiento:
  batch_size: 32       # ↓ a 16 u 8 si la RAM se queda corta
  epocas: 20

datos:
  split:
    agrupar_por_origen: false   # ← ponlo en true si el dataset son recortes
    regex_origen: "^([A-Za-z]+[_-]?\\d+)"

evaluacion:
  umbral: 0.5          # ↓ a ~0.3 para priorizar recall (menos falsos negativos)

riesgo:
  desaplome_atencion_grados: 1.0
  desaplome_severo_grados: 2.0
```

**Fuga de datos:** si tu dataset son parches recortados de un número reducido de fotografías madre, el reparto aleatorio pondrá parches casi idénticos en entrenamiento y en prueba, y la exactitud saldrá irrealmente alta. Activa `agrupar_por_origen: true` y ajusta `regex_origen` al patrón real de tus nombres de archivo. El notebook de EDA incluye una celda que estima ese riesgo.

**`cache`: calcula si te cabe antes de activarlo.** `cache: true` guarda en RAM las imágenes ya decodificadas y redimensionadas, lo que acelera mucho a partir de la segunda época — **si caben**. La cuenta es directa:

```
bytes = (n_train + n_val) × alto × ancho × 3 × 4
```

Con 49 417 imágenes a 160×160 eso son **15.2 GB**. En un equipo de 16 GB no cabe, Windows empieza a paginar a disco y cada época se vuelve *más lenta* que sin caché. Si la suma se acerca a tu RAM libre, déjalo en `false`: se vuelve a decodificar cada época, pero `prefetch` y `num_parallel_calls` lo solapan con el cómputo y el tiempo es predecible.

**`barajar_buffer`: no lo bajes.** `tf.data` solo mezcla dentro de una ventana deslizante de ese tamaño. El inventario se baraja ya en Python al hacer el split (`src/data/loader.py`, función `dividir`), así que este buffer solo aporta variación entre épocas y 2000 basta. **Pero si modificas el cargador y el inventario vuelve a quedar ordenado por clase**, un buffer de 2000 sobre 40 000 imágenes hará que el modelo vea cientos de lotes seguidos de una sola clase, colapse a predecir siempre la última que vio y dé `val_auc = 0.5` con una exactitud de entrenamiento del 99 %. Es un fallo silencioso: no lanza ningún error.

---

## Reproducibilidad

- Semilla global única en `config.yaml` (`proyecto.semilla: 42`), propagada a `random`, NumPy y TensorFlow desde `src/utils/semillas.py`.
- Versiones fijadas con `==` en `requirements.txt`; árbol completo en `requirements-lock.txt`.
- Splits deterministas: misma semilla, mismo reparto.
- Cada corrida guarda su resumen en `reports/metricas/entrenamiento_*.json` con hiperparámetros, tiempos por época y hardware usado.

Para determinismo bit a bit (a costa de velocidad), llama a `fijar_semillas(42, determinismo_estricto=True)`.

---

## Aviso

Este es un **prototipo académico de tamizaje**. No sustituye la inspección de un ingeniero estructural matriculado ni constituye un dictamen técnico bajo la NSR-10.

El sistema **no puede medir el ancho real de una grieta** sin una referencia métrica en la escena, y el ancho de fisura es precisamente el criterio que usa la norma. Las limitaciones, los sesgos del dataset y las implicaciones éticas están desarrollados en [`reports/analisis.md`](reports/analisis.md).

---

## Commits sugeridos por fase

```
feat(fase0): andamiaje del proyecto, entorno y configuración reproducible
feat(fase1): carga de datos, split estratificado y notebook de EDA
feat(fase2): CNN de línea base entrenada desde cero
feat(fase3): transfer learning con MobileNetV2 y fine-tuning en dos etapas
feat(fase4): inclinometría con Canny + Hough y validación por rotaciones
feat(fase5): métricas, análisis de complejidad y motor de reglas de riesgo
feat(fase6): aplicación Streamlit con panel interactivo y exportación TFLite
docs(fase7): análisis crítico, limitaciones e implicaciones éticas
```
