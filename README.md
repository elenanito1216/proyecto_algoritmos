# Detección de Grietas e Inclinación de Elementos Estructurales

**Evaluación de riesgo en edificaciones mediante visión por computador**

Proyecto de la asignatura *Algoritmos y Programación* · Ingeniería en Inteligencia Artificial · Universidad Industrial de Santander · Semestre 2026-2

---

## Para quien evalúa este trabajo

Este documento explica **qué hace el proyecto, cómo se construyó y qué se descubrió al medirlo**. Está pensado para leerse de principio a fin sin ejecutar nada; las instrucciones para ponerlo en marcha están agrupadas más abajo.

Si dispone de poco tiempo, estas son las secciones que concentran el trabajo:

| Si le interesa… | Vaya a |
|---|---|
| Qué hace el sistema y qué tan bien funciona | [Qué hace](#qué-hace-el-sistema) y [Resultados](#resultados-medidos) |
| El análisis crítico y los hallazgos | [Lo que se descubrió al medir](#lo-que-se-descubrió-al-medir) |
| El informe completo | [`reports/analisis.md`](reports/analisis.md) |
| Ejecutarlo | [Cómo ponerlo en marcha](#cómo-ponerlo-en-marcha) |

Al final hay un [glosario](#pequeño-glosario) con los términos técnicos explicados en lenguaje corriente.

---

## El problema

Después de un sismo, o simplemente con el paso de los años, las edificaciones desarrollan grietas y se inclinan. Distinguir una fisura inofensiva de una que anuncia un problema estructural requiere un ingeniero, y no siempre hay uno disponible cuando hace falta revisar muchas viviendas en poco tiempo.

Este proyecto construye una **herramienta de tamizaje**: a partir de una fotografía tomada con un teléfono corriente, señala qué elementos merecen la visita de un profesional y cuáles probablemente no. No reemplaza al ingeniero — le ayuda a decidir por dónde empezar.

---

## Qué hace el sistema

A partir de **una sola fotografía**, el sistema responde tres preguntas y las combina en un juicio que puede explicarse:

| Pregunta | Cómo la responde | En qué consiste |
|---|---|---|
| **¿Hay una grieta?** | Una red neuronal | El computador aprendió a reconocer grietas viendo cuarenta mil fotografías ya etiquetadas |
| **¿Está derecho el elemento?** | Geometría clásica | Se detectan los bordes rectos de la imagen y se mide cuánto se desvían de la vertical |
| **¿Qué riesgo implica?** | Un conjunto de reglas | Reglas escritas a mano, inspiradas en la norma colombiana NSR-10 |

La diferencia frente a un sistema que solo diga «riesgo alto» es que aquí **la respuesta viene con sus motivos**: la aplicación muestra qué reglas se activaron y qué criterio de ingeniería respalda cada una. Quien recibe el resultado puede discutirlo, no solo aceptarlo.

### Por qué tres módulos y no uno solo

Se podría haber entrenado una única red neuronal que dijera directamente «riesgo alto» o «riesgo bajo». Se descartó por dos razones:

1. **No habría forma de explicar sus decisiones.** Una red neuronal es una caja opaca; un conjunto de reglas se puede leer, discutir y corregir.
2. **No habría forma de ajustarla sin volver a entrenarla.** Si un ingeniero considera que el límite de inclinación debería ser 1.5° en vez de 2°, aquí se cambia una línea en un archivo de texto. Con una red neuronal habría que reentrenar desde cero.

---

## Resultados medidos

### Detección de grietas

| Qué se midió | Resultado | Qué significa |
|---|---|---|
| **Acierto general (F1)** | **0.9423** | Resume en un número cuántas grietas encuentra y cuántas veces se equivoca al avisar. Va de 0 a 1 |
| Grietas que sí detecta | 90.7 % | De cada 100 grietas reales, encuentra unas 91 |
| Tiempo por fotografía | **4.82 milisegundos** | Unas 207 fotografías por segundo, en un computador corriente sin tarjeta gráfica |
| Tamaño del modelo | **1.74 MB** | Cabe holgadamente en un teléfono |

Ese modelo comprimido para teléfono es **30.8 veces más rápido y 8 veces más pequeño** que la versión original, sin perder acierto de forma apreciable.

### Medición de inclinación

| Qué se midió | Resultado |
|---|---|
| Error al medir el ángulo | **0.039°** — trece veces mejor que el criterio de aceptación |
| Fotografías en las que consigue medir | **1 de cada 40** |

La segunda cifra es incómoda y se reporta igual. El módulo es **muy preciso cuando funciona, y funciona pocas veces**: necesita ver el borde vertical completo de una columna o un muro, y la mayoría de las fotografías son primeros planos de la superficie. Reportar solo el 0.039° habría sido contar media historia.

### Costo de construirlo

**4 horas y 5 minutos** de cómputo en un portátil corriente, sin tarjeta gráfica, para entrenar los tres modelos. El proyecto está diseñado para reproducirse sin infraestructura especial.

---

## Lo que se descubrió al medir

Esta es la parte del trabajo que el equipo considera más valiosa. Son cuatro hallazgos, y **dos de ellos son negativos**: mejoras que parecían buenas y que las mediciones obligaron a descartar.

### 1. El conjunto de datos público estaba contaminado

Los datos públicos vienen divididos en dos partes: una para que el modelo aprenda y otra, apartada, para examinarlo. La segunda solo sirve si contiene imágenes que el modelo nunca vio.

Al comprobarlo, resultó que **el 15.27 % de las imágenes del examen ya estaban en el material de estudio**. El modelo las acertaba todas — no porque hubiera aprendido, sino porque las recordaba. Es el equivalente a examinar a un estudiante con las preguntas que ya le dieron resueltas.

Todas las cifras de este documento están calculadas **después** de retirar esas imágenes repetidas. Son más bajas que las que saldrían sin depurar, y son las honestas.

### 2. El problema no estaba en el modelo, estaba en cómo le entregábamos las fotos

El modelo funcionaba bien con las fotografías del conjunto público y mucho peor con las que tomó el equipo. La explicación que se dio por buena durante semanas fue que las paredes colombianas son distintas: otro material, otra luz, otra cámara.

Al medir algo que nadie había mirado —**el tamaño de las imágenes**— apareció otra explicación:

| Origen | Tamaño típico | Cuánto se encoge al entrar al modelo |
|---|---|---|
| Fotos de entrenamiento | 227 × 227 píxeles | Se reduce 1.4 veces |
| Fotos del equipo | 1200 × 1600 píxeles | **Se reduce 7.5 veces** |

El modelo trabaja con imágenes de 160 × 160 píxeles, así que toda fotografía se encoge antes de entrar. Una grieta de 3 píxeles de ancho sobrevive como 2 píxeles en el primer caso, pero queda en **0.4 píxeles** en el segundo: desaparece. **La grieta no se le escapaba al modelo — la borrábamos nosotros antes de enseñársela.**

La solución no exige volver a entrenar nada: se trocea la fotografía en ventanas y se analiza cada una por separado, de modo que cada trozo llega al modelo a un tamaño reconocible.

| | Grietas que detecta | Grietas que se le escapan |
|---|---|---|
| Antes (fotografía completa) | 80 % | 6 de 30 |
| **Después (por ventanas)** | **97 %** | **1 de 30** |

El precio también se reporta: aparecen 2 falsas alarmas donde antes no había ninguna, y cada fotografía tarda 0.78 segundos en vez de 0.32. En este dominio es el intercambio correcto — una falsa alarma cuesta una visita de inspección; una grieta no detectada puede costar vidas.

Como beneficio no buscado, el troceado indica **en qué parte de la fotografía** está la grieta, algo que antes era imposible saber.

### 3. Existe un ajuste que mejora los números, y no se usa

El sistema decide «hay grieta» cuando su confianza supera cierto nivel. Ajustar ese nivel no cuesta nada y parecía la última mejora fácil.

Buscándolo bien, aparece uno que sube el acierto de 0.9355 a **0.9831** — el mejor número de todo el proyecto. **Se descartó.**

El motivo: ese nivel de corte pasa a **seis diezmilésimas** de la fotografía sana peor clasificada. No separa dos categorías, separa dos fotografías concretas de una muestra de sesenta. Cualquier foto nueva con una junta un poco más marcada caería del otro lado.

Se probó además una alternativa que parecía más sólida —exigir que la grieta aparezca en varias ventanas y no en una sola— y, midiéndola correctamente, resultó **peor que no hacer nada**.

Se mantiene el valor original. Un 0.9831 que no se sostiene vale menos que un 0.9355 que sí.

### 4. La continuación lógica no funcionó

Si el problema era que las grietas se destruyen al encoger la imagen, lo natural era entrenar el modelo mostrándole grietas ya encogidas, para que aprendiera a reconocerlas así. Se implementó y se probó con dos entrenamientos idénticos salvo por esa diferencia.

Resultado: los dos modelos dan **exactamente las mismas respuestas** en las 60 fotografías, y la prueba estadística arroja el resultado menos significativo posible. La idea, sencillamente, no hace nada.

Se descartó antes de gastar las 2 horas del entrenamiento completo, gracias a una comprobación de 20 minutos diseñada justamente para eso.

### El patrón que une los cuatro

En los cuatro casos la conclusión salió de **comparar contra la alternativa aburrida antes de aceptar la interesante**. Ninguno necesitó una técnica avanzada; todos necesitaron mirar una cifra elemental que nadie había mirado.

---

## La aplicación

Se ejecuta en el navegador y tiene cuatro pestañas.

**🔍 Análisis en vivo.** Se carga una fotografía y se obtiene el resultado completo: probabilidad de grieta, ángulo de inclinación, orientación de la fisura y el semáforo de riesgo con las reglas que lo justifican. Muestra la imagen original y la procesada lado a lado, y señala con recuadros en qué zona encontró la evidencia.

**📹 Cámara en vivo.** Analiza en tiempo real lo que ve la cámara, con el vídeo y el veredicto de riesgo uno al lado del otro. Puede usar la cámara del computador o **la del teléfono a través de la red local**, que es lo práctico para caminar por una edificación.

**📊 Métricas del modelo.** Gráficas interactivas del desempeño. La aplicación **no calcula estas cifras**: las lee de archivos generados por los programas de evaluación, de modo que lo que aparece en pantalla es exactamente lo que se midió.

**ℹ️ Acerca del proyecto.** Problema, enfoque, limitaciones y equipo.

**Sobre accesibilidad:** el semáforo de riesgo nunca depende solo del color. Cada nivel lleva icono (`✓ ▲ ✕`) y etiqueta escrita, porque entre el 5 y el 8 % de los hombres tiene alguna dificultad para distinguir el rojo del verde, y en un panel de seguridad eso no es un detalle estético.

### Qué modelo usa y por qué no hay que elegirlo

La aplicación **no pregunta qué modelo usar**: lo decide sola, porque la respuesta está medida y no depende de la preferencia de nadie.

| Situación | Modelo que usa | Motivo |
|---|---|---|
| Analizar una fotografía | El más preciso | Se admite medio segundo de cálculo, así que manda el acierto |
| Vídeo en vivo | El más rápido | Hay 33 milésimas de segundo por imagen; ningún otro cabe en ese margen |

Ofrecer un selector parecía más flexible, pero trasladaba al usuario una decisión que no puede tomar bien: para elegir con criterio habría que conocer las cifras de acierto y velocidad de cada opción, que están en el informe y no en la pantalla.

---

## Cómo ponerlo en marcha

> **Requisito previo:** Python **3.10, 3.11 o 3.12**. La librería TensorFlow todavía no funciona con la versión 3.13 o superior.

### 1. Preparar el entorno

**En Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

**En Linux o macOS**

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Y una comprobación de que todo quedó bien instalado:

```bash
python scripts/verificar_entorno.py
```

### 2. Colocar las imágenes

El repositorio no incluye las fotografías, porque pesan demasiado. Se necesitan dos conjuntos, en carpetas separadas por clase:

```
data/raw/          ← dataset público, para entrenar
├── Positive/      ← imágenes con grieta
└── Negative/      ← imágenes sin grieta

data/propias/      ← fotografías del equipo, para el examen final
├── Positive/
└── Negative/
```

Las fotografías propias son **60 imágenes tomadas por el equipo** y cumplen una función que el dataset público no puede cumplir: son el único material que el modelo no ha visto en ninguna forma. **Nunca se usan para entrenar ni para tomar decisiones de diseño** — en cuanto se usaran para decidir algo dejarían de ser una medida honesta.

### 3. Ejecutar la aplicación

Si los modelos ya están entrenados (hay archivos en la carpeta `models/`), basta con:

```bash
streamlit run app/app.py
```

### 4. Reconstruirlo todo desde cero

Solo si se desea reproducir el proceso completo. Se ejecuta desde la carpeta raíz del proyecto, en este orden:

```bash
python scripts/train_baseline.py --config config.yaml
```

```bash
python scripts/train_transfer.py --config config.yaml
```

```bash
python scripts/export_tflite.py --config config.yaml
```

```bash
python scripts/evaluate.py --config config.yaml
```

En un portátil sin tarjeta gráfica esto toma unas **4 horas**. Para comprobar antes que todo encaja, hay un ensayo de cinco minutos que recorre el proceso completo con una fracción de los datos:

```bash
python scripts/train_transfer.py --config config.yaml --subset 0.05 --epochs 2 --epochs-ft 1
```

Y si el entrenamiento largo se interrumpe (se cierra el portátil, se va la luz), continúa exactamente donde quedó:

```bash
python scripts/train_transfer.py --config config.yaml --resume
```

### 5. Reproducir los experimentos del análisis

Cada hallazgo de la sección anterior tiene su propio programa, y cada uno puede ejecutarse por separado:

| Hallazgo | Comando |
|---|---|
| Imágenes repetidas entre estudio y examen | `python scripts/analizar_fuga_datos.py --config config.yaml` |
| Análisis por ventanas | `python scripts/evaluar_mosaicos.py --config config.yaml` |
| Ajuste del nivel de corte | `python scripts/calibrar_umbral.py --config config.yaml` |
| Medición de la inclinación | `python scripts/validar_inclinacion.py --config config.yaml --imagenes data/propias` |

### 6. Ejecutar las pruebas automáticas

```bash
pytest tests -v
```

Son **160 pruebas** que verifican el comportamiento del sistema:

| Archivo | Pruebas | Qué comprueba |
|---|---|---|
| `test_inclinacion.py` | 45 | Medición de ángulos y sus casos límite |
| `test_camara.py` | 42 | Vídeo en tiempo real y presentación en pantalla |
| `test_reglas.py` | 33 | Motor de riesgo, incluida la coherencia entre niveles |
| `test_mosaicos.py` | 29 | Troceado de imágenes y elección automática de modelo |
| `test_degradacion.py` | 11 | El experimento descartado del hallazgo 4 |

---

## Cómo está organizado el código

```
crack-risk-assessment/
├── config.yaml         ← TODOS los ajustes del proyecto viven aquí
├── README.md
├── requirements.txt    ← librerías necesarias, con versiones exactas
│
├── src/                ← la lógica del proyecto
│   ├── data/           ·  leer imágenes y repartirlas en grupos
│   ├── models/         ·  las redes neuronales y cómo se usan
│   ├── vision/         ·  medición de ángulos y manejo de la cámara
│   ├── risk/           ·  las reglas de riesgo
│   ├── eval/           ·  cálculo de métricas y de costo computacional
│   └── utils/          ·  rutas, configuración, semillas aleatorias
│
├── scripts/            ← programas ejecutables desde la terminal
├── app/                ← la aplicación web
├── tests/              ← las 160 pruebas automáticas
├── notebooks/          ← exploración inicial de los datos
├── reports/            ← el informe y las cifras medidas
│   ├── analisis.md     ·  ANÁLISIS CRÍTICO COMPLETO
│   ├── metricas/       ·  cifras en bruto que la aplicación lee
│   └── figuras/        ·  gráficas
│
├── models/             ← modelos entrenados (no se versionan: pesan mucho)
└── data/               ← imágenes (no se versionan)
```

### Reglas que el proyecto se impuso

Estas restricciones se decidieron al comenzar y se respetaron durante todo el desarrollo:

- **Un único notebook**, y solo para explorar los datos. Toda la lógica vive en `src/`.
- **Todo lo que entrena o evalúa es un programa de terminal**, con sus opciones documentadas.
- **Ninguna ruta de archivo escrita a mano.** Todas se construyen a partir de la carpeta raíz, de modo que el proyecto funciona en cualquier computador sin retocar nada.
- **Ningún número suelto dentro del código.** Cada umbral y cada parámetro está en `config.yaml`, acompañado de un comentario que explica por qué tiene ese valor y no otro.
- **La aplicación no calcula métricas, las lee.** Así es imposible que la pantalla muestre cifras distintas de las que se midieron.

---

## Reproducibilidad

Un resultado que no se puede repetir no es un resultado. El proyecto toma cuatro medidas para que cualquiera obtenga lo mismo:

- **Una única semilla aleatoria** (el número 42, declarado en `config.yaml`) que se propaga a todas las librerías. El azar del entrenamiento es siempre el mismo azar.
- **Versiones de librerías fijadas exactamente**, no «la más reciente disponible».
- **Reparto de datos determinista**: la misma semilla produce siempre la misma división entre entrenamiento y examen.
- **Cada entrenamiento deja su registro** en `reports/metricas/`, con los parámetros usados, el tiempo por vuelta y el equipo en el que corrió.

---

## Advertencia sobre el alcance

Este es un **prototipo académico de tamizaje**. No sustituye la inspección de un ingeniero estructural matriculado ni constituye un dictamen técnico bajo la norma NSR-10.

Hay una limitación de fondo que conviene entender: **el sistema no puede medir el ancho real de una grieta** a partir de una fotografía si no hay en la escena algún objeto de tamaño conocido que sirva de referencia — y el ancho de la fisura es precisamente el criterio que la norma usa para dictaminar. El sistema dice *dónde mirar*, no *qué concluir*.

Las limitaciones completas, los sesgos del conjunto de datos y las implicaciones éticas están desarrollados en [`reports/analisis.md`](reports/analisis.md).

---

## Pequeño glosario

Los términos que aparecen en el informe, en lenguaje corriente:

| Término | Qué significa aquí |
|---|---|
| **Falso negativo** | Hay grieta y el sistema dice que no. **Es el error caro**: una grieta que nadie va a revisar |
| **Falso positivo** | No hay grieta y el sistema avisa. Cuesta una inspección innecesaria |
| **Recall** | De todas las grietas que existen, qué proporción encuentra el sistema |
| **Precisión** | De todas las veces que el sistema avisa, qué proporción son grietas de verdad |
| **F1** | Un solo número que resume recall y precisión. Sube cuando ambos suben |
| **Transfer learning** | Partir de una red que ya aprendió a ver imágenes en general y reentrenarla para grietas. Ahorra datos y tiempo |
| **Umbral** | El nivel de confianza a partir del cual el sistema afirma que hay grieta. Bajarlo detecta más grietas y también más falsas alarmas |
| **Cuantización (int8)** | Guardar los números del modelo con menos detalle. Lo vuelve mucho más pequeño y rápido, a cambio de algo de precisión |
| **Desaplome** | Cuánto se desvía de la vertical un elemento que debería estar derecho |
| **Sobreajuste** | Cuando un modelo memoriza los ejemplos en vez de aprender el patrón. Acierta en lo conocido y falla en lo nuevo |
| **Época** | Una vuelta completa del entrenamiento a todas las imágenes disponibles |

---

## Equipo

Proyecto desarrollado por Santiago Gómez García y equipo para la asignatura *Algoritmos y Programación*, Ingeniería en Inteligencia Artificial, Universidad Industrial de Santander, semestre 2026-2.
