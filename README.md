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
| Cómo se entrenó la red neuronal | [Cómo aprendió el modelo](#cómo-aprendió-el-modelo-a-reconocer-grietas) |
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
| **¿Cuánto mide la grieta?** | Geometría + un marcador impreso | Recorrido, ancho y forma. En milímetros si hay una referencia de tamaño en la foto |

La diferencia frente a un sistema que solo diga «riesgo alto» es que aquí **la respuesta viene con sus motivos**: la aplicación muestra qué reglas se activaron y qué criterio de ingeniería respalda cada una. Quien recibe el resultado puede discutirlo, no solo aceptarlo.

### Por qué tres módulos y no uno solo

Se podría haber entrenado una única red neuronal que dijera directamente «riesgo alto» o «riesgo bajo». Se descartó por dos razones:

1. **No habría forma de explicar sus decisiones.** Una red neuronal es una caja opaca; un conjunto de reglas se puede leer, discutir y corregir.
2. **No habría forma de ajustarla sin volver a entrenarla.** Si un ingeniero considera que el límite de inclinación debería ser 1.5° en vez de 2°, aquí se cambia una línea en un archivo de texto. Con una red neuronal habría que reentrenar desde cero.

---

## Cómo aprendió el modelo a reconocer grietas

Una red neuronal no se programa con reglas del tipo «si hay una línea oscura, es una grieta». Se le muestran miles de ejemplos ya etiquetados y ella misma va corrigiendo, poco a poco, sus criterios internos cada vez que se equivoca. Esta sección cuenta cómo se hizo, con las cifras que dejó registradas el propio entrenamiento.

### 1. El material de estudio

**58 138 fotografías** de superficies de hormigón, cada una etiquetada como «con grieta» o «sin grieta». Se repartieron al azar en tres grupos que nunca se mezclan:

| Grupo | Fotografías | Para qué sirve | Equivalente escolar |
|---|---|---|---|
| Estudio | 40 696 (70 %) | El modelo las ve una y otra vez y aprende de sus errores | Los ejercicios del libro |
| Validación | 8 721 (15 %) | Se consultan al terminar cada vuelta, para saber si está aprendiendo o memorizando | Los simulacros |
| Prueba | 8 721 (15 %) | Se usan una sola vez, al final, para poner la nota | El examen final |

**Hay más fotos sin grieta (59 %) que con grieta (41 %).** Si no se corrige, el modelo descubre que responder «no hay grieta» es la apuesta que más veces acierta — y en seguridad estructural callarse una grieta es justamente el error caro. Por eso, durante el estudio, **cada foto con grieta cuenta 1.44 veces más** que una sin grieta: equivocarse en ellas le cuesta más al modelo, y aprende a no hacerlo.

### 2. Cómo se le presentan las fotografías

**Todas se reducen a 160 × 160 píxeles.** Es el tamaño con el que trabaja la red y lo que permite entrenarla en un portátil. Esta reducción tuvo una consecuencia que solo se descubrió después, con las fotografías propias, y que se cuenta en el [hallazgo 2](#2-el-problema-no-estaba-en-el-modelo-estaba-en-cómo-le-entregábamos-las-fotos).

**Cada vez que el modelo ve una foto de estudio, la ve un poco distinta.** Se voltea, se gira hasta 36°, se acerca o aleja un 10 %, se le cambia el brillo y el contraste hasta un 15 % y se desplaza ligeramente. Es lo que se llama *aumento de datos*, y tiene dos razones:

- **Una grieta no tiene «arriba».** Volteada o girada sigue siendo una grieta, y el modelo debe aprender que lo que importa es su forma, no su orientación ni la luz con que se fotografió.
- **Multiplica el material sin fotografiar nada nuevo.** Como las alteraciones son aleatorias, el modelo prácticamente nunca ve dos veces la misma imagen exacta, y eso le dificulta memorizar.

Estas alteraciones **se aplican solo a las fotos de estudio**. Las de validación y prueba se usan tal cual: el examen debe parecerse a la realidad, no ser más fácil ni más difícil que ella.

### 3. Dos modelos, dos estrategias

Se entrenaron dos redes muy distintas, para poder comparar.

**La línea base: aprender a ver desde cero.** Una red pequeña diseñada por el equipo, con **28 145 parámetros** (los ajustes internos que la red va corrigiendo). Tiene tres bloques que detectan rasgos cada vez más complejos —primero bordes, luego texturas, luego formas— y empieza sin saber absolutamente nada. Sirve de vara de medir: si una técnica más sofisticada no la supera, no vale la pena.

**MobileNetV2: partir de un ojo ya entrenado.** Una red publicada por Google que ya fue entrenada con **1.2 millones de fotografías cotidianas** —animales, vehículos, muebles—. No sabe nada de grietas, pero ya sabe *ver*: reconoce bordes, texturas y contrastes. Es como enseñarle a detectar grietas a alguien que ya sabe mirar, en lugar de a un recién nacido. A esto se le llama *transfer learning*, y se hizo en dos etapas:

| Etapa | Qué se entrena | Velocidad de aprendizaje | Vueltas |
|---|---|---|---|
| **1. Red congelada** | Solo una pieza nueva al final, que traduce lo que la red ve en «grieta» o «no grieta». La red original no se toca | Normal | 10 |
| **2. Ajuste fino** | Además, las últimas 19 capas de la red original | **100 veces más lenta** | 5 |

¿Por qué tan lenta en la segunda etapa? Porque esas capas ya contienen un conocimiento valioso. A la velocidad normal se sobrescribirían en pocas vueltas y se perdería lo que costó 1.2 millones de imágenes aprender. **Es retocar, no reescribir.**

### 4. Tres reglas que vigilan el aprendizaje

Una *vuelta* (en la jerga, una *época*) es una pasada completa por las 40 696 fotos de estudio. Al terminar cada una, el modelo hace un simulacro con las de validación, y tres reglas automáticas deciden qué pasa después:

1. **Si el simulacro no mejora en dos vueltas seguidas, se aprende más despacio.** La velocidad de aprendizaje se reduce a la mitad, como un estudiante que, al estancarse, pasa a repasar con más detalle.
2. **Si no mejora en cinco vueltas, se detiene.** Seguir solo serviría para memorizar las fotos de estudio, no para aprender a reconocer grietas nuevas.
3. **Se conserva la mejor vuelta, no la última.**

### 5. Cómo fue, vuelta a vuelta

**Línea base.** Las cifras son del simulacro: el *error* mide cuánto se equivoca el modelo (cuanto más bajo, mejor) y el *acierto*, el porcentaje de fotos que clasifica bien.

| Vuelta | Error | Acierto | Qué pasó |
|---|---|---|---|
| 1 | 0.39 | 89.4 % | |
| 2 | 7.83 | 41.1 % | **Se desploma** |
| 3 | 32.14 | 41.0 % | Sigue desplomado; la regla 1 baja la velocidad a la mitad |
| 4 | 0.17 | 93.5 % | Se recupera |
| 7 | 0.14 | 94.7 % | La velocidad se vuelve a reducir |
| **12** | **0.13** | **95.1 %** | **La mejor vuelta: es el modelo que se guarda** |
| 17 | 0.14 | 95.2 % | Cinco vueltas sin mejorar: la regla 2 lo detiene |

Las vueltas 2 y 3 merecen una explicación, porque se dejaron a la vista a propósito. El acierto cae al **41 %, que es exactamente la proporción de fotos con grieta**: durante esas dos vueltas el modelo respondía «grieta» a todo. No es un error de programación, sino la inestabilidad típica de una red que parte de cero con una velocidad de aprendizaje alta. La recuperación coincide con la reducción de velocidad de la regla 1, y **la regla 3 es la que garantiza que un tropiezo así nunca llegue al modelo final**: lo que se guarda es la vuelta 12.

**MobileNetV2.**

| Vuelta | Etapa | Error | Acierto | Qué pasó |
|---|---|---|---|---|
| 1 | Congelada | 0.14 | 96.0 % | **Ya supera a la mejor vuelta de la línea base** |
| 5 | Congelada | 0.12 | 96.1 % | |
| 9 | Congelada | 0.11 | 96.2 % | Mejor vuelta de la primera etapa |
| 11 | Ajuste fino | 0.10 | 96.3 % | Empieza el ajuste, a velocidad cien veces menor |
| **14** | **Ajuste fino** | **0.09** | **96.8 %** | **La mejor vuelta: es el modelo que se guarda** |
| 15 | Ajuste fino | 0.11 | 96.5 % | Fin de la segunda etapa |

La primera fila es la que resume el transfer learning: **en su primera vuelta, MobileNetV2 ya acierta más que la línea base después de doce**. Eso es lo que vale partir de un ojo ya entrenado.

El ajuste fino, medido sobre las 8 721 fotos de validación:

| | Tras la etapa 1 | Tras el ajuste fino | Cambio |
|---|---|---|---|
| Grietas que se le escapan | 284 | **255** | 10 % menos |
| Falsas alarmas | 49 | **26** | 47 % menos |

> **Una advertencia sobre estas cifras.** El grupo de validación tiene el mismo defecto que el [hallazgo 1](#1-el-conjunto-de-datos-público-estaba-contaminado) describe para el de prueba: **una de cada seis fotos de validación ya estaba entre las de estudio**. Por eso estos aciertos sirven para comparar vueltas y etapas entre sí —todas tienen la misma ventaja—, pero no como medida del acierto real. Ese está en [Resultados medidos](#resultados-medidos), calculado después de retirar los duplicados.

### 6. Comprimirlo para que quepa en un teléfono

El modelo entrenado ocupa **13.90 MB**. Para llevarlo a un teléfono se comprimió en dos pasos:

1. **Quitarle lo que solo sirve para entrenar**, como las notas internas que la red lleva sobre cómo ha ido corrigiéndose: pasa a **5.46 MB**.
2. **Cuantizarlo**: guardar cada uno de sus números con 8 bits en lugar de 32. Es parecido a redondear los precios a pesos enteros: se pierde algo de detalle, pero cada número ocupa la cuarta parte. Para decidir cómo redondear sin estropear el modelo se usan **200 fotografías de calibración**. Resultado: **1.74 MB**.

En total, **8 veces más pequeño y 38.5 veces más rápido**, a cambio de una pérdida pequeña de acierto que el informe cuantifica en §3.1. Es la versión que usa el modo de vídeo en vivo.

### 7. Cuánto costó

Todo el entrenamiento se hizo en el procesador de un portátil corriente, **sin tarjeta gráfica**:

| Modelo | Vueltas | Tiempo por vuelta | Total |
|---|---|---|---|
| Línea base | 17 | 7.3 min | 2 h 3 min |
| MobileNetV2, etapa 1 | 10 | 7.9 min | 1 h 19 min |
| MobileNetV2, ajuste fino | 5 | 8.5 min | 42 min |
| **Total** | | | **4 h 5 min** |

Un resultado que sorprendió: la línea base, con **52 veces menos parámetros**, tardó por vuelta **solo un 8 % menos** que MobileNetV2. El tiempo no se iba en el cálculo de la red sino en leer y descomprimir las 40 696 fotografías del disco en cada vuelta. Antes de medirlo, el equipo suponía lo contrario.

**Para ver más:** las curvas completas de las dos redes están en la pestaña **📊 Métricas del modelo** de la aplicación; el análisis detallado, en §3.1 y §3.3 de [`reports/analisis.md`](reports/analisis.md); y los comandos para repetir el entrenamiento, en [Reconstruirlo todo desde cero](#5-reconstruirlo-todo-desde-cero).

---

## Resultados medidos

### Detección de grietas

| Qué se midió | Resultado | Qué significa |
|---|---|---|
| **Acierto general (F1)** | **0.9423** | Resume en un número cuántas grietas encuentra y cuántas veces se equivoca al avisar. Va de 0 a 1 |
| Grietas que sí detecta | 90.7 % | De cada 100 grietas reales, encuentra unas 91 |
| Tiempo por fotografía | **4.72 milisegundos** | Unas 212 fotografías por segundo, en un computador corriente sin tarjeta gráfica |
| Tamaño del modelo | **1.74 MB** | Cabe holgadamente en un teléfono |

Ese modelo comprimido para teléfono es **38.5 veces más rápido y 8 veces más pequeño** que la versión original, sin perder acierto de forma apreciable.

### Medición de la grieta

Con un marcador impreso de tamaño conocido junto a la fisura, el sistema pasa de «hay grieta» a:

> *Fisura ramificada de **46 cm** de recorrido, con un ancho medio de **0.85 mm** y hasta **3.05 mm** en el punto más abierto.*

El ancho es la magnitud con la que la NSR-10 gradúa el daño, y era **la limitación más grave que el propio informe declaraba** (§5.1).

| Qué se midió | Resultado |
|---|---|
| Error de la escala | **0.4 %** sobre el lado del marcador |
| Incertidumbre del ancho | El tamaño de un píxel, que se muestra junto a cada cifra |
| Tiempo por fotografía | 0.06 a 0.9 s según el tamaño de la grieta |

**Sin marcador no hay milímetros, y el sistema lo dice** en vez de estimarlos. La escala de una fotografía no está en la fotografía: una fisura de 3 px puede tener 0.1 mm o 5 mm según a qué distancia se disparó.

### Medición de inclinación

| Qué se midió | Resultado |
|---|---|
| Error al medir el ángulo | **0.039°** — trece veces mejor que el criterio de aceptación |
| Fotografías en las que consigue medir | **1 de cada 40** |

La segunda cifra es incómoda y se reporta igual. El módulo es **muy preciso cuando funciona, y funciona pocas veces**: necesita ver el borde vertical completo de una columna o un muro, y la mayoría de las fotografías son primeros planos de la superficie. Reportar solo el 0.039° habría sido contar media historia.

---

## Lo que se descubrió al medir

Esta es la parte del trabajo que el equipo considera más valiosa. Son cinco hallazgos, y **dos de ellos son negativos**: mejoras que parecían buenas y que las mediciones obligaron a descartar.

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

### 5. Para medir hizo falta meter la escala en la escena

El sistema ya extraía la geometría de la fisura en píxeles. Convertirla a milímetros parecía el paso pequeño, y resultó ser el que no se puede programar: **la escala de una fotografía no está en la fotografía**.

La solución es física: se imprime un marcador de 5 cm y se pega junto a la grieta. La demostración de que funciona es que **la medida en milímetros no cambia al alejar la cámara**, aunque la medida en píxeles se reduzca a la mitad.

Por el camino aparecieron diez errores que no lanzaban ninguna excepción y producían números con dos decimales: tortuosidades geométricamente imposibles, 432 «ramificaciones» en una grieta única, un filtro que descartaba justamente las grietas diagonales, y un sesgo de ancho de exactamente +3 píxeles que resultó ser una constante y por tanto corregible. Están documentados uno a uno en §4.10 del informe.

**Cuatro de ellos los encontró el usuario mirando la pantalla, no las pruebas.** Las pruebas sintéticas verificaban que el algoritmo medía bien *lo que se le daba*; ninguna comprobaba que se le estuviera dando lo correcto.

### El patrón que une los cinco

En los cinco casos la conclusión salió de **medir antes de dar algo por bueno**: comparar contra la alternativa aburrida, o mirar una cifra elemental que nadie había mirado. Ninguno necesitó una técnica avanzada.

---

## La aplicación

Se ejecuta en el navegador y tiene cuatro pestañas.

**🔍 Análisis en vivo.** Se carga una fotografía y se obtiene el resultado completo: probabilidad de grieta, ángulo de inclinación, orientación de la fisura y el semáforo de riesgo con las reglas que lo justifican. Muestra la imagen original y la procesada lado a lado, y señala con recuadros en qué zona encontró la evidencia.

**📹 Cámara en vivo.** Analiza en tiempo real lo que ve la cámara, con el vídeo y el veredicto de riesgo uno al lado del otro. Puede usar la cámara del computador o **la del teléfono a través de la red local**, que es lo práctico para caminar por una edificación. Un botón **Medir** congela el fotograma y extrae la geometría de la fisura.

Dentro de esta pestaña, el modo **Foto a foto** analiza una sola imagen a fondo, y ofrece tres orígenes:

| Origen | Dónde funciona |
|---|---|
| **Subir foto** | En el celular, el selector de archivos ofrece *Hacer foto* y abre la cámara del sistema. Es la vía con mejor resolución |
| **Cámara de este dispositivo** | Solo en el computador |
| **Capturar del stream** | Un fotograma de la cámara IP del teléfono |

> **Por qué la cámara del navegador no sirve desde el celular.** Los navegadores solo dan acceso a la cámara en contexto seguro —HTTPS o localhost—, y la aplicación se sirve por HTTP en la red local. Por eso desde el teléfono hay que usar *Subir foto*, que abre la cámara del sistema y no pasa por esa restricción.

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

**Sobre el equilibrio entre clases.** El conjunto público tiene más paredes sanas que agrietadas, en proporción **1.44 a 1** (23 851 con grieta frente a 34 287 sin ella). No se corrige eliminando imágenes, sino **ponderando el error durante el entrenamiento**: equivocarse en una grieta cuesta 1.44 veces más que dar una falsa alarma. Así se compensa el desequilibrio sin renunciar a 10 436 fotografías reales. El razonamiento completo está en §1.5 del informe.

Las fotografías propias son **60 imágenes tomadas por el equipo** y cumplen una función que el dataset público no puede cumplir: son el único material que el modelo no ha visto en ninguna forma. **Nunca se usan para entrenar ni para tomar decisiones de diseño** — en cuanto se usaran para decidir algo dejarían de ser una medida honesta.

### 3. Imprimir el marcador de escala (opcional, pero es lo que da los milímetros)

```bash
python scripts/generar_marcador.py --lado-mm 50
```

Genera `reports/marcador_escala.pdf` y su versión PNG. **Imprime el PDF**, no el PNG: una imagen no lleva dentro su tamaño físico y cada programa la escala como quiere — la app de Fotos de Windows, por ejemplo, solo ofrece «rellenar página» y «ajustar a la página», y ambas deforman la escala.

1. Abre el PDF y elige **Tamaño real** o **Escala 100 %**.
2. Mide con una regla la escalita impresa en la hoja: debe marcar 50 mm exactos.
3. Si marca otra cosa, **no reimprimas**: mide el lado negro del cuadrado y escribe ese valor en `config.yaml` → `escala.lado_marcador_mm`. Lo que el sistema necesita no es que mida 50 mm, sino que la configuración diga la verdad sobre el marcador que tienes en la mano.
4. **Si lo recortas, deja 1 cm de blanco alrededor.** Un borde fino de 1 a 3 mm hace medir hasta un 14 % de más — y siempre hacia arriba, lo que reportaría las grietas más estrechas de lo que son.

Al fotografiar: pega la hoja **junto a la grieta y en su mismo plano**, y sitúate **de frente** a la pared. Como el marcador es cuadrado, si lo fotografías en ángulo sale como un rombo y el sistema lo detecta y avisa.

### 4. Ejecutar la aplicación

La aplicación necesita los modelos entrenados en la carpeta `models/`. No están dentro del repositorio por su tamaño, pero se publicaron aparte en el [release `v1.0-modelos`](https://github.com/elenanito1216/proyecto_algoritmos/releases/tag/v1.0-modelos), así que **no hace falta entrenar nada**: se descargan en segundos.

**En Windows (PowerShell)**, desde la carpeta raíz del proyecto:

```powershell
foreach ($f in "mobilenetv2_finetuned.keras","mobilenetv2_int8.tflite","baseline_cnn.keras") { Invoke-WebRequest "https://github.com/elenanito1216/proyecto_algoritmos/releases/download/v1.0-modelos/$f" -OutFile "models\$f" }
```

**En Linux o macOS**

```bash
for f in mobilenetv2_finetuned.keras mobilenetv2_int8.tflite baseline_cnn.keras; do curl -L -o "models/$f" "https://github.com/elenanito1216/proyecto_algoritmos/releases/download/v1.0-modelos/$f"; done
```

| Archivo | Tamaño | Lo usa |
|---|---|---|
| `mobilenetv2_finetuned.keras` | 14.6 MB | El modo fotografía |
| `mobilenetv2_int8.tflite` | 1.8 MB | El modo video en vivo |
| `baseline_cnn.keras` | 0.4 MB | La evaluación comparativa (`scripts/evaluate.py`) |

Con los modelos en su sitio, se abre la aplicación:

```bash
streamlit run app/app.py
```

Quien prefiera entrenarlos por su cuenta puede hacerlo con los pasos del apartado siguiente.

### 5. Reconstruirlo todo desde cero

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

### 6. Reproducir los experimentos del análisis

Cada hallazgo de la sección anterior tiene su propio programa, y cada uno puede ejecutarse por separado:

| Hallazgo | Comando |
|---|---|
| Imágenes repetidas entre estudio y examen | `python scripts/analizar_fuga_datos.py --config config.yaml` |
| Análisis por ventanas | `python scripts/evaluar_mosaicos.py --config config.yaml` |
| Ajuste del nivel de corte | `python scripts/calibrar_umbral.py --config config.yaml` |
| Medición de la inclinación | `python scripts/validar_inclinacion.py --config config.yaml --imagenes data/propias` |
| Hoja del marcador de escala | `python scripts/generar_marcador.py --lado-mm 50` |

### 7. Ejecutar las pruebas automáticas

```bash
pytest tests -v
```

Son **232 pruebas** que verifican el comportamiento del sistema:

| Archivo | Pruebas | Qué comprueba |
|---|---|---|
| `test_inclinacion.py` | 45 | Medición de ángulos y sus casos límite |
| `test_camara.py` | 48 | Vídeo en tiempo real, orientación y presentación |
| `test_reglas.py` | 33 | Motor de riesgo, incluida la coherencia entre niveles |
| `test_mosaicos.py` | 29 | Troceado de imágenes y elección automática de modelo |
| `test_morfologia.py` | 39 | Geometría de la fisura sobre figuras de tamaño conocido |
| `test_escala.py` | 20 | Conversión a milímetros, de ida y vuelta |
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
├── tests/              ← las 232 pruebas automáticas
├── notebooks/          ← exploración inicial de los datos
├── reports/            ← el informe y las cifras medidas
│   ├── analisis.md     ·  ANÁLISIS CRÍTICO COMPLETO
│   ├── metricas/       ·  cifras en bruto que la aplicación lee
│   └── figuras/        ·  gráficas
│
├── models/             ← modelos entrenados (se descargan del release v1.0-modelos)
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

Sobre la medición: el sistema **sí mide el ancho de la fisura cuando hay un marcador de tamaño conocido en la escena**, y muestra la incertidumbre junto a la cifra. Sin marcador reporta píxeles y lo advierte, porque ese dato no está en la imagen.

Pero medir no es dictaminar. Un ingeniero valora además el elemento, su función estructural y el patrón de fisuración del conjunto. El sistema dice *dónde mirar y cuánto mide*, no *qué concluir*.

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
| **Marcador ArUco** | Cuadrado en blanco y negro con un código dentro. Impreso a tamaño conocido, permite convertir píxeles a milímetros |
| **Tortuosidad** | Cuánto serpentea la grieta: su recorrido dividido por la distancia entre sus extremos. Vale 1 si es recta |
| **Segmentar** | Separar en la imagen los píxeles que son grieta de los que son pared |
| **Época** | Una vuelta completa del entrenamiento a todas las imágenes de estudio |
| **Aumento de datos** | Mostrarle al modelo cada foto de estudio ligeramente alterada —volteada, girada, con otra luz— para que aprenda la forma de la grieta y no la foto concreta |
| **Validación** | Fotografías apartadas que el modelo no estudia y que se usan como simulacro al final de cada época |
| **Parámetros** | Los ajustes internos de la red que el entrenamiento va corrigiendo. Son los que guardan lo aprendido |

---

## Equipo

Proyecto desarrollado por Santiago Gómez García y equipo para la asignatura *Algoritmos y Programación*, Ingeniería en Inteligencia Artificial, Universidad Industrial de Santander, semestre 2026-2.
