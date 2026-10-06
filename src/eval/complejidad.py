"""Analisis de complejidad computacional y costo de inferencia.

La asignatura es de Algoritmos: el desempeno predictivo por si solo no basta,
hay que cuantificar lo que cuesta obtenerlo. Este modulo mide las cuatro
magnitudes que deciden si un modelo cabe en un telefono:

1. Numero de parametros (entrenables y congelados).
2. Tamano del artefacto en disco (``.keras``, ``.tflite`` float32, ``.tflite`` int8).
3. Latencia por imagen: media y desviacion estandar sobre >= 50 corridas,
   descartando las de calentamiento.
4. Pico de memoria durante la inferencia.

Complejidad asintotica del pipeline de inferencia
-------------------------------------------------
Para una CNN, el costo dominante son las convoluciones. Una capa convolucional
con entrada ``H x W x C_in``, kernel ``k x k`` y ``C_out`` filtros cuesta
``O(H * W * k^2 * C_in * C_out)`` multiplicaciones-acumulaciones (MACs).

- El costo es **cuadratico en la resolucion**: pasar de 224x224 a 160x160 lo
  reduce a ``(160/224)^2 = 0.51``, casi la mitad. Esta es la razon principal de
  la eleccion de 160 en ``config.yaml``.
- MobileNetV2 sustituye la convolucion estandar por una **separable en
  profundidad**: primero ``k x k`` por canal (``H*W*k^2*C_in``) y luego ``1x1``
  entre canales (``H*W*C_in*C_out``). El cociente frente a la convolucion densa
  es ``1/C_out + 1/k^2``, es decir, ~8-9 veces menos operaciones con ``k=3``.
  Ahi esta el grueso de su eficiencia, no en tener menos capas.
- El ``alpha`` (width multiplier) escala ``C_in`` y ``C_out`` a la vez, asi que
  el costo cae aproximadamente con ``alpha^2``: ``alpha=0.75`` deja el computo en
  ~56% del original.
- La cuantizacion a int8 **no cambia el orden asintotico**, cambia la constante:
  operaciones enteras de 8 bits en lugar de flotantes de 32, y 4 veces menos
  ancho de banda de memoria. En CPU movil eso suele traducirse en 2-3x de
  aceleracion real.

El pipeline clasico (Canny + Hough) es lineal en el numero de pixeles; su
analisis detallado esta en ``src/vision/inclinacion.py``.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.rutas import resolver

BYTES_POR_MB = 1024 * 1024


def contar_parametros(modelo: Any) -> dict[str, int]:
    """Cuenta los parametros del modelo separando entrenables de congelados.

    Args:
        modelo: Modelo de Keras.

    Returns:
        Diccionario con ``total``, ``entrenables`` y ``no_entrenables``.
    """
    entrenables = int(sum(int(np.prod(v.shape)) for v in modelo.trainable_weights))
    no_entrenables = int(sum(int(np.prod(v.shape)) for v in modelo.non_trainable_weights))
    return {
        "total": entrenables + no_entrenables,
        "entrenables": entrenables,
        "no_entrenables": no_entrenables,
    }


def tamano_en_disco_mb(ruta: str | Path) -> float | None:
    """Mide el tamano de un artefacto en megabytes.

    Args:
        ruta: Ruta al archivo (o directorio SavedModel), relativa o absoluta.

    Returns:
        Tamano en MB, o ``None`` si la ruta no existe.
    """
    destino = resolver(ruta)
    if destino.is_file():
        return round(destino.stat().st_size / BYTES_POR_MB, 4)
    if destino.is_dir():
        total = sum(p.stat().st_size for p in destino.rglob("*") if p.is_file())
        return round(total / BYTES_POR_MB, 4)
    return None


def _estadisticos_latencia(tiempos_s: list[float]) -> dict[str, float]:
    """Resume una serie de tiempos de inferencia.

    Args:
        tiempos_s: Duraciones individuales en segundos.

    Returns:
        Diccionario con media, desviacion estandar, mediana, percentil 95,
        minimo y maximo, todo en milisegundos, mas ``imagenes_por_segundo``.
    """
    arreglo = np.asarray(tiempos_s, dtype=float) * 1000.0
    media = float(np.mean(arreglo))
    return {
        "latencia_media_ms": round(media, 3),
        "latencia_std_ms": round(float(np.std(arreglo, ddof=1)) if len(arreglo) > 1 else 0.0, 3),
        "latencia_mediana_ms": round(float(np.median(arreglo)), 3),
        "latencia_p95_ms": round(float(np.percentile(arreglo, 95)), 3),
        "latencia_min_ms": round(float(np.min(arreglo)), 3),
        "latencia_max_ms": round(float(np.max(arreglo)), 3),
        "imagenes_por_segundo": round(1000.0 / media, 2) if media > 0 else 0.0,
        "n_mediciones": int(len(arreglo)),
    }


def medir_latencia_keras(
    modelo: Any,
    forma_entrada: tuple[int, int, int],
    repeticiones: int = 60,
    calentamiento: int = 10,
    batch: int = 1,
    semilla: int = 42,
) -> dict[str, float]:
    """Mide la latencia de inferencia de un modelo Keras.

    Se descartan las primeras ``calentamiento`` corridas porque la primera
    llamada incluye la construccion del grafo, la reserva de buffers y el
    llenado de las caches de la CPU: incluirlas contaminaria la media con un
    valor atipico de varios cientos de milisegundos.

    Se usa ``modelo(x, training=False)`` en lugar de ``modelo.predict``: este
    ultimo anade la maquinaria de callbacks y el troceado en lotes, que no forma
    parte del costo de inferencia que se quiere medir.

    Args:
        modelo: Modelo de Keras.
        forma_entrada: Forma ``(alto, ancho, canales)`` de una imagen.
        repeticiones: Numero de mediciones validas (el enunciado exige >= 50).
        calentamiento: Corridas previas descartadas.
        batch: Tamano de lote. 1 reproduce el escenario real de la aplicacion.
        semilla: Semilla para la entrada sintetica.

    Returns:
        Diccionario de estadisticos devuelto por :func:`_estadisticos_latencia`,
        mas ``batch``.
    """
    import tensorflow as tf

    generador = np.random.default_rng(semilla)
    entrada = tf.constant(
        generador.random((batch, *forma_entrada), dtype=np.float32), dtype=tf.float32
    )

    for _ in range(calentamiento):
        modelo(entrada, training=False)

    tiempos: list[float] = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        modelo(entrada, training=False)
        tiempos.append((time.perf_counter() - inicio) / batch)

    resultado = _estadisticos_latencia(tiempos)
    resultado["batch"] = batch
    return resultado


def medir_latencia_tflite(
    ruta_modelo: str | Path,
    repeticiones: int = 60,
    calentamiento: int = 10,
    n_hilos: int = 1,
    semilla: int = 42,
) -> dict[str, Any]:
    """Mide la latencia de un modelo TFLite con el interprete de referencia.

    Se fija ``n_hilos=1`` por defecto para emular el escenario conservador de un
    telefono de gama baja. Medir con todos los hilos de un portatil daria una
    cifra optimista que no representa el dispositivo objetivo.

    Args:
        ruta_modelo: Ruta al archivo ``.tflite``.
        repeticiones: Numero de mediciones validas.
        calentamiento: Corridas previas descartadas.
        n_hilos: Hilos del interprete.
        semilla: Semilla para la entrada sintetica.

    Returns:
        Diccionario de estadisticos mas ``tipo_entrada``, ``forma_entrada`` y
        ``n_hilos``.

    Raises:
        FileNotFoundError: Si el archivo ``.tflite`` no existe.
    """
    import tensorflow as tf

    destino = resolver(ruta_modelo)
    if not destino.is_file():
        raise FileNotFoundError(f"No existe el modelo TFLite: {destino}")

    interprete = tf.lite.Interpreter(model_path=str(destino), num_threads=n_hilos)
    interprete.allocate_tensors()
    detalle_entrada = interprete.get_input_details()[0]

    forma = tuple(int(v) for v in detalle_entrada["shape"])
    dtype = detalle_entrada["dtype"]
    generador = np.random.default_rng(semilla)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        entrada = generador.integers(info.min, info.max, size=forma, dtype=dtype)
    else:
        entrada = generador.random(forma).astype(dtype)

    indice = detalle_entrada["index"]
    salida = interprete.get_output_details()[0]["index"]

    for _ in range(calentamiento):
        interprete.set_tensor(indice, entrada)
        interprete.invoke()
        interprete.get_tensor(salida)

    tiempos: list[float] = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        interprete.set_tensor(indice, entrada)
        interprete.invoke()
        interprete.get_tensor(salida)
        tiempos.append(time.perf_counter() - inicio)

    resultado: dict[str, Any] = _estadisticos_latencia(tiempos)
    resultado["tipo_entrada"] = str(np.dtype(dtype))
    resultado["forma_entrada"] = list(forma)
    resultado["n_hilos"] = n_hilos
    return resultado


def medir_pico_memoria(funcion_inferencia: Any, repeticiones: int = 20) -> dict[str, float]:
    """Mide el incremento de memoria residente durante la inferencia.

    Se mide el RSS del proceso con ``psutil`` en lugar de ``tracemalloc`` porque
    el grueso de la memoria de TensorFlow se reserva en buffers de C++ que
    ``tracemalloc`` (que solo ve el asignador de Python) no contabiliza.

    La cifra es un **incremento sobre la linea base**, no el consumo absoluto del
    proceso: lo que interesa es cuanta RAM extra pide una inferencia, que es lo
    que decide si el modelo cabe en un dispositivo con memoria limitada.

    Args:
        funcion_inferencia: Callable sin argumentos que ejecuta una inferencia.
        repeticiones: Cuantas inferencias ejecutar mientras se mide.

    Returns:
        Diccionario con ``memoria_base_mb``, ``memoria_pico_mb`` y
        ``incremento_mb``.
    """
    import psutil

    proceso = psutil.Process()
    gc.collect()
    base = proceso.memory_info().rss / BYTES_POR_MB

    pico = base
    for _ in range(repeticiones):
        funcion_inferencia()
        pico = max(pico, proceso.memory_info().rss / BYTES_POR_MB)

    return {
        "memoria_base_mb": round(base, 2),
        "memoria_pico_mb": round(pico, 2),
        "incremento_mb": round(pico - base, 2),
    }


def perfilar_modelo_keras(
    modelo: Any,
    forma_entrada: tuple[int, int, int],
    ruta_artefacto: str | Path | None,
    config: dict[str, Any],
    etiqueta: str,
) -> dict[str, Any]:
    """Perfila un modelo Keras completo: parametros, tamano, latencia y memoria.

    Args:
        modelo: Modelo de Keras cargado.
        forma_entrada: Forma ``(alto, ancho, canales)``.
        ruta_artefacto: Ruta del ``.keras`` en disco, o ``None`` si no se guardo.
        config: Configuracion del proyecto (seccion ``complejidad``).
        etiqueta: Nombre con el que aparecera en la tabla comparativa.

    Returns:
        Diccionario con todas las mediciones, listo para serializar a JSON.
    """
    import tensorflow as tf

    cfg = config.get("complejidad", {})
    repeticiones = int(cfg.get("repeticiones_latencia", 60))
    calentamiento = int(cfg.get("repeticiones_calentamiento", 10))
    batch = int(cfg.get("batch_inferencia", 1))

    entrada_fija = tf.constant(
        np.random.default_rng(0).random((batch, *forma_entrada), dtype=np.float32)
    )

    perfil: dict[str, Any] = {
        "etiqueta": etiqueta,
        "formato": "keras",
        "parametros": contar_parametros(modelo),
        "tamano_mb": tamano_en_disco_mb(ruta_artefacto) if ruta_artefacto else None,
        "latencia": medir_latencia_keras(modelo, forma_entrada, repeticiones, calentamiento, batch),
        "memoria": medir_pico_memoria(lambda: modelo(entrada_fija, training=False)),
    }
    return perfil


def perfilar_modelo_tflite(
    ruta_modelo: str | Path, config: dict[str, Any], etiqueta: str, n_hilos: int = 1
) -> dict[str, Any]:
    """Perfila un artefacto TFLite: tamano, latencia y memoria.

    Args:
        ruta_modelo: Ruta al ``.tflite``.
        config: Configuracion del proyecto (seccion ``complejidad``).
        etiqueta: Nombre con el que aparecera en la tabla comparativa.
        n_hilos: Hilos del interprete.

    Returns:
        Diccionario con las mediciones. TFLite no expone un conteo de parametros,
        por lo que ese campo queda en ``None``; el tamano en disco cumple ese
        papel.
    """
    import tensorflow as tf

    cfg = config.get("complejidad", {})
    destino = resolver(ruta_modelo)

    latencia = medir_latencia_tflite(
        destino,
        repeticiones=int(cfg.get("repeticiones_latencia", 60)),
        calentamiento=int(cfg.get("repeticiones_calentamiento", 10)),
        n_hilos=n_hilos,
    )

    interprete = tf.lite.Interpreter(model_path=str(destino), num_threads=n_hilos)
    interprete.allocate_tensors()
    detalle = interprete.get_input_details()[0]
    if np.issubdtype(detalle["dtype"], np.integer):
        info = np.iinfo(detalle["dtype"])
        muestra = np.random.default_rng(0).integers(
            info.min, info.max, size=detalle["shape"], dtype=detalle["dtype"]
        )
    else:
        muestra = np.random.default_rng(0).random(detalle["shape"]).astype(detalle["dtype"])

    def _inferir() -> None:
        interprete.set_tensor(detalle["index"], muestra)
        interprete.invoke()

    return {
        "etiqueta": etiqueta,
        "formato": "tflite",
        "parametros": None,
        "tamano_mb": tamano_en_disco_mb(destino),
        "latencia": latencia,
        "memoria": medir_pico_memoria(_inferir),
    }


def construir_tabla_comparativa(
    perfiles: list[dict[str, Any]], metricas_por_modelo: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Cruza desempeno contra costo en una unica tabla.

    Es la tabla que responde a la pregunta del proyecto: *cuanto desempeno se
    pierde al hacer el modelo lo bastante barato como para correr en un telefono*.

    Args:
        perfiles: Lista de perfiles de :func:`perfilar_modelo_keras` o
            :func:`perfilar_modelo_tflite`.
        metricas_por_modelo: Diccionario ``{etiqueta: metricas}`` con la salida
            de ``calcular_metricas`` para cada modelo.

    Returns:
        Lista de filas (diccionarios planos) lista para ``pandas.DataFrame`` o
        para una tabla de Plotly.
    """
    filas: list[dict[str, Any]] = []
    for perfil in perfiles:
        etiqueta = perfil["etiqueta"]
        metricas = metricas_por_modelo.get(etiqueta, {})
        parametros = perfil.get("parametros") or {}
        filas.append(
            {
                "modelo": etiqueta,
                "formato": perfil.get("formato"),
                "parametros_total": parametros.get("total"),
                "parametros_entrenables": parametros.get("entrenables"),
                "tamano_mb": perfil.get("tamano_mb"),
                "latencia_media_ms": perfil.get("latencia", {}).get("latencia_media_ms"),
                "latencia_std_ms": perfil.get("latencia", {}).get("latencia_std_ms"),
                "imagenes_por_segundo": perfil.get("latencia", {}).get("imagenes_por_segundo"),
                "memoria_incremento_mb": perfil.get("memoria", {}).get("incremento_mb"),
                "accuracy": metricas.get("accuracy"),
                "precision": metricas.get("precision"),
                "recall": metricas.get("recall"),
                "f1": metricas.get("f1"),
                "falsos_negativos": (metricas.get("matriz_confusion") or {}).get("fn"),
            }
        )
    return filas
