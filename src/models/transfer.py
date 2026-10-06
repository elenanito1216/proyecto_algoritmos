"""Transfer learning con MobileNetV2.

Sobre la eleccion de arquitectura
---------------------------------
El enunciado contemplaba EfficientNet-Lite. **No esta disponible en
``keras.applications``**: la familia Lite solo existe como modelo de TF Hub o
via TFLite Model Maker, no como constructor nativo de Keras. Forzarlo anadiria
una dependencia externa y un formato de pesos que complica la exportacion a
TFLite, asi que se usa **MobileNetV2**, que es el estandar de facto para
inferencia en dispositivos moviles, esta incluido en Keras y cuantiza a int8 sin
sorpresas. Esta decision queda documentada tambien en ``reports/analisis.md``.

Sobre ``alpha`` (width multiplier)
----------------------------------
``alpha=0.75`` reduce el numero de canales de cada capa al 75%, recortando los
parametros de la base de ~2.26 M a ~1.38 M (-39%) y las multiplicaciones-
acumulaciones en proporcion aproximadamente cuadratica. Para un problema binario
de textura como "hay grieta o no", la capacidad sobrante de ``alpha=1.0`` no se
aprovecha y si se paga en latencia. Los pesos ImageNet existen para
``alpha in {0.35, 0.5, 0.75, 1.0, 1.3, 1.4}`` y resoluciones
``{96, 128, 160, 192, 224}``, por eso ``config.yaml`` fija 160x160.
"""

from __future__ import annotations

from typing import Any

from src.utils.config import forma_entrada, obtener


def construir_transfer(config: dict[str, Any], base_entrenable: bool = False) -> tuple[Any, Any]:
    """Construye el modelo MobileNetV2 con cabeza densa personalizada.

    Estructura: ``Entrada [0,1] -> Rescaling a [-1,1] -> MobileNetV2 (base) ->
    GlobalAveragePooling2D -> Dropout -> Densa ReLU -> Dropout -> Sigmoide``.

    La capa de reescalado va **dentro** del modelo a proposito: asi el artefacto
    exportado (``.keras`` o ``.tflite``) acepta directamente imagenes en ``[0,1]``
    y la aplicacion Streamlit no tiene que recordar cual es el preprocesado
    exacto de cada arquitectura. Un modelo que encapsula su propio preprocesado
    es un modelo que no se puede usar mal.

    Args:
        config: Configuracion del proyecto.
        base_entrenable: Si ``False`` (etapa 1) la base queda congelada y solo
            se entrena la cabeza. ``True`` se usa desde el script de
            fine-tuning, que ademas selecciona cuantas capas descongelar.

    Returns:
        Tupla ``(modelo, base)``. Se devuelve tambien la base para que el script
        de fine-tuning pueda descongelar sus ultimas capas sin buscarla por
        nombre.
    """
    import tensorflow as tf

    alto, ancho, canales = forma_entrada(config)
    alpha = float(obtener(config, "transfer.alpha", 0.75))
    pesos = obtener(config, "transfer.pesos", "imagenet")
    dropout = float(obtener(config, "transfer.dropout", 0.3))
    unidades = int(obtener(config, "transfer.unidades_densa", 64))
    nombre = str(obtener(config, "transfer.nombre", "mobilenetv2"))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(alto, ancho, canales),
        alpha=alpha,
        include_top=False,
        weights=pesos,
    )
    base.trainable = base_entrenable

    entradas = tf.keras.Input(shape=(alto, ancho, canales), name="imagen")
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="preproceso_mobilenet")(entradas)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout1")(x)
    x = tf.keras.layers.Dense(unidades, activation="relu", name="densa")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout2")(x)
    salidas = tf.keras.layers.Dense(1, activation="sigmoid", name="p_grieta")(x)

    modelo = tf.keras.Model(entradas, salidas, name=nombre)
    return modelo, base


def descongelar_ultimas_capas(base: Any, n_capas: int) -> int:
    """Habilita el entrenamiento de las ultimas ``n_capas`` de la base.

    Las capas de ``BatchNormalization`` se dejan congeladas de forma explicita:
    con lotes pequenos (32 en CPU) sus estadisticas moviles se vuelven ruidosas y
    degradan el modelo en lugar de mejorarlo. Es la practica recomendada en el
    fine-tuning de redes moviles.

    Args:
        base: Modelo base preentrenado.
        n_capas: Cuantas capas finales descongelar.

    Returns:
        Numero real de capas que quedaron entrenables.
    """
    import tensorflow as tf

    base.trainable = True
    corte = max(len(base.layers) - n_capas, 0)
    entrenables = 0
    for indice, capa in enumerate(base.layers):
        if indice < corte or isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False
        else:
            capa.trainable = True
            entrenables += 1
    return entrenables


def resumen_parametros(modelo: Any) -> dict[str, int]:
    """Cuenta los parametros del modelo separando entrenables de congelados.

    Args:
        modelo: Modelo de Keras construido.

    Returns:
        Diccionario con ``total``, ``entrenables`` y ``no_entrenables``.
    """
    entrenables = int(sum(int(v.numpy().size) for v in modelo.trainable_weights))
    no_entrenables = int(sum(int(v.numpy().size) for v in modelo.non_trainable_weights))
    return {
        "total": entrenables + no_entrenables,
        "entrenables": entrenables,
        "no_entrenables": no_entrenables,
    }
