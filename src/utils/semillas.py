"""Fijacion de semillas para reproducibilidad.

Punto unico de control de la aleatoriedad del proyecto. Se invoca al principio
de cada script; ningun otro modulo debe llamar a ``seed`` por su cuenta, porque
sembrar dos veces en mitad de una ejecucion rompe la trazabilidad de los
resultados.
"""

from __future__ import annotations

import os
import random


def fijar_semillas(semilla: int = 42, determinismo_estricto: bool = False) -> None:
    """Siembra los generadores de numeros aleatorios de Python, NumPy y TensorFlow.

    Args:
        semilla: Valor de la semilla global.
        determinismo_estricto: Si es ``True``, activa las operaciones
            deterministas de TensorFlow. Garantiza resultados identicos entre
            corridas, pero puede reducir el rendimiento de forma notable porque
            desactiva kernels paralelos no deterministas. Se recomienda ``True``
            solo para reproducir un resultado publicado, no para explorar.

    Note:
        ``PYTHONHASHSEED`` debe estar fijado antes de que arranque el interprete
        para afectar al hash de cadenas. Se establece aqui igualmente porque si
        TensorFlow lanza subprocesos, estos heredaran el valor.
    """
    os.environ["PYTHONHASHSEED"] = str(semilla)

    random.seed(semilla)

    import numpy as np

    np.random.seed(semilla)

    import tensorflow as tf

    tf.random.set_seed(semilla)
    tf.keras.utils.set_random_seed(semilla)

    if determinismo_estricto:
        tf.config.experimental.enable_op_determinism()
