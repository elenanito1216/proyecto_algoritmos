"""Interfaz unificada de inferencia para modelos ``.keras`` y ``.tflite``.

La aplicacion Streamlit y el script de evaluacion necesitan exactamente lo mismo:
dar una imagen y recibir ``P(grieta)``, midiendo cuanto tardo. Sin esta capa, esa
logica se duplicaria en ambos sitios y, peor, se duplicaria el preprocesado, que
es donde se cometen los errores silenciosos (una normalizacion distinta entre
entrenamiento e inferencia degrada el modelo sin lanzar ningun error).

Ambos predictores exponen el mismo contrato, de modo que el selector
``.keras`` / ``.tflite`` de la interfaz es un cambio de una linea.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import forma_entrada, obtener
from src.utils.rutas import resolver


def preparar_imagen(imagen_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Redimensiona y normaliza una imagen para alimentar al modelo.

    Replica exactamente lo que hace ``src.data.loader._decodificar``: mismo
    tamano, misma interpolacion bilineal y mismo rango ``[0, 1]``. Cualquier
    divergencia aqui produciria un desplazamiento de dominio entre entrenamiento
    e inferencia.

    Args:
        imagen_rgb: Imagen RGB ``(H, W, 3)`` con valores enteros en ``[0, 255]``
            o flotantes en ``[0, 1]``.
        config: Configuracion del proyecto.

    Returns:
        Tensor ``(1, alto, ancho, 3)`` de tipo ``float32`` en el rango ``[0, 1]``.
        La dimension de lote se anade siempre: la forma del tensor es
        ``(N, H, W, C)`` con ``N = 1``.
    """
    import cv2

    alto, ancho, _ = forma_entrada(config)
    imagen = np.asarray(imagen_rgb)

    if imagen.ndim == 2:  # escala de grises -> replicar canales
        imagen = np.stack([imagen] * 3, axis=-1)
    if imagen.shape[-1] == 4:  # RGBA -> descartar alfa
        imagen = imagen[..., :3]

    if imagen.dtype != np.float32:
        imagen = imagen.astype(np.float32)
    if imagen.max() > 1.0:
        imagen = imagen / 255.0

    redimensionada = cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(redimensionada.astype(np.float32), axis=0)


class Predictor(ABC):
    """Contrato comun de los predictores de grietas.

    Attributes:
        ruta: Ruta del artefacto cargado.
        formato: ``"keras"`` o ``"tflite"``.
    """

    def __init__(self, ruta: Path, formato: str) -> None:
        self.ruta = ruta
        self.formato = formato

    @abstractmethod
    def _inferir(self, lote: np.ndarray) -> np.ndarray:
        """Ejecuta la inferencia sobre un lote ya preprocesado.

        Args:
            lote: Tensor ``(N, H, W, C)`` float32 en ``[0, 1]``.

        Returns:
            Vector ``(N,)`` de probabilidades.
        """

    @abstractmethod
    def descripcion(self) -> dict[str, Any]:
        """Devuelve metadatos del modelo para mostrarlos en la interfaz.

        Returns:
            Diccionario con al menos ``formato``, ``archivo`` y ``tamano_mb``.
        """

    def predecir(self, lote: np.ndarray) -> tuple[np.ndarray, float]:
        """Infiere y cronometra.

        Args:
            lote: Tensor ``(N, H, W, C)`` float32 en ``[0, 1]``.

        Returns:
            Tupla ``(probabilidades, milisegundos)``.
        """
        inicio = time.perf_counter()
        probabilidades = self._inferir(lote)
        return probabilidades, (time.perf_counter() - inicio) * 1000.0

    def predecir_imagen(
        self, imagen_rgb: np.ndarray, config: dict[str, Any]
    ) -> tuple[float, float]:
        """Infiere sobre una sola imagen sin preprocesar.

        Args:
            imagen_rgb: Imagen RGB ``(H, W, 3)``.
            config: Configuracion del proyecto.

        Returns:
            Tupla ``(probabilidad_grieta, milisegundos)``.
        """
        lote = preparar_imagen(imagen_rgb, config)
        probabilidades, ms = self.predecir(lote)
        return float(probabilidades[0]), ms


class PredictorKeras(Predictor):
    """Predictor respaldado por un modelo ``.keras`` cargado en memoria."""

    def __init__(self, ruta: Path) -> None:
        import tensorflow as tf

        super().__init__(ruta, "keras")
        self._modelo = tf.keras.models.load_model(ruta)

    @property
    def modelo(self) -> Any:
        """Modelo de Keras subyacente.

        Returns:
            El objeto ``keras.Model`` cargado.
        """
        return self._modelo

    def _inferir(self, lote: np.ndarray) -> np.ndarray:
        # Llamada directa en lugar de .predict(): evita la maquinaria de
        # callbacks y troceado, que no forma parte del costo real de inferir.
        salida = self._modelo(lote, training=False)
        return np.asarray(salida).reshape(-1)

    def descripcion(self) -> dict[str, Any]:
        """Metadatos del modelo Keras.

        Returns:
            Diccionario con arquitectura, parametros y tamano en disco.
        """
        from src.eval.complejidad import contar_parametros, tamano_en_disco_mb

        parametros = contar_parametros(self._modelo)
        return {
            "formato": "Keras (.keras)",
            "archivo": self.ruta.name,
            "arquitectura": self._modelo.name,
            "parametros_total": parametros["total"],
            "parametros_entrenables": parametros["entrenables"],
            "tamano_mb": tamano_en_disco_mb(self.ruta),
            "entrada": list(self._modelo.input_shape),
        }


class PredictorTFLite(Predictor):
    """Predictor respaldado por un interprete de TensorFlow Lite.

    Gestiona por su cuenta la (de)cuantizacion de entrada y salida, de modo que
    quien lo usa siempre trabaja con float32 en ``[0, 1]`` y recibe una
    probabilidad, independientemente de como se exporto el modelo.
    """

    def __init__(self, ruta: Path, n_hilos: int = 1) -> None:
        import tensorflow as tf

        super().__init__(ruta, "tflite")
        self._interprete = tf.lite.Interpreter(model_path=str(ruta), num_threads=n_hilos)
        self._interprete.allocate_tensors()
        self._entrada = self._interprete.get_input_details()[0]
        self._salida = self._interprete.get_output_details()[0]
        self._n_hilos = n_hilos

    def _inferir(self, lote: np.ndarray) -> np.ndarray:
        resultados: list[float] = []
        forma_esperada = tuple(int(v) for v in self._entrada["shape"])

        for indice in range(lote.shape[0]):
            muestra = lote[indice : indice + 1]

            if np.issubdtype(self._entrada["dtype"], np.integer):
                escala, punto_cero = self._entrada["quantization"]
                if escala == 0:  # modelo sin parametros de cuantizacion validos
                    escala = 1.0
                muestra = np.round(muestra / escala + punto_cero)
                info = np.iinfo(self._entrada["dtype"])
                muestra = np.clip(muestra, info.min, info.max).astype(self._entrada["dtype"])
            else:
                muestra = muestra.astype(self._entrada["dtype"])

            if muestra.shape != forma_esperada:
                self._interprete.resize_tensor_input(self._entrada["index"], muestra.shape)
                self._interprete.allocate_tensors()

            self._interprete.set_tensor(self._entrada["index"], muestra)
            self._interprete.invoke()
            bruto = self._interprete.get_tensor(self._salida["index"])

            if np.issubdtype(self._salida["dtype"], np.integer):
                escala, punto_cero = self._salida["quantization"]
                escala = escala if escala != 0 else 1.0
                bruto = (bruto.astype(np.float32) - punto_cero) * escala

            resultados.append(float(np.asarray(bruto).reshape(-1)[0]))

        return np.asarray(resultados, dtype=np.float32)

    def descripcion(self) -> dict[str, Any]:
        """Metadatos del modelo TFLite.

        Returns:
            Diccionario con tipo de entrada, tamano en disco e hilos.
        """
        from src.eval.complejidad import tamano_en_disco_mb

        return {
            "formato": "TensorFlow Lite (.tflite)",
            "archivo": self.ruta.name,
            "arquitectura": "MobileNetV2 convertido",
            "parametros_total": None,
            "tamano_mb": tamano_en_disco_mb(self.ruta),
            "tipo_entrada": str(np.dtype(self._entrada["dtype"])),
            "tipo_salida": str(np.dtype(self._salida["dtype"])),
            "entrada": [int(v) for v in self._entrada["shape"]],
            "n_hilos": self._n_hilos,
        }


class PredictorEnsemble(Predictor):
    """Combina varios predictores promediando sus probabilidades.

    Por que existe
    --------------
    Medido sobre las fotografias propias del equipo (reports/analisis.md, §3.8),
    promediar la CNN de linea base con MobileNetV2 sube el F1 de 0.6667 a 0.8235
    y elimina 4 de los 10 falsos negativos, **sin coste apreciable de latencia**:
    la linea base representa el 1.9% de los parametros del conjunto.

    El detalle que lo hace interesante es que sobre el conjunto de prueba publico
    el ensemble **empeora** ligeramente (0.9423 -> 0.9376). Los dos modelos
    fallan en imagenes distintas fuera de distribucion, y ahi es donde
    promediarlos aporta. Elegir la tecnica mirando solo el conjunto de prueba
    habria llevado a descartarla.

    Attributes:
        componentes: Predictores que se promedian.
    """

    def __init__(
        self,
        componentes: list[Predictor],
        pesos: list[float] | None = None,
        agregacion: str = "media",
    ) -> None:
        """Construye el ensemble.

        Args:
            componentes: Predictores ya cargados. Debe haber al menos uno.
            pesos: Peso de cada componente. ``None`` reparte por igual, que es
                la unica configuracion validada: ajustar pesos sobre 40
                fotografias seria sobreajustar la muestra de validacion.
            agregacion: ``"media"`` (reduce varianza) o ``"maximo"`` (dispara si
                cualquier componente ve grieta, privilegiando el recall).

        Raises:
            ValueError: Si no hay componentes o los pesos no cuadran.
        """
        if not componentes:
            raise ValueError("El ensemble necesita al menos un predictor.")
        if pesos is not None and len(pesos) != len(componentes):
            raise ValueError(
                f"Se dieron {len(pesos)} pesos para {len(componentes)} componentes."
            )

        super().__init__(componentes[0].ruta.parent, "ensemble")
        self.componentes = componentes
        self._pesos = list(pesos) if pesos else [1.0] * len(componentes)
        self._agregacion = agregacion.lower().strip()

    def _inferir(self, lote: np.ndarray) -> np.ndarray:
        salidas = np.stack([c.predecir(lote)[0] for c in self.componentes])
        if self._agregacion == "maximo":
            return salidas.max(axis=0)
        return np.average(salidas, axis=0, weights=self._pesos)

    def descripcion(self) -> dict[str, Any]:
        """Metadatos agregados de los componentes.

        Returns:
            Diccionario con el total de parametros y tamano sumados, mas la
            ficha de cada componente.
        """
        fichas = [c.descripcion() for c in self.componentes]
        parametros = [f.get("parametros_total") for f in fichas]
        tamanos = [f.get("tamano_mb") for f in fichas]

        return {
            "formato": f"Ensemble ({self._agregacion} de {len(self.componentes)} modelos)",
            "archivo": " + ".join(c.ruta.name for c in self.componentes),
            "arquitectura": " + ".join(str(f.get("arquitectura", "?")) for f in fichas),
            "parametros_total": (
                sum(p for p in parametros if p) if any(parametros) else None
            ),
            "tamano_mb": (
                round(sum(t for t in tamanos if t), 4) if any(tamanos) else None
            ),
            "entrada": fichas[0].get("entrada"),
            "componentes": fichas,
        }


def cargar_predictor(
    config: dict[str, Any], formato: str = "keras", ruta: str | Path | None = None
) -> Predictor:
    """Fabrica el predictor adecuado segun el formato solicitado.

    Args:
        config: Configuracion del proyecto.
        formato: ``"keras"``, ``"tflite"`` o ``"ensemble"``.
        ruta: Ruta explicita al artefacto. Si es ``None`` se toma de la seccion
            ``app`` del YAML. Se ignora para el ensemble, cuyos componentes se
            declaran en ``app.ensemble.componentes``.

    Returns:
        Instancia de :class:`PredictorKeras`, :class:`PredictorTFLite` o
        :class:`PredictorEnsemble`.

    Raises:
        FileNotFoundError: Si el artefacto no existe.
        ValueError: Si el formato no se reconoce o el ensemble esta mal definido.
    """
    formato = formato.lower().strip()

    if formato == "ensemble":
        componentes = obtener(config, "app.ensemble.componentes", []) or []
        if not componentes:
            raise ValueError(
                "El ensemble no tiene componentes. Declara 'app.ensemble.componentes' "
                "en config.yaml con las rutas de los modelos .keras a promediar."
            )
        faltantes = [c for c in componentes if not resolver(c).is_file()]
        if faltantes:
            raise FileNotFoundError(
                f"Faltan componentes del ensemble: {faltantes}. Entrena los modelos "
                "que faltan o ajusta 'app.ensemble.componentes'."
            )
        return PredictorEnsemble(
            [PredictorKeras(resolver(c)) for c in componentes],
            pesos=obtener(config, "app.ensemble.pesos"),
            agregacion=str(obtener(config, "app.ensemble.agregacion", "media")),
        )

    if ruta is None:
        clave = "app.archivo_keras" if formato == "keras" else "app.archivo_tflite"
        ruta = obtener(config, clave)

    destino = resolver(ruta)
    if not destino.is_file():
        raise FileNotFoundError(
            f"No existe el artefacto {destino}.\n"
            "Entrena y exporta primero:\n"
            "  python scripts/train_transfer.py --config config.yaml\n"
            "  python scripts/export_tflite.py --config config.yaml"
        )

    if formato == "keras":
        return PredictorKeras(destino)
    if formato == "tflite":
        return PredictorTFLite(destino)
    raise ValueError(
        f"Formato no reconocido: '{formato}'. Usa 'keras', 'tflite' o 'ensemble'."
    )


def predecir_dataset(predictor: Predictor, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Recorre un ``tf.data.Dataset`` con un predictor y acumula los resultados.

    Args:
        predictor: Predictor Keras o TFLite.
        dataset: Dataset que produce pares ``(imagenes, etiquetas)``.

    Returns:
        Tupla ``(y_true, y_prob)`` con dos vectores ``(N,)``.
    """
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for lote_x, lote_y in dataset:
        probabilidades, _ = predictor.predecir(np.asarray(lote_x, dtype=np.float32))
        y_prob.append(np.asarray(probabilidades).reshape(-1))
        y_true.append(np.asarray(lote_y).reshape(-1))
    if not y_true:
        return np.empty(0), np.empty(0)
    return np.concatenate(y_true).astype(float), np.concatenate(y_prob).astype(float)
