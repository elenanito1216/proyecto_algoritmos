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
from dataclasses import dataclass
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


# =============================================================================
# INFERENCIA POR MOSAICOS
#
# El problema que resuelve
# ------------------------
# El modelo se entreno con parches cuya mediana es 227x227 px. Reducirlos a 160
# es un factor 1.4x: una grieta de 3 px de ancho sobrevive como 2.1 px.
#
# Una fotografia de telefono mide 1200x1600. Reducirla ENTERA a 160 es un factor
# 7.5x: esa misma grieta pasa a 0.4 px, es decir, desaparece en el filtrado
# bilineal antes de que el modelo llegue a verla.
#
# Medido sobre el dataset de entrenamiento (n=400) y las fotos propias (n=40):
#   entrenamiento  mediana  227x227   reduccion 1.4x
#   propias        mediana 1200x1600  reduccion 7.5x
#
# Es decir: buena parte del "desplazamiento de dominio" que §6 atribuia al
# material y la iluminacion es en realidad un artefacto de escala introducido
# por nosotros mismos en el redimensionado.
#
# La correccion no exige reentrenar: basta con trocear la fotografia en
# ventanas del tamano con el que el modelo aprendio y evaluar cada una. Cada
# mosaico llega al modelo con la grieta a su escala original.
# =============================================================================


@dataclass(frozen=True)
class ResultadoMosaicos:
    """Salida de :func:`predecir_por_mosaicos`.

    Attributes:
        probabilidad: Probabilidad agregada de la imagen completa.
        probabilidades: Vector ``(N,)`` con la probabilidad de cada mosaico.
        ventanas: Lista de ``(x0, y0, x1, y1)`` en pixeles de la imagen original.
        milisegundos: Tiempo total de inferencia.
        indice_maximo: Indice del mosaico con mayor probabilidad, o ``-1`` si no
            hubo ninguno. Sirve para senalar en la imagen DONDE se vio la grieta,
            informacion que la inferencia sobre la imagen entera no proporciona.
        lado: Lado efectivo del mosaico en pixeles.
    """

    probabilidad: float
    probabilidades: np.ndarray
    ventanas: list[tuple[int, int, int, int]]
    milisegundos: float
    indice_maximo: int
    lado: int


def generar_ventanas(
    alto: int, ancho: int, lado: int, solape: float = 0.5
) -> list[tuple[int, int, int, int]]:
    """Cubre una imagen con ventanas cuadradas solapadas.

    El solape no es opcional: una grieta que cayera justo en la frontera entre
    dos mosaicos quedaria partida en dos mitades, y ninguna de las dos seria
    reconocible. Con un solape del 50% cualquier region del tamano del mosaico
    aparece completa en al menos una ventana.

    Si la imagen es mas pequena que el lado pedido, se devuelve una unica ventana
    con la imagen entera: trocear no aportaria nada y solo anadiria coste.

    Args:
        alto: Alto de la imagen en pixeles.
        ancho: Ancho de la imagen en pixeles.
        lado: Lado del mosaico en pixeles.
        solape: Fraccion de solape entre ventanas contiguas, en ``[0, 0.9]``.

    Returns:
        Lista de tuplas ``(x0, y0, x1, y1)`` con coordenadas de pixel, extremo
        derecho excluido. Nunca esta vacia.
    """
    lado = max(1, int(lado))
    solape = float(min(max(solape, 0.0), 0.9))

    if alto <= lado or ancho <= lado:
        return [(0, 0, int(ancho), int(alto))]

    paso = max(1, int(round(lado * (1.0 - solape))))

    def _inicios(dimension: int) -> list[int]:
        posiciones = list(range(0, dimension - lado + 1, paso))
        # La ultima ventana se ancla al borde para no dejar sin cubrir la franja
        # final cuando la dimension no es multiplo del paso.
        if posiciones[-1] + lado < dimension:
            posiciones.append(dimension - lado)
        return posiciones

    return [
        (x, y, x + lado, y + lado) for y in _inicios(alto) for x in _inicios(ancho)
    ]


def predecir_por_mosaicos(
    predictor: Predictor,
    imagen_rgb: np.ndarray,
    config: dict[str, Any],
    lado: int | None = None,
    solape: float | None = None,
    agregacion: str | None = None,
    maximo_mosaicos: int | None = None,
) -> ResultadoMosaicos:
    """Analiza una imagen troceandola en ventanas a la escala de entrenamiento.

    Los mosaicos se infieren en un unico lote. Agruparlos importa: el coste fijo
    por llamada al modelo domina sobre el coste por imagen, asi que treinta
    llamadas de un mosaico tardan varias veces mas que una llamada de treinta.

    La agregacion por defecto es el maximo, y es la decision de diseno relevante:
    una grieta ocupa una fraccion pequena de la fotografia, de modo que la mayoria
    de los mosaicos son pared sana. Promediarlos diluiria la unica evidencia
    positiva hasta hacerla invisible. El maximo formaliza el criterio "si algun
    trozo tiene grieta, la imagen tiene grieta", que es tambien el criterio de un
    inspector. El precio es asimetrico y hay que decirlo: con N mosaicos, N
    oportunidades de falso positivo (§4.1 argumenta por que ese es el error que
    conviene cometer en evaluacion estructural).

    Args:
        predictor: Predictor ya cargado.
        imagen_rgb: Imagen RGB ``(H, W, 3)``.
        config: Configuracion del proyecto.
        lado: Lado del mosaico en pixeles. Por defecto ``app.mosaicos.lado_px``.
        solape: Solape entre ventanas. Por defecto ``app.mosaicos.solape``.
        agregacion: ``"maximo"`` o ``"media"``. Por defecto ``app.mosaicos.agregacion``.
        maximo_mosaicos: Cota superior de ventanas a evaluar. Si se supera, se
            aumenta el lado hasta cumplirla, en lugar de descartar zonas: perder
            resolucion es preferible a dejar parte del muro sin mirar.

    Returns:
        Un :class:`ResultadoMosaicos`.
    """
    lado = int(lado if lado is not None else obtener(config, "app.mosaicos.lado_px", 320))
    solape = float(solape if solape is not None else obtener(config, "app.mosaicos.solape", 0.5))
    agregacion = str(
        agregacion if agregacion is not None else obtener(config, "app.mosaicos.agregacion", "maximo")
    ).lower()
    maximo_mosaicos = int(
        maximo_mosaicos
        if maximo_mosaicos is not None
        else obtener(config, "app.mosaicos.maximo_mosaicos", 64)
    )

    imagen = np.asarray(imagen_rgb)
    alto, ancho = imagen.shape[:2]

    ventanas = generar_ventanas(alto, ancho, lado, solape)
    while len(ventanas) > maximo_mosaicos and lado < max(alto, ancho):
        lado = int(lado * 1.5)
        ventanas = generar_ventanas(alto, ancho, lado, solape)

    lote = np.concatenate(
        [preparar_imagen(imagen[y0:y1, x0:x1], config) for (x0, y0, x1, y1) in ventanas],
        axis=0,
    )
    probabilidades, ms = predictor.predecir(lote)
    probabilidades = np.asarray(probabilidades, dtype=np.float32).reshape(-1)

    if agregacion == "media":
        agregada = float(probabilidades.mean())
    else:
        agregada = float(probabilidades.max())

    return ResultadoMosaicos(
        probabilidad=agregada,
        probabilidades=probabilidades,
        ventanas=ventanas,
        milisegundos=ms,
        indice_maximo=int(np.argmax(probabilidades)) if probabilidades.size else -1,
        lado=lado,
    )

# =============================================================================
# QUE MODELO USA CADA MODO
#
# La aplicacion no ofrece un selector de modelo, y esa ausencia es una decision
# de diseno, no una simplificacion. Las dos formas de uso imponen restricciones
# opuestas y cada una tiene una respuesta medida en reports/analisis.md:
#
#   FOTOGRAFIA - se admite un segundo de calculo, manda el acierto.
#     Con mosaicos de 480 px sobre las 60 fotos propias (§4.6):
#       MobileNetV2  F1 0.9355  recall 0.97  1 FN  3 FP  0.62 s/foto
#       Ensemble     F1 0.9508  recall 0.97  1 FN  2 FP  0.78 s/foto
#       TFLite int8  F1 0.9123  recall 0.87  4 FN  1 FP  0.28 s/foto
#
#   VIDEO - hay 33 ms por fotograma para sostener 30 FPS, manda la latencia.
#       TFLite int8   4.9 ms      MobileNetV2 170.8 ms      Ensemble 189.4 ms
#
# Pedirle al usuario que elija seria trasladarle una decision que la aplicacion
# puede tomar mejor: la respuesta esta medida, no depende de su preferencia.
# =============================================================================

ETIQUETAS_MODELO: dict[str, str] = {
    "ensemble": "Ensemble",
    "keras": "MobileNetV2",
    "tflite": "TFLite int8",
}

# Orden de repliegue cuando el modelo elegido para un modo no esta en disco. Se
# prefiere degradar a otro antes que dejar la funcionalidad muerta, pero quien
# llame debe advertir de que no se esta usando el que dicen las mediciones.
REPLIEGUE_POR_MODO: dict[str, tuple[str, ...]] = {
    "foto": ("keras", "ensemble", "tflite"),
    "video": ("tflite", "keras", "ensemble"),
}


def elegir_modelo(
    config: dict[str, Any], modo: str, disponibles: dict[str, bool]
) -> str | None:
    """Elige el formato de modelo que corresponde a un modo de uso.

    Recibe la disponibilidad ya calculada en lugar de mirar el disco, de modo que
    la decision es una funcion pura: se puede probar con todas las combinaciones
    de artefactos presentes y ausentes sin tocar el sistema de archivos.

    Args:
        config: Configuracion del proyecto.
        modo: ``"foto"`` o ``"video"``.
        disponibles: Mapa ``{formato: existe_en_disco}``.

    Returns:
        ``"keras"``, ``"tflite"`` o ``"ensemble"``, o ``None`` si no hay ningun
        artefacto disponible para ese modo.
    """
    preferido = str(obtener(config, f"app.modelos.{modo}", "") or "")
    candidatos = (preferido, *REPLIEGUE_POR_MODO.get(modo, ()))
    return next((f for f in candidatos if f and disponibles.get(f)), None)
