"""Conversion de pixeles a milimetros mediante un marcador de referencia.

Que problema resuelve
---------------------
El modulo de medicion (``src.vision.morfologia``) entrega la geometria de la
fisura en pixeles, y ahi se detiene a proposito: **la escala de una fotografia no
esta en la fotografia**. Una fisura de 3 px puede tener 0.1 mm o 5 mm segun la
distancia a la que se disparo, y ninguna cantidad de procesamiento puede
recuperar ese dato porque nunca se registro.

La consecuencia practica es seria: la NSR-10 clasifica la severidad del dano por
el **ancho de fisura en milimetros**, de modo que sin escala el sistema puede
decir "hay una grieta diagonal aqui" pero no "esta grieta mide 1.2 mm y por tanto
requiere revision". Es la limitacion que §5.1 del informe declara como la mas
grave del proyecto.

La solucion: meter la escala en la escena
-----------------------------------------
Si la informacion no esta en la imagen, hay que ponerla. Se fotografia junto a la
grieta un **marcador ArUco** impreso de tamano conocido: un cuadrado en blanco y
negro con un patron interno que lo identifica sin ambiguedad. El sistema lo
localiza, mide su lado en pixeles, y como sabe cuanto mide en milimetros, obtiene
la equivalencia.

Un marcador y no una moneda o una regla, por tres motivos: se detecta de forma
automatica y fiable, lleva codificada su propia identidad —de modo que el sistema
sabe **que** marcador es y por tanto cuanto mide—, y al ser cuadrado permite
detectar si se fotografio de frente o en angulo, que es la principal fuente de
error de este metodo.

Lo que este modulo no hace
--------------------------
No adivina. Si no encuentra marcador, lo dice y las medidas siguen en pixeles.
Un sistema que estimara la escala "a ojo" produciria dictamenes con apariencia
normativa y fundamento inventado, que es peor que no dar ninguno.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.config import obtener


@dataclass(frozen=True)
class ReferenciaEscala:
    """Resultado de buscar un marcador de escala en una fotografia.

    Attributes:
        detectada: Si se encontro algun marcador utilizable.
        mm_por_px: Milimetros que representa cada pixel, o ``None``.
        lado_px: Lado medido del marcador en pixeles.
        lado_mm: Lado declarado del marcador en milimetros.
        identificador: Numero del marcador detectado.
        centro: Coordenada ``(x, y)`` de su centro.
        esquinas: Las cuatro esquinas detectadas.
        deformacion: Cuanto se aparta el marcador de un cuadrado perfecto, como
            fraccion. Es la medida de perspectiva: 0.0 significa fotografiado de
            frente, y valores altos que se tomo en angulo, en cuyo caso la
            equivalencia deja de ser valida uniformemente sobre la imagen.
        fiable: Si la deformacion esta dentro del limite admitido.
        aviso: Advertencia cuando se detecto pero conviene desconfiar.
        motivo: Explicacion cuando no se detecto.
    """

    detectada: bool
    mm_por_px: float | None = None
    lado_px: float = 0.0
    lado_mm: float = 0.0
    identificador: int | None = None
    centro: tuple[int, int] | None = None
    esquinas: list[tuple[float, float]] = field(default_factory=list)
    deformacion: float = 0.0
    fiable: bool = False
    aviso: str | None = None
    motivo: str | None = None

    def resumen(self) -> str:
        """Describe la referencia en una linea legible.

        Returns:
            Texto con la equivalencia hallada, o el motivo de su ausencia.
        """
        if not self.detectada or self.mm_por_px is None:
            return self.motivo or "Sin referencia de escala"
        texto = (
            f"Marcador #{self.identificador} de {self.lado_mm:.0f} mm "
            f"({self.lado_px:.0f} px) · 1 px = {self.mm_por_px:.4f} mm"
        )
        return texto + (f" · {self.aviso}" if self.aviso else "")


def _diccionario(config: dict[str, Any]) -> Any:
    """Obtiene el diccionario ArUco configurado.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Diccionario de marcadores de OpenCV.

    Raises:
        ValueError: Si el nombre declarado no existe en OpenCV.
    """
    import cv2

    nombre = str(obtener(config, "escala.diccionario", "DICT_4X4_50"))
    if not hasattr(cv2.aruco, nombre):
        raise ValueError(
            f"El diccionario ArUco '{nombre}' no existe en OpenCV {cv2.__version__}. "
            "Usa uno de los DICT_* disponibles, por ejemplo DICT_4X4_50."
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, nombre))


def detectar_escala(imagen_bgr: np.ndarray, config: dict[str, Any]) -> ReferenciaEscala:
    """Busca un marcador de referencia y calcula la equivalencia mm/px.

    Debe ejecutarse sobre la **fotografia completa y sin redimensionar**, no
    sobre el recorte donde se mide la fisura: el marcador suele quedar fuera de
    ese recorte, y cualquier cambio de tamano de la imagen alteraria la
    equivalencia que se esta calculando.

    El lado del marcador se toma como el promedio de sus cuatro lados. Promediar
    no es solo por precision: **la dispersion entre ellos es la senal de que la
    fotografia se tomo en angulo**, y es lo que permite avisar de que la escala
    no es uniforme sobre la imagen.

    Args:
        imagen_bgr: Fotografia completa en BGR.
        config: Configuracion del proyecto.

    Returns:
        Una :class:`ReferenciaEscala`. Nunca lanza excepcion por no encontrar
        marcador: que no lo haya es un caso normal, no un error.
    """
    import cv2

    cfg = obtener(config, "escala", {}) or {}
    lado_mm = float(cfg.get("lado_marcador_mm", 50.0))
    deformacion_maxima = float(cfg.get("deformacion_maxima", 0.15))

    imagen = np.asarray(imagen_bgr)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen

    detector = cv2.aruco.ArucoDetector(_diccionario(config), cv2.aruco.DetectorParameters())
    esquinas, identificadores, _ = detector.detectMarkers(gris)

    if identificadores is None or len(identificadores) == 0:
        return ReferenciaEscala(
            detectada=False,
            lado_mm=lado_mm,
            motivo=(
                "No se encontro ningun marcador de referencia. Las medidas quedan en "
                "pixeles: sin un objeto de tamano conocido en la escena, la escala no "
                "se puede deducir de la imagen."
            ),
        )

    # Si hay varios, se usa el mayor: es el mas cercano a la camara y el que
    # menos error relativo introduce al medir su lado.
    mejor = max(range(len(esquinas)), key=lambda i: cv2.contourArea(esquinas[i].reshape(4, 2)))
    puntos = esquinas[mejor].reshape(4, 2).astype(np.float64)

    lados = [float(np.linalg.norm(puntos[i] - puntos[(i + 1) % 4])) for i in range(4)]
    lado_px = float(np.mean(lados))
    if lado_px < 1.0:
        return ReferenciaEscala(
            detectada=False,
            lado_mm=lado_mm,
            motivo="El marcador detectado es demasiado pequeno para medirlo con fiabilidad.",
        )

    deformacion = float((max(lados) - min(lados)) / lado_px)
    fiable = deformacion <= deformacion_maxima

    aviso = None
    if not fiable:
        aviso = (
            f"El marcador se ve deformado un {deformacion:.0%}: la foto se tomo en "
            "angulo y la escala no es uniforme sobre la imagen. Repite la toma "
            "situandote perpendicular a la superficie."
        )
    elif lado_px < float(cfg.get("lado_minimo_px", 60)):
        aviso = (
            f"El marcador ocupa solo {lado_px:.0f} px: la escala se apoya en muy pocos "
            "pixeles y su error relativo es alto. Acercate o imprimelo mas grande."
        )

    centro = puntos.mean(axis=0)
    return ReferenciaEscala(
        detectada=True,
        mm_por_px=lado_mm / lado_px,
        lado_px=lado_px,
        lado_mm=lado_mm,
        identificador=int(identificadores[mejor][0]),
        centro=(int(centro[0]), int(centro[1])),
        esquinas=[(float(x), float(y)) for x, y in puntos],
        deformacion=deformacion,
        fiable=fiable,
        aviso=aviso,
    )


def aviso_por_lejania(
    referencia: ReferenciaEscala, region: tuple[int, int, int, int], config: dict[str, Any]
) -> str | None:
    """Advierte si el marcador esta lejos de la zona que se va a medir.

    La equivalencia mm/px solo es exacta **en el plano y a la distancia del
    marcador**. Si la grieta esta en otra parte de la fotografia, la perspectiva
    hace que un pixel represente alli una distancia distinta, y el error crece
    con la separacion. Es la limitacion de fondo de medir con una sola camara.

    Args:
        referencia: Referencia de escala detectada.
        region: Ventana ``(x0, y0, x1, y1)`` donde se midio la fisura.
        config: Configuracion del proyecto.

    Returns:
        Texto de advertencia, o ``None`` si la separacion es razonable.
    """
    if not referencia.detectada or referencia.centro is None:
        return None

    x0, y0, x1, y1 = region
    centro_region = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    separacion = math.dist(referencia.centro, centro_region)
    # Se expresa en lados de marcador: es la unidad natural aqui, porque el error
    # de perspectiva depende de la separacion relativa al tamano conocido.
    lados_de_distancia = separacion / max(referencia.lado_px, 1.0)
    limite = float(obtener(config, "escala.lados_maximos_de_distancia", 6.0))

    if lados_de_distancia <= limite:
        return None
    return (
        f"El marcador esta a {lados_de_distancia:.0f} veces su propio lado de la zona "
        "medida. La escala se calculo donde esta el marcador, no donde esta la grieta: "
        "acercalos para que la conversion a milimetros sea fiable."
    )


def anotar_referencia(imagen_bgr: np.ndarray, referencia: ReferenciaEscala) -> np.ndarray:
    """Dibuja el marcador detectado sobre la fotografia.

    Mostrar donde se detecto la referencia permite comprobar de un vistazo que la
    escala sale de donde debe. Un numero en milimetros sin esa evidencia visual no
    se puede auditar.

    Args:
        imagen_bgr: Fotografia en BGR.
        referencia: Referencia detectada.

    Returns:
        Copia anotada.
    """
    import cv2

    lienzo = np.asarray(imagen_bgr).copy()
    if not referencia.detectada or not referencia.esquinas:
        return lienzo

    puntos = np.array(referencia.esquinas, dtype=np.int32).reshape(-1, 1, 2)
    color = (60, 200, 60) if referencia.fiable else (40, 160, 250)
    grosor = max(2, int(min(lienzo.shape[:2]) / 400))
    cv2.polylines(lienzo, [puntos], True, color, grosor, cv2.LINE_AA)

    if referencia.centro is not None:
        cv2.putText(
            lienzo,
            f"{referencia.lado_mm:.0f} mm",
            (referencia.centro[0] - 30, referencia.centro[1] - int(referencia.lado_px / 2) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            grosor * 0.35,
            color,
            grosor,
            cv2.LINE_AA,
        )
    return lienzo
