"""Inclinometria y orientacion de grietas mediante vision clasica (OpenCV).

Este modulo es la mitad "algoritmica" del proyecto: no aprende nada de los datos,
resuelve el problema con geometria. Aporta dos medidas que la CNN no puede dar:

1. **Desaplome**: desviacion angular de un elemento vertical (columna, muro,
   viga) respecto a la vertical de la imagen.
2. **Orientacion de la grieta**: vertical / horizontal / diagonal, que es el
   dato que la ingenieria estructural usa para interpretar la gravedad de una
   fisura.

Complejidad asintotica (discusion exigida por la asignatura)
------------------------------------------------------------
Sea ``n = H x W`` el numero de pixeles de la imagen.

- Conversion a gris y desenfoque gaussiano separable de kernel ``k``:
  ``O(n * k)``. Con ``k`` fijo (5) es ``O(n)``.
- **Canny**: gradientes Sobel ``O(n)``, supresion de no maximos ``O(n)``,
  histeresis ``O(n)`` amortizado (cada pixel entra una vez en la pila). Total
  ``O(n)``.
- **HoughLinesP** (Hough probabilistico): sea ``m`` el numero de pixeles de
  borde tras Canny (``m << n``, tipicamente 1-5% de ``n``). El algoritmo muestrea
  puntos de borde y, por cada uno, vota sobre ``T`` valores discretos de theta.
  El costo es ``O(m * T)`` mas el recorrido de los segmentos aceptados. Con
  ``T = 180`` (resolucion de 1 grado), el termino dominante del pipeline completo
  sigue siendo lineal en el numero de pixeles: **``O(n + m*T)``**.
- Filtrado y estadistica robusta sobre ``L`` segmentos detectados: ordenar para
  la mediana ponderada cuesta ``O(L log L)``, con ``L`` del orden de decenas.

En la practica el termino que domina el tiempo de pared es Canny (``O(n)``), por
lo que **reducir la resolucion antes de procesar es la unica optimizacion que
importa**: el costo cae cuadraticamente con el lado de la imagen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.utils.config import obtener

# Colores BGR para la anotacion. Se eligen de alto contraste para que sigan
# siendo legibles proyectados en un videobeam durante la sustentacion.
_COLOR_LINEA_VALIDA = (0, 220, 0)
_COLOR_LINEA_DESCARTADA = (140, 140, 140)
_COLOR_REFERENCIA = (0, 160, 255)
_COLOR_TEXTO = (255, 255, 255)
_COLOR_FONDO_TEXTO = (30, 30, 30)


@dataclass
class ResultadoInclinacion:
    """Salida del estimador de desaplome.

    Attributes:
        angulo_grados: Desviacion respecto a la vertical. Positivo = el elemento
            se inclina hacia la derecha (su parte superior esta desplazada a la
            derecha de su base). ``None`` si no se pudo estimar.
        fiable: ``True`` si el numero de lineas coherentes alcanza el minimo
            configurado.
        confianza: Numero de segmentos coherentes que sustentan el angulo.
        dispersion_grados: Desviacion absoluta mediana (MAD) de los angulos
            coherentes. Es la incertidumbre de la medida.
        n_lineas_detectadas: Segmentos devueltos por Hough antes de filtrar.
        n_lineas_validas: Segmentos que pasaron los filtros de longitud y
            verticalidad.
        mensaje: Explicacion legible del resultado, pensada para mostrarse en la
            interfaz.
        motivo_rechazo: Si un guardarrail de verosimilitud descarto la medida,
            el motivo. ``None`` si no se activo ninguno. Permite a la interfaz
            distinguir "no se detecto nada" de "se midio algo que no cuadra".
        lineas_validas: Segmentos aceptados como ``(x1, y1, x2, y2)``.
        angulos_validos: Angulo de cada segmento aceptado, en grados.
    """

    angulo_grados: float | None
    fiable: bool
    confianza: int
    dispersion_grados: float
    n_lineas_detectadas: int
    n_lineas_validas: int
    mensaje: str
    motivo_rechazo: str | None = None
    lineas_validas: list[tuple[int, int, int, int]] = field(default_factory=list)
    angulos_validos: list[float] = field(default_factory=list)
    lineas_descartadas: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class ResultadoOrientacion:
    """Salida del clasificador de orientacion de grieta.

    Attributes:
        orientacion: ``"horizontal"``, ``"vertical"``, ``"diagonal"`` o
            ``"indeterminada"``.
        angulo_grados: Angulo dominante respecto a la horizontal, en ``[0, 90]``.
        confianza: Numero de segmentos que sustentan la estimacion.
        mensaje: Explicacion legible.
    """

    orientacion: str
    angulo_grados: float | None
    confianza: int
    mensaje: str


# --------------------------------------------------------------------------- #
# Primitivas geometricas
# --------------------------------------------------------------------------- #


def _longitud(x1: float, y1: float, x2: float, y2: float) -> float:
    """Longitud euclidea de un segmento.

    Args:
        x1: Coordenada x del primer extremo.
        y1: Coordenada y del primer extremo.
        x2: Coordenada x del segundo extremo.
        y2: Coordenada y del segundo extremo.

    Returns:
        Longitud en pixeles.
    """
    return math.hypot(x2 - x1, y2 - y1)


def angulo_desviacion_vertical(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calcula la desviacion de un segmento respecto a la vertical de la imagen.

    Convencion de signo: el segmento se orienta de la base hacia el tope (en
    coordenadas de imagen, ``y`` crece hacia abajo, asi que el tope es el extremo
    con ``y`` menor). Un resultado **positivo** significa que el tope esta
    desplazado hacia la **derecha** de la base, es decir, el elemento se inclina
    a la derecha.

    Args:
        x1: Coordenada x del primer extremo.
        y1: Coordenada y del primer extremo.
        x2: Coordenada x del segundo extremo.
        y2: Coordenada y del segundo extremo.

    Returns:
        Angulo en grados en el intervalo ``(-90, 90]``. Cero = perfectamente
        vertical.

    Example:
        >>> round(angulo_desviacion_vertical(0, 100, 0, 0), 3)
        0.0
        >>> round(angulo_desviacion_vertical(0, 100, 10, 0), 3) > 0
        True
    """
    # Reordenar para que (x2, y2) sea el extremo superior.
    if y1 < y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx = x2 - x1
    dy = y1 - y2  # positivo por construccion
    return math.degrees(math.atan2(dx, dy))


def angulo_respecto_horizontal(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calcula la inclinacion de un segmento respecto a la horizontal.

    Args:
        x1: Coordenada x del primer extremo.
        y1: Coordenada y del primer extremo.
        x2: Coordenada x del segundo extremo.
        y2: Coordenada y del segundo extremo.

    Returns:
        Angulo en grados en ``[0, 90]``: 0 = horizontal, 90 = vertical. Se toma
        el valor absoluto porque una recta no tiene sentido, solo direccion.
    """
    dx = x2 - x1
    dy = -(y2 - y1)  # invertir el eje y de imagen para razonar en ejes cartesianos
    if dx == 0 and dy == 0:
        return 0.0
    angulo = math.degrees(math.atan2(dy, dx)) % 180.0
    return angulo if angulo <= 90.0 else 180.0 - angulo


def mediana_ponderada(valores: Sequence[float], pesos: Sequence[float]) -> float:
    """Mediana ponderada de una muestra.

    Se usa en vez del promedio simple porque la salida de Hough contiene
    valores atipicos casi siempre: el borde de una ventana, un cable o el marco
    de la propia fotografia producen segmentos largos con angulos arbitrarios.
    Un unico atipico desplaza la media; la mediana ponderada por longitud es
    robusta y ademas da mas voto al segmento largo, que es el que realmente
    describe el eje del elemento.

    Args:
        valores: Muestra de valores.
        pesos: Peso de cada valor (aqui, la longitud del segmento). Deben ser
            no negativos.

    Returns:
        El valor cuya masa acumulada de peso alcanza el 50% del total. Devuelve
        ``0.0`` si la muestra esta vacia.

    Example:
        >>> mediana_ponderada([1.0, 2.0, 100.0], [10.0, 10.0, 1.0])
        2.0
    """
    if len(valores) == 0:
        return 0.0
    orden = np.argsort(np.asarray(valores, dtype=float))
    v = np.asarray(valores, dtype=float)[orden]
    p = np.asarray(pesos, dtype=float)[orden]
    total = p.sum()
    if total <= 0:
        return float(np.median(v))
    acumulado = np.cumsum(p)
    indice = int(np.searchsorted(acumulado, total / 2.0))
    return float(v[min(indice, len(v) - 1)])


def _mad(valores: Sequence[float], centro: float) -> float:
    """Desviacion absoluta mediana respecto a un centro dado.

    Args:
        valores: Muestra de valores.
        centro: Valor central de referencia.

    Returns:
        Mediana de ``|valor - centro|``. Cero si la muestra esta vacia.
    """
    if len(valores) == 0:
        return 0.0
    return float(np.median(np.abs(np.asarray(valores, dtype=float) - centro)))


def _angulo_axial_dominante(angulos: Sequence[float], pesos: Sequence[float]) -> float:
    """Promedio circular de angulos no orientados (rectas), en ``[0, 180)``.

    Promediar angulos de rectas con la media aritmetica falla en el envolvimiento
    (``179`` y ``1`` grados son casi la misma direccion, pero su media da ``90``).
    La solucion estandar es duplicar el angulo, promediar como vectores unitarios
    y volver a dividir entre dos.

    Args:
        angulos: Angulos en grados, en ``[0, 180)``.
        pesos: Peso de cada angulo.

    Returns:
        Angulo dominante en grados dentro de ``[0, 180)``.
    """
    if len(angulos) == 0:
        return 0.0
    radianes = np.radians(np.asarray(angulos, dtype=float) * 2.0)
    w = np.asarray(pesos, dtype=float)
    seno = float(np.sum(w * np.sin(radianes)))
    coseno = float(np.sum(w * np.cos(radianes)))
    return float(math.degrees(math.atan2(seno, coseno)) / 2.0) % 180.0


# --------------------------------------------------------------------------- #
# Deteccion de lineas
# --------------------------------------------------------------------------- #


def detectar_lineas(
    imagen_bgr: np.ndarray,
    config: dict[str, Any],
    canny_bajo: int | None = None,
    canny_alto: int | None = None,
    min_longitud: int | None = None,
    max_separacion: int | None = None,
    umbral_hough: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta el pipeline gris -> desenfoque -> Canny -> HoughLinesP.

    Los parametros opcionales existen para que los controles de la barra lateral
    de la aplicacion Streamlit modifiquen el resultado en vivo sin tocar el YAML.
    Si se dejan en ``None``, se usan los valores de ``config.yaml``.

    Args:
        imagen_bgr: Imagen en formato BGR (convencion de OpenCV), ``(H, W, 3)``.
        config: Configuracion del proyecto.
        canny_bajo: Umbral inferior de histeresis de Canny.
        canny_alto: Umbral superior de histeresis de Canny.
        min_longitud: Longitud minima de segmento para HoughLinesP.
        max_separacion: Hueco maximo tolerado dentro de un mismo segmento.
        umbral_hough: Votos minimos en el acumulador para aceptar una linea.

    Returns:
        Tupla ``(bordes, lineas)``. ``bordes`` es el mapa binario de Canny
        ``(H, W)``; ``lineas`` es un arreglo ``(L, 4)`` con los segmentos
        ``(x1, y1, x2, y2)``, vacio si no se detecto ninguno.
    """
    cfg = config.get("inclinacion", {})
    k = int(cfg.get("desenfoque_kernel", 5))
    k = k if k % 2 == 1 else k + 1  # el kernel gaussiano debe ser impar

    bajo = int(canny_bajo if canny_bajo is not None else cfg.get("canny_umbral_bajo", 50))
    alto = int(canny_alto if canny_alto is not None else cfg.get("canny_umbral_alto", 150))
    if alto <= bajo:
        # Canny exige alto > bajo; se corrige en silencio para que un slider mal
        # puesto en la interfaz no lance una excepcion durante la demo.
        alto = bajo + 1

    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (k, k), 0)
    bordes = cv2.Canny(suavizada, bajo, alto, L2gradient=True)

    lado_menor = min(imagen_bgr.shape[:2])
    min_rel = float(cfg.get("min_longitud_relativa", 0.15))
    longitud = int(
        min_longitud
        if min_longitud is not None
        else max(int(cfg.get("hough_min_longitud_linea", 80)), int(lado_menor * min_rel))
    )

    lineas = cv2.HoughLinesP(
        bordes,
        rho=float(cfg.get("hough_rho", 1)),
        theta=math.radians(float(cfg.get("hough_theta_grados", 1.0))),
        threshold=int(umbral_hough if umbral_hough is not None else cfg.get("hough_umbral", 60)),
        minLineLength=max(longitud, 1),
        maxLineGap=int(
            max_separacion if max_separacion is not None else cfg.get("hough_max_separacion", 10)
        ),
    )

    if lineas is None:
        return bordes, np.empty((0, 4), dtype=np.int32)
    return bordes, lineas.reshape(-1, 4)


# --------------------------------------------------------------------------- #
# Estimacion de desaplome
# --------------------------------------------------------------------------- #


def _motivo_inverosimil(
    angulo: float,
    dispersion: float,
    lineas: list[tuple[int, int, int, int]],
    alto_imagen: int,
    cfg: dict[str, Any],
) -> str | None:
    """Comprueba si una medida de desaplome es fisicamente inverosimil.

    Hough no distingue una arista estructural de una grieta ni de un cable: si
    en el encuadre no hay ningun elemento vertical, la unica recta casi vertical
    puede ser la propia fisura, y el modulo la reportaria como desaplome. Estos
    tres criterios detectan ese caso **por las propiedades de la medida**, sin
    necesidad de entender la escena.

    Args:
        angulo: Desaplome estimado, en grados.
        dispersion: MAD de los angulos coherentes, en grados.
        lineas: Segmentos coherentes que sustentan la medida.
        alto_imagen: Alto de la imagen en pixeles.
        cfg: Seccion ``inclinacion`` de la configuracion.

    Returns:
        Mensaje explicando por que la medida se rechaza, o ``None`` si supera
        los tres criterios.
    """
    maximo = float(cfg.get("desaplome_maximo_plausible_grados", 10.0))
    if abs(angulo) > maximo:
        return (
            f"Desaplome de {abs(angulo):.2f} grados: inverosimil para una edificacion "
            f"en pie (limite {maximo:.0f} grados). Lo mas probable es que se este "
            "midiendo una grieta o un objeto, no el eje del elemento. Medida descartada."
        )

    dispersion_maxima = float(cfg.get("dispersion_maxima_grados", 2.0))
    if dispersion > dispersion_maxima:
        return (
            f"Los segmentos no coinciden entre si (dispersion +-{dispersion:.2f} grados, "
            f"limite {dispersion_maxima:.1f}). Es la firma de una linea sinuosa, no de "
            "una arista recta. Medida descartada."
        )

    minima_extension = float(cfg.get("min_extension_vertical", 0.30))
    if lineas and alto_imagen > 0:
        ys = [coord for x1, y1, x2, y2 in lineas for coord in (y1, y2)]
        extension = (max(ys) - min(ys)) / alto_imagen
        if extension < minima_extension:
            return (
                f"Los segmentos solo abarcan el {100 * extension:.0f}% del alto de la "
                f"imagen (minimo {100 * minima_extension:.0f}%). No describen un elemento "
                "vertical completo. Medida descartada."
            )

    return None


def estimar_inclinacion(
    imagen_bgr: np.ndarray,
    config: dict[str, Any],
    canny_bajo: int | None = None,
    canny_alto: int | None = None,
    min_longitud: int | None = None,
    max_separacion: int | None = None,
    umbral_hough: int | None = None,
    aplicar_guardarrailes: bool = True,
) -> ResultadoInclinacion:
    """Estima el desaplome de un elemento vertical en la fotografia.

    Algoritmo:
        1. Detectar segmentos con Canny + HoughLinesP.
        2. Descartar los segmentos cortos (ruido y textura del material).
        3. Descartar los que se alejan de la vertical mas que
           ``inclinacion.tolerancia_vertical_grados``: son suelos, dinteles,
           juntas horizontales o el horizonte, no el eje del elemento.
        4. Calcular la mediana de los angulos ponderada por longitud.
        5. Refinar: quedarse solo con los segmentos coherentes (a menos de 5
           grados de esa mediana) y recalcular. Este paso convierte el estimador
           en uno de tipo "trimmed", inmune a un grupo minoritario de segmentos
           paralelos espurios.

    Nunca lanza excepcion por ausencia de lineas: devuelve ``fiable=False`` con
    un mensaje explicativo, porque en la aplicacion ese caso es un resultado
    valido ("no se detecto ningun elemento vertical"), no un error.

    Args:
        imagen_bgr: Imagen BGR ``(H, W, 3)``.
        config: Configuracion del proyecto.
        canny_bajo: Umbral inferior de Canny (sobrescribe el YAML).
        canny_alto: Umbral superior de Canny (sobrescribe el YAML).
        min_longitud: Longitud minima de segmento (sobrescribe el YAML).
        max_separacion: Hueco maximo dentro de un segmento (sobrescribe el YAML).
        umbral_hough: Votos minimos del acumulador (sobrescribe el YAML).
        aplicar_guardarrailes: Si se aplican los filtros de verosimilitud
            (angulo maximo plausible, dispersion y extension vertical). Debe
            quedar en ``True`` para inspeccion real. Se pone en ``False``
            unicamente al **validar la geometria** del estimador, donde se
            generan a proposito angulos que ninguna edificacion en pie tendria:
            el guardarrail codifica una suposicion sobre la escena, no sobre la
            correccion del algoritmo, y aplicarlo ahi impediria medir su error.

    Returns:
        Objeto :class:`ResultadoInclinacion`.
    """
    cfg = config.get("inclinacion", {})
    tolerancia = float(cfg.get("tolerancia_vertical_grados", 35.0))
    min_lineas = int(cfg.get("min_lineas_confiables", 3))

    _, lineas = detectar_lineas(
        imagen_bgr, config, canny_bajo, canny_alto, min_longitud, max_separacion, umbral_hough
    )

    if len(lineas) == 0:
        return ResultadoInclinacion(
            angulo_grados=None,
            fiable=False,
            confianza=0,
            dispersion_grados=0.0,
            n_lineas_detectadas=0,
            n_lineas_validas=0,
            mensaje=(
                "No se detectaron bordes rectos. Prueba a bajar el umbral de Canny "
                "o a reducir la longitud minima de linea."
            ),
        )

    validas: list[tuple[int, int, int, int]] = []
    descartadas: list[tuple[int, int, int, int]] = []
    angulos: list[float] = []
    longitudes: list[float] = []

    for x1, y1, x2, y2 in lineas:
        largo = _longitud(x1, y1, x2, y2)
        angulo = angulo_desviacion_vertical(x1, y1, x2, y2)
        if abs(angulo) <= tolerancia:
            validas.append((int(x1), int(y1), int(x2), int(y2)))
            angulos.append(angulo)
            longitudes.append(largo)
        else:
            descartadas.append((int(x1), int(y1), int(x2), int(y2)))

    if not validas:
        return ResultadoInclinacion(
            angulo_grados=None,
            fiable=False,
            confianza=0,
            dispersion_grados=0.0,
            n_lineas_detectadas=len(lineas),
            n_lineas_validas=0,
            mensaje=(
                f"Se detectaron {len(lineas)} lineas, pero ninguna se aproxima a la "
                f"vertical (tolerancia +-{tolerancia:.0f} grados). La foto puede no "
                "contener un elemento vertical completo."
            ),
            lineas_descartadas=descartadas,
        )

    angulo_bruto = mediana_ponderada(angulos, longitudes)

    # Refinamiento: conservar solo lo coherente con la estimacion inicial.
    margen_coherencia = 5.0
    coherentes = [
        (linea, ang, largo)
        for linea, ang, largo in zip(validas, angulos, longitudes, strict=False)
        if abs(ang - angulo_bruto) <= margen_coherencia
    ]
    if coherentes:
        lineas_c = [c[0] for c in coherentes]
        angulos_c = [c[1] for c in coherentes]
        pesos_c = [c[2] for c in coherentes]
    else:  # pragma: no cover - defensivo; la mediana siempre deja algo dentro
        lineas_c, angulos_c, pesos_c = validas, angulos, longitudes

    angulo_final = mediana_ponderada(angulos_c, pesos_c)
    dispersion = _mad(angulos_c, angulo_final)
    confianza = len(angulos_c)
    fiable = confianza >= min_lineas

    if fiable:
        sentido = "derecha" if angulo_final > 0 else "izquierda"
        mensaje = (
            f"Desaplome de {abs(angulo_final):.2f} grados hacia la {sentido}, "
            f"estimado con {confianza} lineas coherentes (dispersion "
            f"+-{dispersion:.2f} grados)."
        )
    else:
        mensaje = (
            f"Solo {confianza} linea(s) coherente(s), por debajo del minimo de "
            f"{min_lineas}. El angulo de {angulo_final:.2f} grados es orientativo "
            "y no debe usarse para decidir."
        )

    # Guardarrailes de verosimilitud: solo se aplican si la medida habia pasado
    # el filtro de confianza. Ninguno corrige el angulo; lo declaran no fiable
    # con el motivo, que es lo que el motor de reglas necesita para no disparar
    # R5/R6 sobre una medida que no describe el elemento (ver analisis.md, §4.5).
    motivo_rechazo = None
    if fiable and aplicar_guardarrailes:
        motivo_rechazo = _motivo_inverosimil(
            angulo_final, dispersion, lineas_c, imagen_bgr.shape[0], cfg
        )
        if motivo_rechazo is not None:
            fiable = False
            mensaje = motivo_rechazo

    return ResultadoInclinacion(
        angulo_grados=float(angulo_final),
        fiable=fiable,
        confianza=confianza,
        dispersion_grados=float(dispersion),
        n_lineas_detectadas=len(lineas),
        n_lineas_validas=len(validas),
        mensaje=mensaje,
        motivo_rechazo=motivo_rechazo,
        lineas_validas=lineas_c,
        angulos_validos=[float(a) for a in angulos_c],
        lineas_descartadas=descartadas,
    )


def anotar_imagen(
    imagen_bgr: np.ndarray,
    resultado: ResultadoInclinacion,
    dibujar_descartadas: bool = True,
    dibujar_referencia: bool = True,
) -> np.ndarray:
    """Dibuja sobre la imagen las lineas detectadas y el angulo estimado.

    Args:
        imagen_bgr: Imagen original BGR.
        resultado: Salida de :func:`estimar_inclinacion`.
        dibujar_descartadas: Si se pintan en gris las lineas rechazadas. Ayuda a
            entender por que el algoritmo decidio lo que decidio.
        dibujar_referencia: Si se pinta la vertical de referencia.

    Returns:
        Copia anotada de la imagen, en BGR. La original no se modifica.
    """
    lienzo = imagen_bgr.copy()
    alto, ancho = lienzo.shape[:2]
    grosor = max(1, int(round(min(alto, ancho) / 400)))

    if dibujar_descartadas:
        for x1, y1, x2, y2 in resultado.lineas_descartadas:
            cv2.line(lienzo, (x1, y1), (x2, y2), _COLOR_LINEA_DESCARTADA, grosor)

    for x1, y1, x2, y2 in resultado.lineas_validas:
        cv2.line(lienzo, (x1, y1), (x2, y2), _COLOR_LINEA_VALIDA, grosor + 1)

    if dibujar_referencia and resultado.angulo_grados is not None:
        # Vertical de referencia por el centro: da al observador humano el mismo
        # marco que usa el algoritmo.
        cx = ancho // 2
        for y in range(0, alto, 20):
            cv2.line(lienzo, (cx, y), (cx, min(y + 10, alto)), _COLOR_REFERENCIA, grosor)

    etiqueta = (
        f"Desaplome: {resultado.angulo_grados:+.2f} deg"
        if resultado.angulo_grados is not None
        else "Desaplome: no estimable"
    )
    estado = "FIABLE" if resultado.fiable else "BAJA CONFIANZA"
    texto = f"{etiqueta}  |  {estado} ({resultado.confianza} lineas)"

    escala = max(0.4, min(alto, ancho) / 800.0)
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
    cv2.rectangle(lienzo, (5, 5), (5 + tw + 10, 5 + th + base + 10), _COLOR_FONDO_TEXTO, -1)
    cv2.putText(
        lienzo,
        texto,
        (10, 10 + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        _COLOR_TEXTO,
        1,
        cv2.LINE_AA,
    )
    return lienzo


# --------------------------------------------------------------------------- #
# Orientacion de la grieta
# --------------------------------------------------------------------------- #


def clasificar_orientacion_grieta(
    imagen_bgr: np.ndarray,
    config: dict[str, Any],
    canny_bajo: int | None = None,
    canny_alto: int | None = None,
    min_longitud: int | None = None,
) -> ResultadoOrientacion:
    """Clasifica la orientacion dominante de una grieta en vertical/horizontal/diagonal.

    Reutiliza la misma transformada de Hough del desaplome, pero sin el filtro
    de verticalidad: aqui interesan todas las direcciones. El angulo dominante se
    obtiene con estadistica **axial** (duplicando el angulo antes de promediar)
    porque una grieta es una recta sin sentido: 179 y 1 grados son casi la misma
    direccion y la media aritmetica los promediaria a 90.

    La orientacion importa porque la interpretacion estructural depende de ella:
    una fisura diagonal en un muro sugiere cortante, una horizontal en una
    columna sugiere flexo-traccion, y una vertical en un muro suele ser
    retraccion del mortero (mucho menos grave).

    Args:
        imagen_bgr: Imagen BGR de la zona de la grieta.
        config: Configuracion del proyecto.
        canny_bajo: Umbral inferior de Canny (sobrescribe el YAML).
        canny_alto: Umbral superior de Canny (sobrescribe el YAML).
        min_longitud: Longitud minima de segmento (sobrescribe el YAML).

    Returns:
        Objeto :class:`ResultadoOrientacion`. Si no hay evidencia suficiente
        devuelve ``orientacion="indeterminada"`` sin lanzar excepcion.
    """
    umbral_h = float(obtener(config, "orientacion_grieta.umbral_horizontal", 25.0))
    umbral_v = float(obtener(config, "orientacion_grieta.umbral_vertical", 65.0))

    # La grieta suele producir segmentos mas cortos que el eje de una columna,
    # asi que se relaja la longitud minima a la mitad si no se pasa explicita.
    if min_longitud is None:
        lado_menor = min(imagen_bgr.shape[:2])
        min_longitud = max(15, int(lado_menor * 0.08))

    _, lineas = detectar_lineas(
        imagen_bgr, config, canny_bajo, canny_alto, min_longitud=min_longitud
    )

    if len(lineas) == 0:
        return ResultadoOrientacion(
            orientacion="indeterminada",
            angulo_grados=None,
            confianza=0,
            mensaje=(
                "No se detectaron segmentos suficientes para determinar la "
                "orientacion de la grieta."
            ),
        )

    angulos = [angulo_respecto_horizontal(*linea) for linea in lineas]
    pesos = [_longitud(*linea) for linea in lineas]

    dominante = _angulo_axial_dominante(angulos, pesos)
    # Llevar a [0, 90]: la inclinacion respecto a la horizontal no distingue
    # entre subir hacia la derecha o hacia la izquierda.
    angulo = dominante if dominante <= 90.0 else 180.0 - dominante

    if angulo < umbral_h:
        orientacion = "horizontal"
    elif angulo > umbral_v:
        orientacion = "vertical"
    else:
        orientacion = "diagonal"

    return ResultadoOrientacion(
        orientacion=orientacion,
        angulo_grados=float(angulo),
        confianza=len(lineas),
        mensaje=(
            f"Orientacion dominante {orientacion} ({angulo:.1f} grados respecto a la "
            f"horizontal), a partir de {len(lineas)} segmentos."
        ),
    )


# --------------------------------------------------------------------------- #
# Validacion sin ground truth: rotaciones controladas
# --------------------------------------------------------------------------- #


def rotar_imagen(imagen_bgr: np.ndarray, angulo_grados: float, recorte: float = 0.8) -> np.ndarray:
    """Rota una imagen un angulo conocido y recorta el centro.

    El recorte central es necesario: al rotar aparecen bordes artificiales en las
    esquinas que Canny detecta como lineas rectas perfectas y que contaminarian
    la validacion con segmentos que no existen en la escena.

    Args:
        imagen_bgr: Imagen BGR de entrada.
        angulo_grados: Angulo de rotacion. Positivo = antihorario (convencion de
            ``cv2.getRotationMatrix2D``).
        recorte: Fraccion central conservada tras rotar, en ``(0, 1]``.

    Returns:
        Imagen rotada y recortada.
    """
    alto, ancho = imagen_bgr.shape[:2]
    centro = (ancho / 2.0, alto / 2.0)
    matriz = cv2.getRotationMatrix2D(centro, angulo_grados, 1.0)
    rotada = cv2.warpAffine(
        imagen_bgr,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if recorte >= 1.0:
        return rotada
    nuevo_alto, nuevo_ancho = int(alto * recorte), int(ancho * recorte)
    y0 = (alto - nuevo_alto) // 2
    x0 = (ancho - nuevo_ancho) // 2
    return rotada[y0 : y0 + nuevo_alto, x0 : x0 + nuevo_ancho]


def validar_con_rotaciones(
    imagen_bgr: np.ndarray,
    config: dict[str, Any],
    angulos_prueba: Sequence[float] = (2.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Valida el estimador de angulo rotando artificialmente una foto a plomo.

    Sin ground truth instrumentado (un inclinometro fisico) no se puede medir el
    error absoluto del metodo. Lo que si se puede medir es su **fidelidad
    diferencial**: si se rota la misma imagen un angulo conocido ``alpha``, la
    estimacion debe desplazarse exactamente ``-alpha`` respecto a la medida de
    referencia de la imagen sin rotar.

    Justificacion del signo: ``cv2`` rota el contenido en sentido antihorario
    para ``alpha > 0``, con lo que el tope de un elemento vertical se desplaza
    hacia la izquierda y la desviacion medida (positiva hacia la derecha)
    disminuye en ``alpha``.

    Args:
        imagen_bgr: Fotografia de un elemento razonablemente a plomo.
        config: Configuracion del proyecto.
        angulos_prueba: Angulos de rotacion a evaluar, en grados. Se evaluan
            tambien sus opuestos para detectar sesgos de signo.

    Returns:
        Diccionario con ``referencia_grados``, la lista ``casos`` (cada uno con
        ``rotacion_aplicada``, ``angulo_esperado``, ``angulo_estimado``,
        ``error_absoluto``, ``fiable``) y los agregados ``error_medio_grados``,
        ``error_maximo_grados`` y ``n_casos_fiables``.

    Raises:
        ValueError: Si la imagen base no produce una estimacion fiable; sin
            referencia el experimento no tiene sentido.
    """
    base = estimar_inclinacion(imagen_bgr, config, aplicar_guardarrailes=False)
    if not base.fiable or base.angulo_grados is None:
        raise ValueError(
            "La imagen de referencia no produce una estimacion fiable "
            f"({base.mensaje}). Usa una foto con un elemento vertical claro y bien "
            "iluminado antes de validar."
        )

    referencia = base.angulo_grados
    casos: list[dict[str, Any]] = []
    errores: list[float] = []

    for magnitud in angulos_prueba:
        for signo in (1.0, -1.0):
            alpha = signo * float(magnitud)
            rotada = rotar_imagen(imagen_bgr, alpha)
            # Los guardarrailes se desactivan a proposito: este experimento
            # fabrica desaplomes de hasta +-10 grados, que en una fotografia de
            # inspeccion se rechazarian por inverosimiles. Aqui lo que se mide es
            # la fidelidad geometrica del estimador, no la plausibilidad de la
            # escena, asi que aplicarlos impediria justamente medir el error.
            estimacion = estimar_inclinacion(rotada, config, aplicar_guardarrailes=False)
            esperado = referencia - alpha

            caso: dict[str, Any] = {
                "rotacion_aplicada": alpha,
                "angulo_esperado": float(esperado),
                "angulo_estimado": (
                    float(estimacion.angulo_grados)
                    if estimacion.angulo_grados is not None
                    else None
                ),
                "fiable": bool(estimacion.fiable),
                "confianza": int(estimacion.confianza),
            }
            if estimacion.angulo_grados is not None:
                error = abs(float(estimacion.angulo_grados) - esperado)
                caso["error_absoluto"] = error
                if estimacion.fiable:
                    errores.append(error)
            else:
                caso["error_absoluto"] = None
            casos.append(caso)

    return {
        "referencia_grados": float(referencia),
        "confianza_referencia": int(base.confianza),
        "casos": casos,
        "error_medio_grados": float(np.mean(errores)) if errores else None,
        "error_maximo_grados": float(np.max(errores)) if errores else None,
        "n_casos_fiables": len(errores),
        "n_casos": len(casos),
    }
