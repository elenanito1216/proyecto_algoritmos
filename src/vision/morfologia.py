"""Caracterizacion geometrica de una grieta: longitud, ancho y forma.

Que problema resuelve
---------------------
Hasta aqui el sistema respondia "hay grieta" o "no hay grieta". La §5.1 del
informe declara que esa es su limitacion mas grave, porque la NSR-10 clasifica
la severidad del dano por el **ancho de fisura**, y un si/no no permite
priorizar entre dos elementos igualmente agrietados.

Este modulo extrae la geometria de la fisura: cuanto mide, que ancho tiene, si
va recta o se ramifica. Opera sobre un recorte donde ya se sabe que hay grieta
—el que localiza el analisis por mosaicos de §4.6— de modo que no tiene que
resolver el problema de deteccion, solo el de medicion.

La advertencia que gobierna todo el modulo
------------------------------------------
**Todas las medidas se expresan en pixeles, nunca en milimetros.** Sin una
referencia de tamano conocido en la escena, una fisura de 3 px puede tener
0.1 mm o 5 mm segun la distancia a la que se tomo la fotografia: la informacion
sencillamente no esta en la imagen. Convertir a milimetros exige un marcador de
escala, y eso es la fase siguiente.

Un sistema que reportara milimetros sin referencia estaria inventando la cifra
mas importante del dictamen. Este modulo prefiere decir "47 px" y que el usuario
sepa lo que eso significa.

Por que vision clasica y no una segunda red
-------------------------------------------
Segmentar la grieta con una red exigiria mascaras etiquetadas pixel a pixel, que
no existen en ningun conjunto publico de este dominio y que el equipo no puede
producir con garantias. Sobre un recorte donde ya se sabe que hay fisura, la
morfologia matematica resuelve el problema sin datos adicionales y de forma
auditable: cada paso se puede visualizar y explicar.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.utils.config import obtener


@dataclass(frozen=True)
class MedidasGrieta:
    """Geometria extraida de una fisura.

    Todas las longitudes estan en **pixeles de la imagen analizada**. Para
    convertirlas a unidades fisicas hace falta una referencia de escala en la
    escena; mientras no la haya, ``escala_mm_por_px`` vale ``None`` y las
    medidas en milimetros no deben calcularse.

    Attributes:
        detectada: Si se aislo alguna fisura con area suficiente.
        longitud_px: Recorrido de la **trayectoria principal**: el camino mas
            largo que se puede seguir por el eje sin repetir. Es lo que responde
            a "cuanto mide esta grieta".
        longitud_total_px: Suma de todo el eje, ramas laterales incluidas.
            Responde a otra pregunta -cuanto material fisurado hay- y se conserva
            porque la diferencia entre ambas cifras mide cuanto se ramifica.
        extremos_principales: Los dos extremos de la trayectoria principal.
        ancho_maximo_px: Ancho en el punto mas abierto.
        ancho_medio_px: Ancho promedio a lo largo del eje.
        ancho_p95_px: Percentil 95 del ancho. Mas estable que el maximo, que
            depende de un unico pixel y es sensible al ruido de segmentacion.
        area_px: Superficie ocupada por la fisura.
        orientacion_grados: Angulo del eje principal respecto a la horizontal,
            en ``[-90, 90)``.
        tortuosidad: Longitud de la trayectoria principal dividida entre la
            distancia en linea recta entre sus extremos. Vale 1.0 en una fisura
            perfectamente recta y crece conforme serpentea. **No puede bajar de
            1.0**, lo que la convierte en una comprobacion de coherencia: un
            valor menor delata que se midio un conjunto de fragmentos sueltos.
        indice_ramificacion: Longitud total del eje dividida entre la de la
            trayectoria principal. Vale 1.0 en una fisura que es una sola linea
            y crece con cada rama lateral. Sustituye al recuento de
            bifurcaciones, que sobre material rugoso daba cifras absurdas -402
            en una grieta unica- porque cada aspereza del borde produce una.
        extremos: Coordenadas ``(x, y)`` de los puntos finales del eje.
        punto_mas_ancho: Coordenada ``(x, y)`` donde se midio el ancho maximo.
        cobertura: Fraccion del recorte ocupada por la fisura.
        longitud_inferida_px: Parte del recorrido que corresponde a tramos
            cosidos entre fragmentos, donde no se observo fisura sino que se
            dedujo que continuaba. Permite saber cuanto de la medida es
            observacion y cuanto reconstruccion.
        union_rechazada: Motivo por el que se renuncio a unir los fragmentos y se
            midio solo el trozo principal. ``None`` cuando la union fue valida.
        fragmentos_descartados: Otros candidatos a fisura que habia en el
            recorte y no se midieron. Un valor alto sugiere textura ruidosa o
            varias grietas, y en ambos casos conviene mirar la imagen anotada.
        escala_mm_por_px: Milimetros que representa cada pixel, cuando hubo un
            marcador de referencia en la escena. ``None`` significa que no lo
            habia, y entonces los campos en milimetros tambien son ``None``.
        longitud_mm: Recorrido de la trayectoria principal en milimetros.
        ancho_medio_mm: Ancho medio en milimetros.
        ancho_maximo_mm: Ancho maximo en milimetros. **Es la magnitud que la
            NSR-10 usa para clasificar la severidad del dano**, y la razon de ser
            de todo el modulo de escala.
        motivo: Explicacion cuando ``detectada`` es ``False``.
    """

    detectada: bool
    longitud_px: float = 0.0
    longitud_total_px: float = 0.0
    extremos_principales: tuple[tuple[int, int], tuple[int, int]] | None = None
    ancho_maximo_px: float = 0.0
    ancho_medio_px: float = 0.0
    ancho_p95_px: float = 0.0
    area_px: int = 0
    orientacion_grados: float | None = None
    tortuosidad: float = 0.0
    indice_ramificacion: float = 1.0
    extremos: list[tuple[int, int]] = field(default_factory=list)
    punto_mas_ancho: tuple[int, int] | None = None
    cobertura: float = 0.0
    longitud_inferida_px: float = 0.0
    union_rechazada: str | None = None
    fragmentos_descartados: int = 0
    escala_mm_por_px: float | None = None
    longitud_mm: float | None = None
    ancho_medio_mm: float | None = None
    ancho_maximo_mm: float | None = None
    motivo: str | None = None

    def resumen(self) -> str:
        """Describe la fisura en una linea legible.

        Returns:
            Texto con las magnitudes principales, o el motivo del descarte.
        """
        if not self.detectada:
            return self.motivo or "No se aislo ninguna fisura"
        if self.ancho_maximo_mm is not None:
            return (
                f"{self.longitud_mm:.0f} mm de recorrido · ancho medio "
                f"{self.ancho_medio_mm:.2f} mm (maximo {self.ancho_maximo_mm:.2f}) · "
                f"tortuosidad {self.tortuosidad:.2f}"
            )
        return (
            f"{self.longitud_px:.0f} px de recorrido · ancho medio "
            f"{self.ancho_medio_px:.1f} px (maximo {self.ancho_maximo_px:.1f}) · "
            f"tortuosidad {self.tortuosidad:.2f} · ramificacion "
            f"{self.indice_ramificacion:.2f}"
        )


def respuesta_de_cresta(
    gris: np.ndarray, escalas: tuple[float, ...] = (1.0, 2.0, 3.0), beta: float = 0.5
) -> np.ndarray:
    """Mide cuanto se parece cada pixel al centro de una linea oscura.

    Por que hace falta ademas del black-hat
    ---------------------------------------
    El black-hat pregunta *cuanto mas oscuro es este pixel que su entorno*, y
    sobre un pañete rugoso la respuesta es "bastante" en media pared: la grieta
    aparece, pero pegada a toda la textura, en un solo componente de 443 424 px
    que despues hay que tirar entero. Medido sobre la fotografia del equipo, el
    umbral de Otsu enciende el 30 % de la imagen.

    Este filtro pregunta otra cosa: *tiene este pixel forma de cresta*. Se
    calcula el Hessiano -las segundas derivadas- a varias escalas y se miran sus
    dos autovalores. En el centro de una linea oscura, la curvatura es fuerte
    cruzando la linea y casi nula a lo largo: esa asimetria es la firma, y la
    textura no la tiene aunque sea igual de oscura. Es el criterio de Frangi,
    el mismo que se usa para realzar vasos sanguineos en angiografia.

    Sobre la misma fotografia, con corte en 0.05, entran los 32 puntos de la
    grieta encendiendo el 10.8 % de la imagen en vez del 30 %.

    Args:
        gris: Imagen en escala de grises, preferiblemente ya ecualizada.
        escalas: Sigmas del suavizado previo. Cada una responde a lineas de un
            grosor distinto; se toma el maximo, de modo que una fisura fina y
            una ancha se detectan igual de bien.
        beta: Cuanto se penaliza que la estructura sea una mancha en vez de una
            linea. 0.5 es el valor clasico de Frangi.

    Returns:
        Respuesta en ``float32`` entre 0 y 1, alta en el centro de las lineas
        oscuras y ~0 en la textura y en el fondo.
    """
    import cv2

    imagen = gris.astype(np.float32)
    salida = np.zeros_like(imagen)
    for sigma in escalas:
        suave = cv2.GaussianBlur(imagen, (0, 0), sigma)
        # La normalizacion por sigma^2 es lo que hace comparables las escalas:
        # sin ella, la respuesta decae con el suavizado y siempre ganaria la
        # escala mas fina, que es la mas ruidosa.
        factor = sigma**2
        dxx = cv2.Sobel(suave, cv2.CV_32F, 2, 0, ksize=3) * factor
        dyy = cv2.Sobel(suave, cv2.CV_32F, 0, 2, ksize=3) * factor
        dxy = cv2.Sobel(suave, cv2.CV_32F, 1, 1, ksize=3) * factor

        raiz = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy**2)
        uno, dos = 0.5 * (dxx + dyy + raiz), 0.5 * (dxx + dyy - raiz)
        menor = np.where(np.abs(uno) <= np.abs(dos), uno, dos)
        mayor = np.where(np.abs(uno) <= np.abs(dos), dos, uno)

        manchez = menor**2 / np.maximum(mayor**2, 1e-6)
        fuerza = np.sqrt(menor**2 + mayor**2)
        # El corte marca que se considera "estructura fuerte". Se probo fijarlo
        # en el percentil 99 en vez de en el maximo, por robustez ante un boquete
        # oscuro que dispara el maximo, y sobre las 31 fotografias el resultado
        # fue mucho mas inestable: la respuesta se vuelve tan sensible que
        # enciende bordes y sombras. Se mantiene el maximo, que es ademas lo que
        # propone Frangi.
        corte = max(0.5 * float(fuerza.max()), 1e-6)
        valor = np.exp(-manchez / (2 * beta**2)) * (1.0 - np.exp(-(fuerza**2) / (2 * corte**2)))
        # mayor <= 0 es una linea CLARA sobre fondo oscuro: no es una fisura.
        valor[mayor <= 0] = 0.0
        salida = np.maximum(salida, valor)
    return salida


def _salto_de_fondo(gris: np.ndarray, componente: np.ndarray) -> float:
    """Mide cuanto cambia el fondo de un lado a otro del trazo.

    Es lo que distingue una fisura de un canto, y hace falta porque por forma no
    se distinguen: **el canto entre dos paredes tambien es una linea larga,
    oscura y de un solo trazo**, y ademas mas larga y mas recta que la grieta. El
    detector de crestas, por si solo, mide el canto y no la fisura.

    La diferencia es fisica, no geometrica. Un canto separa dos superficies que
    reciben luz distinta, asi que el gris salta al cruzarlo. Una fisura es un
    surco sobre una unica superficie: a los dos lados esta la misma pared. Medido
    sobre la fotografia que lo destapo:

        canto de la pared    17.5 niveles de gris
        fisura                2.5 niveles de gris

    Args:
        gris: Imagen en escala de grises.
        componente: Mascara binaria de un solo trazo, del tamano de ``gris``.

    Returns:
        Valor absoluto de la mediana del salto, en niveles de gris.
    """
    vertical = componente.any(axis=1).sum() >= componente.any(axis=0).sum()
    lienzo = gris if vertical else gris.T
    trazo = componente if vertical else componente.T

    saltos: list[float] = []
    filas = np.where(trazo.any(axis=1))[0]
    for fila in filas[::3]:  # una de cada tres basta y cuesta un tercio
        puntos = np.where(trazo[fila] > 0)[0]
        centro = int(puntos.mean())
        radio = int((puntos.max() - puntos.min()) // 2) + 10
        if centro - radio - 12 < 0 or centro + radio + 12 >= lienzo.shape[1]:
            continue  # el trazo toca el borde: no hay fondo que comparar
        izquierda = float(np.median(lienzo[fila, centro - radio - 12 : centro - radio - 2]))
        derecha = float(np.median(lienzo[fila, centro + radio + 2 : centro + radio + 12]))
        saltos.append(derecha - izquierda)
    return abs(float(np.median(saltos))) if saltos else 0.0


def segmentar_grieta(
    imagen_bgr: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Aisla la fisura del fondo devolviendo una mascara binaria.

    El procedimiento tiene tres pasos y cada uno ataca un problema concreto:

    1. **Ecualizacion local del contraste (CLAHE).** Una pared rara vez esta
       iluminada de forma uniforme: hay una zona al sol y otra en sombra. Un
       umbral global fallaria en una de las dos. CLAHE normaliza el contraste por
       regiones, de modo que la fisura destaque igual en ambas.
    2. **Transformacion *black-hat*.** Es la operacion clave. Resta a la imagen
       su propia version "cerrada", lo que deja unicamente **las estructuras
       oscuras y estrechas sobre fondo claro** — que es exactamente la
       descripcion de una grieta. Las manchas grandes, los cambios de color y los
       degradados de iluminacion desaparecen porque no son estrechos.
    3. **Umbral de Otsu y limpieza.** Otsu elige el corte solo, a partir del
       histograma, sin que haya que fijar un numero a mano que dependeria de cada
       fotografia. Despues se eliminan los componentes diminutos, que son ruido
       de textura del material.

    El tamano del nucleo del black-hat es el unico parametro realmente critico:
    debe ser **mayor que el ancho esperado de la fisura** y menor que las
    estructuras que no interesan. Por debajo, la grieta se borra a si misma; muy
    por encima, empiezan a colarse sombras.

    Args:
        imagen_bgr: Recorte en BGR donde se sabe que hay una grieta.
        config: Configuracion del proyecto.

    Returns:
        Tupla ``(mascara, puentes, fragmentos_descartados)``. La mascara lleva
        los pixeles de la fisura; ``puentes`` marca los tramos inferidos que
        cosen fragmentos separados por el umbral; el entero cuenta los candidatos
        que quedaron fuera del grupo elegido.
    """
    import cv2

    cfg = obtener(config, "medicion", {}) or {}
    gris = cv2.cvtColor(np.asarray(imagen_bgr), cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.get("clahe_clip", 2.0)),
        tileGridSize=(int(cfg.get("clahe_rejilla", 8)),) * 2,
    )
    realzada = clahe.apply(gris)

    cresta = obtener(config, "medicion.cresta", {}) or {}
    if cresta.get("activo", True):
        escalas = tuple(float(e) for e in cresta.get("escalas", (1.0, 2.0, 3.0)))
        respuesta = respuesta_de_cresta(realzada, escalas)
        umbral = float(cresta.get("umbral", 0.05))
        binaria = (respuesta >= umbral).astype(np.uint8) * 255
        sombrero = (respuesta * 255).astype(np.uint8)
    else:
        lado = int(cfg.get("kernel_blackhat", 15))
        lado = lado if lado % 2 == 1 else lado + 1  # los nucleos impares tienen centro
        nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
        sombrero = cv2.morphologyEx(realzada, cv2.MORPH_BLACKHAT, nucleo)
        # Un desenfoque suave antes de umbralizar evita que el ruido del sensor
        # genere componentes de uno o dos pixeles que luego hay que descartar.
        sombrero = cv2.GaussianBlur(sombrero, (3, 3), 0)
        _, binaria = cv2.threshold(sombrero, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    limpieza = int(cfg.get("kernel_limpieza", 3))
    nucleo_limpieza = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (limpieza,) * 2)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo_limpieza)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, nucleo_limpieza)

    # Se pasan ademas de la mascara: la respuesta dice CUANTO destaca cada pixel,
    # no solo si supero el umbral, y el gris hace falta para distinguir una
    # fisura de un canto, que por forma son iguales.
    return _quedarse_con_la_fisura(binaria, config, sombrero, gris)


def _quedarse_con_la_fisura(
    binaria: np.ndarray,
    config: dict[str, Any],
    respuesta: np.ndarray | None = None,
    gris: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Descarta los componentes que no pueden ser una grieta.

    Se filtra por dos criterios, y el segundo es el que hace el trabajo fino:

    - **Area minima**: por debajo de cierto tamano es ruido de textura.
    - **Marana**: una linea cumple ``area = largo x ancho``; una red de textura
      rellena una superficie y su area es mucho mayor. El cociente entre ambas
      vale ~1 en una fisura y se dispara en una red. Es el filtro que evita medir
      la textura del panete como si fuera la grieta.
    - **Elongacion sobre la caja minima rotada**: una grieta es larga y
      estrecha. Un componente tan ancho como largo es una mancha, una sombra o
      un desconchado. Se usa la caja rotada y no la alineada con los ejes porque
      una fisura diagonal tiene la caja alineada casi cuadrada, y un filtro sobre
      ella descartaria precisamente las grietas diagonales.

    Args:
        binaria: Mascara binaria de entrada.
        config: Configuracion del proyecto.
        respuesta: Imagen de respuesta del detector, para medir cuanto destaca
            cada candidato sobre el fondo. Opcional: sin ella la eleccion se
            hace solo por geometria.
        gris: Imagen en escala de grises, para rechazar cantos por el salto de
            fondo. Opcional: sin ella no se aplica ese filtro.

    Returns:
        Tupla ``(mascara, descartados)`` con un unico componente y el numero de
        candidatos que se dejaron fuera.
    """
    import cv2

    cfg = obtener(config, "medicion", {}) or {}
    alto, ancho = binaria.shape[:2]
    # El minimo se mide en AREA y escala con el tamano de la imagen. Es discutible
    # -una grieta es una linea, y su area crece con su longitud, no con la
    # superficie de la foto- y de hecho descarta trozos finos de fisura real:
    # en una foto de 1200x1600 exige 960 px, o sea 320 px de recorrido seguido
    # para un trazo de 3 px. Se probo sustituirlo por un minimo de LARGO y el
    # resultado fue peor, medido sobre las 31 fotografias: admite miles de motas
    # de textura que compiten con la fisura, cinco grietas bien medidas se
    # desplomaron -uno de 1465 a 117 px- y el tiempo se multiplico por veinte.
    # Lo que hace falta ahi no es otro umbral de tamano sino separar la fisura de
    # la textura antes de filtrar (§5.5).
    area_minima = max(
        int(float(cfg.get("area_minima_relativa", 0.0005)) * alto * ancho),
        int(cfg.get("area_minima_absoluta", 30)),
    )
    elongacion_minima = float(cfg.get("elongacion_minima", 2.0))
    marana_maxima = float(cfg.get("marana_maxima", 5.0))
    salto_maximo = float((obtener(config, "medicion.cresta", {}) or {}).get("salto_maximo", 8.0))

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    # Se calcula una sola vez: da el ancho local en cada punto, que hace falta
    # para separar una fisura de una red de textura.
    distancias = cv2.distanceTransform(binaria, cv2.DIST_L2, 5)

    candidatos: list[tuple[float, int]] = []
    for indice in range(1, n):  # 0 es el fondo
        area = int(stats[indice, cv2.CC_STAT_AREA])
        if area < area_minima:
            continue

        # Cada componente se examina dentro de su propia caja y no sobre la
        # imagen entera. Con el minimo por area, los componentes eran decenas;
        # con el minimo por largo son miles, y recorrer la imagen completa una
        # vez por cada uno multiplicaba el tiempo por cinco.
        x0 = int(stats[indice, cv2.CC_STAT_LEFT])
        y0 = int(stats[indice, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[indice, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[indice, cv2.CC_STAT_HEIGHT])
        recorte_etiquetas = etiquetas[y0:y1, x0:x1]
        recorte_distancias = distancias[y0:y1, x0:x1]

        # La elongacion se mide sobre la caja MINIMA ROTADA, no sobre la caja
        # alineada con los ejes, y la diferencia no es un matiz: una grieta
        # diagonal tiene una caja alineada practicamente cuadrada, de modo que
        # un filtro sobre ella la rechazaria por "compacta". Serian justamente
        # las fisuras diagonales -las que §4.4 identifica como estructuralmente
        # significativas en columnas- las que el sistema dejaria de medir.
        componente = (recorte_etiquetas == indice).astype(np.uint8)
        contornos, _ = cv2.findContours(componente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            continue
        (_, _), (ancho_rot, alto_rot), _ = cv2.minAreaRect(max(contornos, key=cv2.contourArea))
        lado_mayor = max(float(ancho_rot), float(alto_rot))
        lado_menor = max(min(float(ancho_rot), float(alto_rot)), 1.0)
        if lado_mayor / lado_menor < elongacion_minima:
            continue

        # Filtro de MARANA. Es el que distingue una grieta de la textura del
        # panete, y la elongacion no basta para eso: una red de textura que ocupa
        # una zona cuadrada tiene elongacion baja, pero una que ocupa una franja
        # la tiene alta y pasaba el filtro.
        #
        # Una linea cumple  area = largo x ancho.  Una malla, no: rellena una
        # superficie, asi que su area es mucho mayor que la de la linea que la
        # recorre. El cociente entre ambas es ~1 en una fisura y se dispara en
        # una red. Medido sobre la peor fotografia del conjunto propio:
        #
        #     componente        area    largo   ancho   cociente
        #     malla de textura 73247      556     5.7      23.1   <- descartar
        #     grieta real       3320      442     4.2       1.8
        #     grieta real       1173      189     3.7       1.7
        #     grieta real       1682      163     4.1       2.5
        #
        # El cociente no depende del tamano ni de la orientacion, que es lo que
        # lo hace utilizable sin recalibrar en cada fotografia.
        ancho_medio = 2.0 * float(recorte_distancias[componente > 0].mean())
        esbeltez = _factor_marana(area, lado_mayor, ancho_medio)
        if esbeltez > marana_maxima:
            continue

        # Filtro de CANTO. Por forma, una fisura y el canto entre dos paredes
        # son la misma cosa; lo que los separa es que el canto cambia el fondo
        # al cruzarlo y la fisura no (ver _salto_de_fondo).
        if gris is not None and salto_maximo > 0:
            entero = np.zeros(binaria.shape[:2], np.uint8)
            entero[y0:y1, x0:x1] = componente
            if _salto_de_fondo(gris, entero) > salto_maximo:
                continue

        candidatos.append((lado_mayor, indice))

    salida = np.zeros_like(binaria)
    if not candidatos:
        return salida, np.zeros_like(binaria), 0

    return _unir_fragmentos(etiquetas, stats, candidatos, config, respuesta)


def _unir_fragmentos(
    etiquetas: np.ndarray,
    stats: np.ndarray,
    candidatos: list[tuple[float, int]],
    config: dict[str, Any],
    respuesta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reune los trozos en que la segmentacion parte una misma grieta.

    El problema
    -----------
    Una fisura real no se segmenta de una pieza: donde se afina o donde la luz la
    disimula, el umbral la pierde y el trazo queda partido. Sobre una fotografia
    del equipo, una grieta que recorria los 1600 px de la imagen aparecio rota en
    cuatro fragmentos principales:

        fragmento 1   y  759 -> 1443     fragmento 3   y  389 ->  769
        fragmento 2   y    0 ->  497     fragmento 4   y 1367 -> 1600

    Quedarse con el mayor -que es lo que hacia la version anterior de esta
    funcion- reportaba 969 px cuando la grieta medía mas del doble. La regla
    existia por un motivo legitimo: medir fragmentos sueltos como si fueran una
    pieza daba tortuosidades de 0.25, geometricamente imposibles. Pero curaba el
    sintoma cortando la grieta.

    La solucion
    -----------
    Unir los fragmentos **que pertenecen a la misma fisura**, y solo esos. Se
    dilata una copia de los candidatos y se agrupan los que quedan conectados;
    los huecos medidos entre trozos contiguos eran de 6 a 15 px, y la union es
    transitiva, de modo que una cadena de saltos cortos recompone la grieta
    entera sin necesidad de un salto largo.

    Dos precauciones importantes:

    - La dilatacion se aplica **solo a los candidatos ya filtrados**, nunca a la
      mascara en bruto. Sobre la mascara completa -2104 componentes en esa
      fotografia- la dilatacion fusionaria la grieta con toda la textura del muro.
    - La dilatacion sirve para **decidir que se une, no para engordar la mascara**.
      Lo que se devuelve son los pixeles originales. Si se devolviera la mascara
      dilatada, los anchos saldrian inflados y la calibracion de +3 px dejaria de
      valer.

    Args:
        etiquetas: Matriz de etiquetas de componentes conexos.
        stats: Estadisticas de cada componente.
        candidatos: Lista ``(lado_mayor, indice)`` que paso los filtros.
        config: Configuracion del proyecto.
        respuesta: Imagen del black-hat, para elegir el grupo por contraste.

    Returns:
        Tupla ``(mascara, puentes, descartados)``. La mascara lleva los pixeles
        originales del grupo elegido; ``puentes`` marca los tramos inventados
        para coser los fragmentos, separados a proposito para que las medidas
        puedan distinguir lo observado de lo inferido.
    """
    import cv2

    cfg = obtener(config, "medicion", {}) or {}
    radio = int(cfg.get("radio_puente_px", 20))

    indices = [indice for _, indice in candidatos]
    mascara_candidatos = np.isin(etiquetas, indices).astype(np.uint8) * 255

    if radio <= 0 or len(indices) == 1:
        mejor = max(candidatos)[1]
        return (
            (etiquetas == mejor).astype(np.uint8) * 255,
            np.zeros_like(mascara_candidatos),
            len(candidatos) - 1,
        )

    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radio * 2 + 1,) * 2)
    dilatada = cv2.dilate(mascara_candidatos, nucleo)
    _, grupos = cv2.connectedComponents(dilatada, connectivity=8)

    # De cada fragmento se toma un pixel para saber en que grupo cayo.
    por_grupo: dict[int, list[int]] = {}
    for indice in indices:
        ys, xs = np.where(etiquetas == indice)
        por_grupo.setdefault(int(grupos[ys[0], xs[0]]), []).append(indice)

    mejor_grupo = _elegir_grupo(list(por_grupo.values()), etiquetas, stats, config, respuesta)

    mascara = np.isin(etiquetas, mejor_grupo).astype(np.uint8) * 255
    puentes = _coser_fragmentos(etiquetas, mejor_grupo, mascara.shape)
    return mascara, puentes, len(candidatos) - len(mejor_grupo)


def _factor_marana(area: float, lado_mayor: float, ancho_medio: float) -> float:
    """Calcula cuanto se aleja un trazo de ser una linea.

    Una linea cumple ``area = largo x ancho``, de modo que el cociente vale ~1.
    Una red de textura rellena una superficie y el cociente se dispara.

    Args:
        area: Pixeles encendidos.
        lado_mayor: Lado mayor de la caja minima rotada.
        ancho_medio: Ancho local medio, del mapa de distancias.

    Returns:
        El cociente, sin unidades.
    """
    return float(area) / max(float(lado_mayor) * float(ancho_medio), 1.0)


def _elegir_grupo(
    grupos: list[list[int]],
    etiquetas: np.ndarray,
    stats: np.ndarray,
    config: dict[str, Any],
    respuesta: np.ndarray | None,
) -> list[int]:
    """Elige que grupo de fragmentos se mide.

    Por que no vale el area
    -----------------------
    La version anterior se quedaba con el grupo que mas area sumaba, y sobre un
    muro de panete eso elige la textura. Medido sobre la fotografia que destapo
    el fallo, con la grieta y la pared compitiendo dentro de la misma region:

        grupo   area   recorrido   marana   contraste   que es
          5     5319       ~300     6.20       54.2     textura del panete  <- ganaba
          4      794       93.3     2.61       54.4     mancha de textura
          1      714      129.3     1.66       62.9     churrete de pintura
          2      546      111.4     1.61       67.4     LA GRIETA

    La grieta es el trazo mas fino de todos: por area pierde contra cualquier
    mancha, y pierde mas cuanto mas limpia sea la fisura. El area mide cuanta
    tinta hay, no cuanta fisura.

    Los tres rasgos, y por que hacen falta los tres
    -----------------------------------------------
    Una fisura es **larga, oscura y de un solo trazo**. Cada rasgo por separado
    se deja enganar, y esta comprobado sobre las 30 fotografias propias mas esta:

    - **Solo contraste**: acierta aqui, pero en otras dos abandona una grieta de
      1465 px para irse a una mota oscura de veinte.
    - **Solo recorrido**: la mancha de textura tambien es larga, y gana.
    - **Descartar por marana**: no hay corte posible. La textura de aqui da 6.20,
      pero una grieta autentica y ramificada de otra foto da 5.13.

    Se combinan **normalizando los tres dentro de la propia fotografia** y
    sumando. La normalizacion importa: el contraste del black-hat vale ~60 en una
    imagen y ~10 en otra segun la luz, asi que solo tiene sentido comparado con
    los demas trazos de esa misma foto. Sumar en vez de multiplicar evita que un
    rasgo alto tape a los otros dos, que es justo lo que dejaba ganar al churrete
    de pintura por un 5%.

    Resultado sobre las 31 fotografias: elige lo mismo que el area en 24, mejor
    en 4 -incluida esta- y algo mas corto sobre la misma grieta en 1, revisadas
    una a una sobre la imagen.

    Args:
        grupos: Listas de indices de componente, un elemento por grupo.
        etiquetas: Matriz de etiquetas de componentes conexos.
        stats: Estadisticas de cada componente.
        config: Configuracion del proyecto.
        respuesta: Imagen del black-hat. Sin ella no hay medida de contraste y se
            decide por area, como antes.

    Returns:
        Los indices del grupo elegido.
    """
    import cv2

    if respuesta is None or len(grupos) == 1:
        return max(grupos, key=lambda g: sum(int(stats[i, cv2.CC_STAT_AREA]) for i in g))

    rasgos = []
    for miembros in grupos:
        mascara = np.isin(etiquetas, miembros).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            rasgos.append((0.0, 0.0, 0.0))
            continue
        (_, _), (ancho, alto), _ = cv2.minAreaRect(np.vstack(contornos))
        recorrido = max(float(ancho), float(alto))
        distancias = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
        ancho_medio = 2.0 * float(distancias[mascara > 0].mean())
        marana = _factor_marana(int(mascara.sum()), recorrido, ancho_medio)
        rasgos.append((recorrido, float(np.median(respuesta[mascara > 0])), 1.0 / max(marana, 1.0)))

    valores = np.array(rasgos, dtype=float)
    rango = np.ptp(valores, axis=0)
    rango[rango == 0] = 1.0  # un rasgo igual en todos no desempata
    puntos = ((valores - valores.min(axis=0)) / rango).sum(axis=1)
    return grupos[int(np.argmax(puntos))]


def _coser_fragmentos(
    etiquetas: np.ndarray, miembros: list[int], forma: tuple[int, ...]
) -> np.ndarray:
    """Traza los tramos que unen fragmentos contiguos de la misma fisura.

    Agrupar los fragmentos no basta para medirlos: el eje central seguiria
    partido y el recorrido no podria recorrerse de un extremo al otro. Hacen
    falta tramos explicitos que los cosan.

    Se unen con el arbol de recubrimiento minimo sobre las distancias entre
    fragmentos: cada trozo se enlaza con el mas cercano que ya este unido, y no
    con todos, de modo que la grieta se recompone como una linea y no como una
    maraña de atajos.

    Los tramos se devuelven **aparte de la mascara**. La distincion importa:
    sobre ellos no se observo ninguna fisura, solo se infirio que continuaba, asi
    que cuentan para el recorrido pero no para el ancho, del que no hay evidencia.

    Args:
        etiquetas: Matriz de etiquetas de componentes conexos.
        miembros: Indices de los fragmentos a coser.
        forma: Alto y ancho de la mascara.

    Returns:
        Mascara con los tramos de union trazados.
    """
    import cv2

    puentes = np.zeros(forma[:2], dtype=np.uint8)
    if len(miembros) < 2:
        return puentes

    # Los contornos bastan para medir distancias entre fragmentos, y se submuestrean
    # porque la precision de un pixel no cambia por donde pasa el tramo de union.
    contornos: dict[int, np.ndarray] = {}
    for indice in miembros:
        trozos, _ = cv2.findContours(
            (etiquetas == indice).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        puntos = np.vstack([t.reshape(-1, 2) for t in trozos])
        contornos[indice] = puntos[:: max(1, len(puntos) // 120)]

    unidos = {miembros[0]}
    pendientes = set(miembros[1:])
    while pendientes:
        mejor: tuple[float, int, tuple[int, int], tuple[int, int]] | None = None
        for dentro in unidos:
            a = contornos[dentro]
            for fuera in pendientes:
                b = contornos[fuera]
                distancias = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
                plano = int(np.argmin(distancias))
                i, j = divmod(plano, distancias.shape[1])
                valor = float(distancias[i, j])
                if mejor is None or valor < mejor[0]:
                    mejor = (valor, fuera, tuple(a[i]), tuple(b[j]))

        if mejor is None:
            break
        _, fuera, punto_a, punto_b = mejor
        cv2.line(
            puentes, (int(punto_a[0]), int(punto_a[1])), (int(punto_b[0]), int(punto_b[1])), 255, 1
        )
        unidos.add(fuera)
        pendientes.discard(fuera)

    return puentes


def esqueletizar(mascara: np.ndarray) -> np.ndarray:
    """Reduce la fisura a su eje central de un pixel de grosor.

    Implementa el algoritmo de **Zhang-Suen**, que adelgaza la forma quitando
    pixeles del borde en dos pasadas alternas y solo cuando hacerlo **no rompe la
    conectividad** ni acorta un extremo. El resultado conserva la topologia
    original: si la grieta se bifurcaba, el eje se bifurca en el mismo sitio.

    Se implementa aqui en lugar de usar ``cv2.ximgproc.thinning`` porque ese
    modulo pertenece a ``opencv-contrib``, que no forma parte del entorno
    congelado del proyecto. Anadirlo por una funcion invalidaria la
    reproducibilidad de todas las metricas ya medidas.

    Args:
        mascara: Mascara binaria con valores 0 y 255.

    Returns:
        Mascara ``uint8`` del mismo tamano con el eje en 255.
    """
    entrada = (np.asarray(mascara) > 0).astype(np.uint8)

    # El adelgazamiento solo puede tocar pixeles que valgan 1, asi que trabajar
    # sobre la caja que los envuelve da exactamente el mismo resultado sobre una
    # superficie menor. En una fisura que ocupa una esquina del encuadre eso es
    # la diferencia entre procesar el recorte entero y procesar una franja.
    filas = np.flatnonzero(entrada.any(axis=1))
    columnas = np.flatnonzero(entrada.any(axis=0))
    if len(filas) == 0:
        return np.zeros_like(entrada)

    y0, y1 = int(filas[0]), int(filas[-1]) + 1
    x0, x1 = int(columnas[0]), int(columnas[-1]) + 1
    img = entrada[y0:y1, x0:x1].copy()

    while True:
        borrados_totales = 0
        for paso in (0, 1):
            p = np.pad(img, 1, mode="constant")
            # Vecinos en el orden del algoritmo: P2 arriba, y despues en sentido
            # horario hasta P9 arriba-izquierda.
            p2, p3 = p[:-2, 1:-1], p[:-2, 2:]
            p4, p5 = p[1:-1, 2:], p[2:, 2:]
            p6, p7 = p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]

            vecinos = [p2, p3, p4, p5, p6, p7, p8, p9]
            cuantos = sum(vecinos)
            # Transiciones 0 -> 1 recorriendo los vecinos en circulo. Que haya
            # exactamente una es lo que garantiza no partir la figura en dos.
            circulo = [*vecinos, p2]
            transiciones = sum(
                ((circulo[i] == 0) & (circulo[i + 1] == 1)).astype(np.uint8) for i in range(8)
            )

            if paso == 0:
                condicion = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condicion = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            borrar = (img == 1) & (cuantos >= 2) & (cuantos <= 6) & (transiciones == 1) & condicion
            borrados = int(borrar.sum())
            if borrados:
                img[borrar] = 0
                borrados_totales += borrados

        if borrados_totales == 0:
            break

    salida = np.zeros_like(entrada)
    salida[y0:y1, x0:x1] = img
    return (salida * 255).astype(np.uint8)


def podar_espolones(
    esqueleto: np.ndarray, longitud_minima: int = 12, iteraciones: int = 6
) -> np.ndarray:
    """Elimina del eje las puas cortas que no corresponden a ramificaciones reales.

    Por que hace falta
    ------------------
    La esqueletizacion es fiel a la forma que recibe, y el borde de una grieta
    segmentada es rugoso: cada entrante del contorno genera una pua de dos o tres
    pixeles que sale del eje principal. Geometricamente son bifurcaciones, y el
    algoritmo las cuenta como tales.

    El efecto medido sobre una fotografia real del equipo fue una grieta unica y
    bien trazada que se reportaba con **432 ramificaciones** y un recorrido
    inflado por la suma de todas las puas. El trazado era correcto; la lectura,
    absurda. Sin podar, ni el numero de ramificaciones ni la longitud significan
    nada.

    Como funciona
    -------------
    Se retiran los puntos de bifurcacion, lo que parte el eje en segmentos
    sueltos. Un segmento es una **pua** si es corto y tiene un extremo libre: sale
    del eje y no lleva a ninguna parte. Una ramificacion autentica es larga, o
    conecta dos partes del eje. Se repite el proceso porque al podar una pua
    pueden aparecer otras que antes quedaban ocultas detras de ella.

    Args:
        esqueleto: Mascara del eje central.
        longitud_minima: Pixeles por debajo de los cuales un segmento terminal se
            considera pua y no ramificacion.
        iteraciones: Tope de pasadas. Evita un bucle interminable ante formas
            patologicas.

    Returns:
        Mascara del eje podado.
    """
    import cv2

    eje = (np.asarray(esqueleto) > 0).astype(np.uint8)

    for _ in range(max(1, iteraciones)):
        vecinos = _contar_vecinos(eje * 255)
        bifurcaciones = (vecinos >= 3) & (eje > 0)
        # Al quitar las bifurcaciones, el eje se separa en tramos independientes.
        tramos = ((eje > 0) & ~bifurcaciones).astype(np.uint8)
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(tramos, connectivity=8)

        # Todo lo que sigue se resuelve con operaciones sobre el conjunto
        # completo de tramos, y no recorriendolos uno a uno. La diferencia no es
        # cosmetica: la version anterior comparaba cada tramo contra la imagen
        # entera, y con varios miles de tramos la poda tardaba 4.4 segundos, mas
        # que todo el resto de la medicion junta.
        cortos = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < longitud_minima)
        cortos = cortos[cortos != 0]  # la etiqueta 0 es el fondo
        if len(cortos) == 0:
            break

        # Un tramo es una pua si termina en el aire. Si no tiene ningun extremo
        # libre, une dos bifurcaciones y forma parte de la estructura.
        con_extremo_libre = np.unique(etiquetas[(vecinos == 1) & (eje > 0)])
        a_podar = np.intersect1d(cortos, con_extremo_libre, assume_unique=False)
        if len(a_podar) == 0:
            break

        eje[np.isin(etiquetas, a_podar)] = 0
        # Una bifurcacion que se queda sin nada que unir deja de serlo.
        eje[(_contar_vecinos(eje * 255) == 0) & (eje > 0)] = 0

    return (eje * 255).astype(np.uint8)


def _contar_vecinos(esqueleto: np.ndarray) -> np.ndarray:
    """Cuenta cuantos vecinos del eje tiene cada pixel del eje.

    Args:
        esqueleto: Mascara del eje central.

    Returns:
        Matriz de enteros con el numero de vecinos de cada pixel, 0 fuera del eje.
    """
    b = (np.asarray(esqueleto) > 0).astype(np.uint8)
    p = np.pad(b, 1, mode="constant")
    vecinos = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )
    return (vecinos * b).astype(np.int32)


def camino_principal(esqueleto: np.ndarray) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Encuentra el recorrido mas largo del eje: la trayectoria de la grieta.

    Por que no basta con sumar todo el eje
    --------------------------------------
    Una grieta real sobre pañete o mortero no es una linea limpia: su eje sale
    con decenas de ramas laterales, porque la fisura arrastra consigo la textura
    rugosa del material. Sumar la longitud de **todas** ellas no responde a la
    pregunta "cuanto mide esta grieta" — responde a "cuanto material fisurado
    hay", que es otra cosa y ademas depende del ruido de la segmentacion.

    Medido sobre una fotografia del equipo, el eje completo daba 1544 px y una
    tortuosidad de 8.51. Una tortuosidad de 8.5 significaria que la grieta
    serpentea recorriendo ocho veces y media la distancia entre sus extremos, lo
    cual no describe nada fisico: es el resultado de sumar un arbol como si fuera
    un camino.

    Lo que si responde a la pregunta es **el trayecto mas largo** que se puede
    recorrer por el eje sin repetir camino. Es la trayectoria principal de la
    fisura, y las ramas laterales quedan como lo que son: un rasgo aparte, que se
    cuenta pero no se suma.

    Se calcula con el metodo clasico del diametro de un grafo: se parte de un
    punto cualquiera, se busca el mas lejano, y desde ese se vuelve a buscar el
    mas lejano. El segundo recorrido es el mas largo posible.

    Args:
        esqueleto: Mascara del eje central.

    Returns:
        Tupla ``(longitud, extremo_a, extremo_b)`` con la longitud en pixeles y
        las coordenadas ``(x, y)`` de los dos extremos. Devuelve ceros si el eje
        tiene menos de dos pixeles.
    """
    import heapq

    puntos = np.argwhere(np.asarray(esqueleto) > 0)
    if len(puntos) < 2:
        return 0.0, (0, 0), (0, 0)

    indice_de = {(int(y), int(x)): i for i, (y, x) in enumerate(puntos)}
    raiz_dos = math.sqrt(2.0)
    vecindad = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def mas_lejano(origen: int) -> tuple[int, np.ndarray]:
        """Dijkstra desde un pixel del eje.

        Args:
            origen: Indice del pixel de partida.

        Returns:
            Tupla ``(indice_mas_lejano, distancias)``.
        """
        distancias = np.full(len(puntos), np.inf)
        distancias[origen] = 0.0
        cola: list[tuple[float, int]] = [(0.0, origen)]
        while cola:
            distancia, actual = heapq.heappop(cola)
            if distancia > distancias[actual]:
                continue
            y, x = int(puntos[actual][0]), int(puntos[actual][1])
            for dy, dx in vecindad:
                vecino = indice_de.get((y + dy, x + dx))
                if vecino is None:
                    continue
                paso = raiz_dos if dy and dx else 1.0
                if distancia + paso < distancias[vecino]:
                    distancias[vecino] = distancia + paso
                    heapq.heappush(cola, (distancias[vecino], vecino))
        finitas = np.where(np.isfinite(distancias), distancias, -1.0)
        return int(np.argmax(finitas)), distancias

    extremo_a, _ = mas_lejano(0)
    extremo_b, distancias = mas_lejano(extremo_a)
    longitud = float(distancias[extremo_b])
    if not math.isfinite(longitud):
        longitud = 0.0

    punto_a = (int(puntos[extremo_a][1]), int(puntos[extremo_a][0]))
    punto_b = (int(puntos[extremo_b][1]), int(puntos[extremo_b][0]))
    return longitud, punto_a, punto_b


def _longitud_del_eje(esqueleto: np.ndarray) -> float:
    """Mide el recorrido del eje sumando los pasos entre pixeles contiguos.

    Contar pixeles subestima el recorrido: un tramo diagonal de diez pixeles
    mide 14.1 y no 10. Aqui cada conexion entre vecinos aporta su longitud real,
    1 en horizontal y vertical y raiz de dos en diagonal.

    Args:
        esqueleto: Mascara del eje central.

    Returns:
        Longitud en pixeles.
    """
    b = (np.asarray(esqueleto) > 0).astype(np.uint8)
    # Cada par de vecinos se cuenta una sola vez mirando solo hacia adelante.
    horizontales = int((b[:, :-1] & b[:, 1:]).sum())
    verticales = int((b[:-1, :] & b[1:, :]).sum())
    diagonal_a = int((b[:-1, :-1] & b[1:, 1:]).sum())
    diagonal_b = int((b[:-1, 1:] & b[1:, :-1]).sum())
    return float(horizontales + verticales) + math.sqrt(2.0) * float(diagonal_a + diagonal_b)


def _medir_sobre(
    mascara: np.ndarray,
    puentes: np.ndarray,
    fragmentos: int,
    config: dict[str, Any],
    escala_mm_por_px: float | None = None,
) -> MedidasGrieta:
    """Calcula la geometria a partir de una mascara ya aislada.

    Se separo de :func:`medir_grieta` para poder medir dos veces la misma
    fotografia: una sobre los fragmentos unidos y otra, si aquello no resulto
    plausible, sobre el trozo dominante solo.

    Args:
        mascara: Pixeles de la fisura.
        puentes: Tramos inferidos entre fragmentos. Cuentan para el recorrido
            pero no para el ancho, del que no hay evidencia sobre ellos.
        fragmentos: Cuantos candidatos quedaron fuera del grupo medido.
        config: Configuracion del proyecto.
        escala_mm_por_px: Milimetros que representa cada pixel, obtenido de
            :func:`src.vision.escala.detectar_escala` sobre la fotografia
            completa. Si es ``None``, las medidas se quedan en pixeles y los
            campos en milimetros valen ``None``: **el modulo no estima la escala
            por su cuenta bajo ninguna circunstancia**.

            El recorte debe proceder de un corte directo de la fotografia, sin
            redimensionar, o la equivalencia dejaria de ser valida.

    Returns:
        Un :class:`MedidasGrieta`. Si no se aisla nada con area suficiente,
        devuelve ``detectada=False`` con el motivo, sin lanzar excepcion: que no
        se pueda medir es un resultado legitimo, no un error.
    """
    import cv2

    area = int((mascara > 0).sum())
    alto, ancho = mascara.shape[:2]

    if area == 0:
        return MedidasGrieta(
            detectada=False,
            motivo=(
                "No se aislo ninguna estructura fina y alargada. Puede que la fisura sea "
                "demasiado tenue para el contraste de la foto, o que lo detectado por el "
                "clasificador sea una mancha y no una grieta."
            ),
        )

    # El eje se calcula sobre la mascara MAS los puentes, para que el recorrido
    # pueda seguirse de un extremo al otro de la grieta aunque el umbral la haya
    # partido por el camino.
    esqueleto = podar_espolones(
        esqueletizar(cv2.bitwise_or(mascara, puentes)),
        longitud_minima=int(obtener(config, "medicion.poda_espolones_px", 12)),
    )
    if int((esqueleto > 0).sum()) < 2:
        return MedidasGrieta(
            detectada=False,
            area_px=area,
            motivo="La region aislada es demasiado compacta para tener un eje medible.",
        )

    # El resultado de distanceTransform en un punto es su distancia al fondo mas
    # cercano, es decir, el radio de la fisura ahi. El ancho es el doble.
    distancias = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    # El ancho se mide SOLO donde se observo fisura, nunca sobre los puentes:
    # ahi no se vio nada, solo se infirio que la grieta continuaba, y asignarles
    # un ancho seria inventarlo. Contarlos hundiria ademas el ancho medio, porque
    # su distancia al fondo es cero.
    puntos = np.argwhere((esqueleto > 0) & (mascara > 0))  # filas (y, x)
    if len(puntos) < 2:
        puntos = np.argwhere(esqueleto > 0)
    anchos = 2.0 * distancias[puntos[:, 0], puntos[:, 1]]

    # Correccion del sesgo de segmentacion.
    #
    # La mascara es sistematicamente mas ancha que la fisura: el cierre
    # morfologico anade un pixel por lado y el umbral incluye la transicion del
    # borde. Medido sobre lineas sinteticas de ancho conocido, el sesgo resulto
    # ser una CONSTANTE, no un porcentaje:
    #
    #     ancho real   3     5     7     9    11
    #     medido       6.00  8.00 10.00 12.00 14.00
    #     sesgo       +3.00 +3.00 +3.00 +3.00 +3.00
    #
    # Que sea constante es lo que permite corregirlo restando. Y que importe es
    # evidente en el extremo del rango: sin corregir, una fisura de 3 px se
    # reporta con el doble de su ancho, y ese es justo el tamano donde la
    # diferencia entre "capilar" y "grieta" se decide.
    sesgo = float(obtener(config, "medicion.sesgo_ancho_px", 3.0))
    anchos = np.maximum(anchos - sesgo, 0.0)

    indice_ancho = int(np.argmax(anchos))
    punto_ancho = (int(puntos[indice_ancho, 1]), int(puntos[indice_ancho, 0]))

    # La longitud de la grieta es su TRAYECTORIA PRINCIPAL, no la suma de todo el
    # eje. Una fisura sobre panete arrastra la textura del material y su eje sale
    # con decenas de ramas laterales; sumarlas responderia a "cuanto material
    # fisurado hay" y no a "cuanto mide esta grieta".
    longitud, extremo_a, extremo_b = camino_principal(esqueleto)
    longitud_total = _longitud_del_eje(esqueleto)

    vecinos = _contar_vecinos(esqueleto)
    extremos = [(int(x), int(y)) for y, x in np.argwhere(vecinos == 1)]
    indice_ramificacion = float(longitud_total / longitud) if longitud > 1e-6 else 1.0

    # Tortuosidad sobre la trayectoria principal: cuanto serpentea respecto a la
    # linea recta que une sus dos extremos. Vale 1.0 en una grieta recta y no
    # puede bajar de ahi, lo que la convierte en una comprobacion de coherencia:
    # un valor menor que 1 delata que se esta midiendo un conjunto de fragmentos
    # y no una fisura conectada.
    separacion = math.dist(extremo_a, extremo_b)
    tortuosidad = float(longitud / separacion) if separacion > 1e-6 else 0.0

    orientacion = _orientacion_principal(mascara)

    # La conversion se hace UNA sola vez y solo si hay escala. No hay valor por
    # defecto ni estimacion: sin marcador, los campos en milimetros quedan en
    # None y quien los lea sabe que ese dato no existe.
    convertir = (
        (lambda px: float(px) * escala_mm_por_px) if escala_mm_por_px else (lambda _px: None)
    )

    return MedidasGrieta(
        detectada=True,
        longitud_px=float(longitud),
        longitud_total_px=float(longitud_total),
        extremos_principales=(extremo_a, extremo_b),
        ancho_maximo_px=float(anchos.max()),
        ancho_medio_px=float(anchos.mean()),
        ancho_p95_px=float(np.percentile(anchos, 95)),
        area_px=area,
        orientacion_grados=orientacion,
        tortuosidad=tortuosidad,
        indice_ramificacion=indice_ramificacion,
        extremos=extremos[:8],
        punto_mas_ancho=punto_ancho,
        cobertura=float(area) / float(alto * ancho),
        longitud_inferida_px=float(_longitud_del_eje(cv2.bitwise_and(esqueleto, puentes))),
        fragmentos_descartados=fragmentos,
        escala_mm_por_px=escala_mm_por_px,
        longitud_mm=convertir(longitud),
        ancho_medio_mm=convertir(anchos.mean()),
        ancho_maximo_mm=convertir(anchos.max()),
    )


def _parece_una_fisura(medidas: MedidasGrieta, config: dict[str, Any]) -> str | None:
    """Comprueba que lo medido pueda ser realmente una grieta.

    Por que hace falta
    ------------------
    La union de fragmentos recompone una grieta partida por el umbral, pero sobre
    un muro de panete rugoso puede hacer algo muy distinto: encadenar decenas de
    trozos de textura en una figura que no describe ninguna fisura. Y el problema
    es que **el resultado sigue teniendo el aspecto de un numero**: recorrido,
    ancho, todo con dos decimales.

    Las propias medidas delatan cuando eso ocurre. Una grieta serpentea, pero no
    recorre siete veces la distancia entre sus extremos; se ramifica, pero no se
    multiplica por veinte. Cuando esas cifras salen del rango de lo fisicamente
    plausible, lo medido no es una fisura.

    Es el mismo criterio que los guardarraíles de la inclinometria (§4.5), que
    rechazan un desaplome de 34 grados sin necesidad de saber que edificio es:
    hay valores que simplemente no describen la realidad que se pretende medir.

    Args:
        medidas: Geometria obtenida.
        config: Configuracion del proyecto.

    Returns:
        Motivo por el que no parece una fisura, o ``None`` si es plausible.
    """
    cfg = obtener(config, "medicion.verosimilitud", {}) or {}
    tortuosidad_maxima = float(cfg.get("tortuosidad_maxima", 3.0))
    ramificacion_maxima = float(cfg.get("ramificacion_maxima", 6.0))

    if medidas.tortuosidad > tortuosidad_maxima:
        return (
            f"el recorrido serpentea {medidas.tortuosidad:.1f} veces la distancia entre "
            f"sus extremos (el limite es {tortuosidad_maxima:.1f})"
        )
    if medidas.indice_ramificacion > ramificacion_maxima:
        return (
            f"el trazado se ramifica {medidas.indice_ramificacion:.1f} veces su propio "
            f"recorrido (el limite es {ramificacion_maxima:.1f})"
        )
    return None


def _componente_dominante(mascara: np.ndarray) -> np.ndarray:
    """Se queda con el trozo de mayor area de una mascara.

    Args:
        mascara: Mascara binaria, posiblemente con varios componentes.

    Returns:
        Mascara con un unico componente.
    """
    import cv2

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(
        (np.asarray(mascara) > 0).astype(np.uint8), connectivity=8
    )
    if n <= 2:
        return np.asarray(mascara).copy()
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (etiquetas == mayor).astype(np.uint8) * 255


def medir_grieta(
    imagen_bgr: np.ndarray, config: dict[str, Any], escala_mm_por_px: float | None = None
) -> MedidasGrieta:
    """Extrae la geometria de la fisura presente en un recorte.

    Mide primero sobre los fragmentos unidos, que es lo correcto cuando el umbral
    ha partido una grieta en trozos. Despues **comprueba que el resultado pueda
    ser una fisura**: si el recorrido serpentea o se ramifica mas alla de lo
    fisicamente plausible, la union encadeno textura en lugar de recomponer una
    grieta, y entonces se vuelve a medir sobre el trozo dominante solo.

    Preferir el trozo dominante no es la respuesta general -corta las grietas
    reales, que es justo el fallo que la union vino a resolver- pero si es la
    respuesta correcta cuando la alternativa es medir la textura del muro.

    Args:
        imagen_bgr: Recorte en BGR donde el clasificador detecto grieta.
        config: Configuracion del proyecto.
        escala_mm_por_px: Milimetros por pixel, de
            :func:`src.vision.escala.detectar_escala`. Sin el, las medidas quedan
            en pixeles: el modulo no estima la escala por su cuenta.

    Returns:
        Un :class:`MedidasGrieta`.
    """
    import cv2

    imagen = np.asarray(imagen_bgr)
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    mascara, puentes, fragmentos = segmentar_grieta(imagen, config)
    medidas = _medir_sobre(mascara, puentes, fragmentos, config, escala_mm_por_px)

    # Los dos detectores se turnan segun la pared, y cual gana no se puede saber
    # de antemano: el de crestas recupera fisuras finas que Otsu funde con la
    # textura -en dos fotografias del conjunto, Otsu no encontraba nada y este
    # las mide-, pero sobre una pared lisa y bien iluminada Otsu sigue el trazo
    # mas lejos. Se miden los dos y se conserva el recorrido mas largo que
    # ademas sea verosimil; ninguno puede empeorar al otro, solo mejorarlo.
    if (obtener(config, "medicion.cresta", {}) or {}).get("activo", True):
        otro = {
            **config,
            "medicion": {**(obtener(config, "medicion", {}) or {}), "cresta": {"activo": False}},
        }
        m2, p2, f2 = segmentar_grieta(imagen, otro)
        alterna = _medir_sobre(m2, p2, f2, config, escala_mm_por_px)
        mejor_alterna = alterna.detectada and _parece_una_fisura(alterna, config) is None
        peor_actual = not medidas.detectada or _parece_una_fisura(medidas, config) is not None
        if mejor_alterna and (peor_actual or alterna.longitud_px > medidas.longitud_px):
            mascara, puentes, fragmentos, medidas = m2, p2, f2, alterna

    if not medidas.detectada:
        return medidas

    motivo = _parece_una_fisura(medidas, config)
    if motivo is None:
        return medidas

    solo_dominante = _componente_dominante(mascara)
    alternativa = _medir_sobre(
        solo_dominante, np.zeros_like(puentes), fragmentos, config, escala_mm_por_px
    )
    if not alternativa.detectada:
        return medidas

    return replace(
        alternativa,
        union_rechazada=(
            f"Se descarto unir los fragmentos porque {motivo}. Se midio solo el trozo "
            "principal, asi que el recorrido puede quedarse corto si la grieta seguia "
            "mas alla."
        ),
    )


def _orientacion_principal(mascara: np.ndarray) -> float | None:
    """Calcula el angulo del eje dominante de la fisura.

    Se usa analisis de componentes principales sobre las coordenadas de la
    mascara: la direccion en la que los puntos se dispersan mas es la direccion
    de la grieta. Es mas estable que ajustar una recta, porque no se deja
    arrastrar por una ramificacion corta.

    Args:
        mascara: Mascara binaria de la fisura.

    Returns:
        Angulo en grados respecto a la horizontal, en ``[-90, 90)``, o ``None``
        si no hay puntos suficientes.
    """
    puntos = np.argwhere(np.asarray(mascara) > 0).astype(np.float64)
    if len(puntos) < 2:
        return None

    centradas = puntos - puntos.mean(axis=0)
    # columnas (x) como eje horizontal, filas (y) como vertical
    coords = np.stack([centradas[:, 1], centradas[:, 0]], axis=1)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    dx, dy = vt[0]
    angulo = math.degrees(math.atan2(dy, dx))
    # Una recta y la misma recta girada 180 grados son la misma recta.
    if angulo >= 90.0:
        angulo -= 180.0
    if angulo < -90.0:
        angulo += 180.0
    return float(angulo)


def clasificar_forma(medidas: MedidasGrieta, config: dict[str, Any]) -> str:
    """Describe la forma de la fisura en terminos interpretables.

    La forma no decide por si sola la gravedad —para eso hace falta el ancho en
    unidades fisicas, que esta fase no puede dar—, pero si aporta un indicio
    util: una fisura de retraccion tiende a ser corta, fina y recta, mientras que
    una de origen estructural suele ser larga y ramificarse al propagarse.

    Args:
        medidas: Geometria extraida.
        config: Configuracion del proyecto.

    Returns:
        Etiqueta descriptiva. ``"indeterminada"`` si no hay medidas.
    """
    if not medidas.detectada:
        return "indeterminada"

    cfg = obtener(config, "medicion.forma", {}) or {}
    if medidas.indice_ramificacion >= float(cfg.get("indice_ramificada", 2.0)):
        return "ramificada"
    if medidas.tortuosidad >= float(cfg.get("tortuosidad_sinuosa", 1.35)):
        return "sinuosa"
    if medidas.longitud_px >= float(cfg.get("longitud_larga_px", 200)):
        return "lineal extensa"
    return "lineal corta"


def anotar_medidas(
    imagen_bgr: np.ndarray,
    medidas: MedidasGrieta,
    mascara: np.ndarray | None = None,
    puentes: np.ndarray | None = None,
) -> np.ndarray:
    """Dibuja sobre el recorte la fisura medida y sus magnitudes.

    Mostrar la mascara es tan importante como mostrar los numeros: permite ver de
    un vistazo si el sistema midio la grieta o midio una sombra. Una cifra sin su
    evidencia visual no se puede auditar.

    Args:
        imagen_bgr: Recorte original en BGR.
        medidas: Geometria extraida.
        mascara: Mascara de la fisura. Si es ``None``, solo se escriben los datos.
        puentes: Tramos cosidos entre fragmentos. Se dibujan en otro color
            porque no son fisura observada sino continuidad inferida, y quien
            mire la imagen tiene derecho a distinguir una cosa de la otra.

    Returns:
        Copia anotada en BGR.
    """
    import cv2

    lienzo = np.asarray(imagen_bgr).copy()
    if lienzo.ndim == 2:
        lienzo = cv2.cvtColor(lienzo, cv2.COLOR_GRAY2BGR)

    if mascara is not None and medidas.detectada:
        # La fisura se tine de naranja translucido: se ve la mascara y tambien la
        # textura que hay debajo, que es lo que permite juzgar si acerto.
        capa = lienzo.copy()
        capa[mascara > 0] = (40, 170, 250)
        lienzo = cv2.addWeighted(capa, 0.45, lienzo, 0.55, 0)

        completa = mascara if puentes is None else cv2.bitwise_or(mascara, puentes)
        esqueleto = podar_espolones(esqueletizar(completa))
        lienzo[(esqueleto > 0) & (mascara > 0)] = (30, 30, 220)
        if puentes is not None:
            # Amarillo para lo inferido: sobre estos tramos no se observo fisura.
            lienzo[(esqueleto > 0) & (mascara == 0)] = (60, 230, 250)

    if medidas.punto_mas_ancho is not None:
        radio = max(4, int(medidas.ancho_maximo_px))
        cv2.circle(lienzo, medidas.punto_mas_ancho, radio, (30, 30, 220), 2, cv2.LINE_AA)

    for extremo in medidas.extremos:
        cv2.circle(lienzo, extremo, 3, (60, 220, 60), -1, cv2.LINE_AA)

    texto = medidas.resumen() if medidas.detectada else (medidas.motivo or "Sin fisura medible")
    # La fuente de OpenCV solo cubre ASCII: cualquier otro caracter sale como
    # interrogantes. Se transliteran los acentos y se sustituyen los separadores
    # tipograficos, en lugar de dejar que la anotacion salga ilegible.
    texto = (
        unicodedata.normalize("NFKD", texto.replace("·", "|"))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    for desplazamiento, color in (((1, 1), (0, 0, 0)), ((0, 0), (255, 255, 255))):
        cv2.putText(
            lienzo,
            texto[:70],
            (8 + desplazamiento[0], lienzo.shape[0] - 10 + desplazamiento[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return lienzo
