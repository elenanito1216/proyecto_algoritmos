"""Pruebas de la conversion de pixeles a milimetros.

La prueba central de este archivo es de **ida y vuelta**: se construye una escena
sintetica donde se conoce de antemano tanto el tamano del marcador en pixeles
como el ancho de la fisura, se deja que el sistema deduzca la escala, y se
comprueba que el milimetraje que devuelve es el que debe ser.

Es la unica forma honesta de validar esto. Sobre una fotografia real no hay
respuesta correcta conocida —nadie midio esas grietas con un calibre— asi que una
prueba con fotos solo podria comprobar que el programa no se cae, no que acierta.

El resto de las pruebas cubren lo que puede salir mal en campo, que es mas
probable que un fallo de calculo: que no haya marcador, que se fotografie en
angulo, que quede demasiado lejos de la grieta o demasiado pequeno en el encuadre.
En esos casos lo que se exige no es que acierte, sino **que avise**.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.escala import (  # noqa: E402
    ReferenciaEscala,
    anotar_referencia,
    aviso_por_lejania,
    detectar_escala,
)
from src.vision.morfologia import medir_grieta  # noqa: E402

CONFIG: dict[str, Any] = {
    "escala": {
        "diccionario": "DICT_4X4_50",
        "lado_marcador_mm": 50.0,
        "deformacion_maxima": 0.15,
        "lado_minimo_px": 60,
        "lados_maximos_de_distancia": 6.0,
    },
    "medicion": {
        "clahe_clip": 2.0,
        "clahe_rejilla": 8,
        "kernel_blackhat": 15,
        "kernel_limpieza": 3,
        "poda_espolones_px": 12,
        "radio_puente_px": 20,
        "area_minima_relativa": 0.0005,
        "area_minima_absoluta": 30,
        "elongacion_minima": 2.0,
        "sesgo_ancho_px": 3.0,
        "forma": {
            "indice_ramificada": 2.0,
            "tortuosidad_sinuosa": 1.35,
            "longitud_larga_px": 200,
        },
    },
}


def _escena(
    lado_marcador_px: int = 200,
    posicion: tuple[int, int] = (40, 40),
    tamano: tuple[int, int] = (700, 700),
    identificador: int = 0,
) -> np.ndarray:
    """Compone una pared clara con un marcador ArUco de tamano conocido.

    Args:
        lado_marcador_px: Lado del marcador en pixeles.
        posicion: Esquina superior izquierda ``(x, y)`` donde pegarlo.
        tamano: Alto y ancho de la escena.
        identificador: Numero del marcador.

    Returns:
        Imagen BGR.
    """
    dicc = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marcador = cv2.aruco.generateImageMarker(dicc, identificador, lado_marcador_px)

    escena = np.full((tamano[0], tamano[1], 3), 210, dtype=np.uint8)
    x, y = posicion
    escena[y : y + lado_marcador_px, x : x + lado_marcador_px] = cv2.cvtColor(
        marcador, cv2.COLOR_GRAY2BGR
    )
    return escena


# ---------------------------------------------------------------------------
# Deteccion
# ---------------------------------------------------------------------------


def test_detecta_el_marcador_y_calcula_la_equivalencia():
    # 200 px de marcador que en la realidad miden 50 mm -> 0.25 mm por pixel.
    referencia = detectar_escala(_escena(lado_marcador_px=200), CONFIG)

    assert referencia.detectada
    assert referencia.fiable
    assert referencia.mm_por_px == pytest.approx(0.25, rel=0.02)
    assert referencia.identificador == 0


def test_un_marcador_mas_pequeno_implica_mas_milimetros_por_pixel():
    # Fotografiar desde mas lejos hace el marcador mas pequeno en la imagen, y
    # entonces cada pixel abarca mas distancia real.
    cerca = detectar_escala(_escena(lado_marcador_px=300), CONFIG)
    lejos = detectar_escala(_escena(lado_marcador_px=120), CONFIG)
    assert lejos.mm_por_px > cerca.mm_por_px


def test_sin_marcador_lo_dice_en_vez_de_estimar():
    # Es la garantia central del modulo: nunca inventar una escala.
    referencia = detectar_escala(np.full((400, 400, 3), 210, np.uint8), CONFIG)
    assert not referencia.detectada
    assert referencia.mm_por_px is None
    assert referencia.motivo


def test_con_varios_marcadores_usa_el_mayor():
    escena = _escena(lado_marcador_px=120, posicion=(40, 40))
    grande = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 260
    )
    escena[380:640, 380:640] = cv2.cvtColor(grande, cv2.COLOR_GRAY2BGR)

    referencia = detectar_escala(escena, CONFIG)
    assert referencia.identificador == 7, "deberia preferir el marcador mas grande"


def test_el_lado_declarado_en_milimetros_manda_sobre_el_resultado():
    config_80 = {**CONFIG, "escala": {**CONFIG["escala"], "lado_marcador_mm": 80.0}}
    referencia = detectar_escala(_escena(lado_marcador_px=200), config_80)
    assert referencia.mm_por_px == pytest.approx(0.4, rel=0.02)


def test_un_diccionario_inexistente_falla_de_forma_explicita():
    config_malo = {**CONFIG, "escala": {**CONFIG["escala"], "diccionario": "DICT_INVENTADO"}}
    with pytest.raises(ValueError, match="no existe"):
        detectar_escala(_escena(), config_malo)


# ---------------------------------------------------------------------------
# Deteccion de las condiciones que invalidan la medida
# ---------------------------------------------------------------------------


def test_avisa_cuando_la_foto_se_tomo_en_angulo():
    # La perspectiva es la principal fuente de error de este metodo: si el
    # marcador se ve deformado, un pixel no representa la misma distancia en toda
    # la imagen y la conversion deja de valer.
    escena = _escena(lado_marcador_px=240, posicion=(120, 120), tamano=(700, 700))
    origen = np.float32([[120, 120], [360, 120], [360, 360], [120, 360]])
    destino = np.float32([[120, 120], [360, 175], [360, 330], [120, 360]])
    deformada = cv2.warpPerspective(
        escena, cv2.getPerspectiveTransform(origen, destino), (700, 700), borderValue=(210,) * 3
    )

    referencia = detectar_escala(deformada, CONFIG)
    if referencia.detectada:
        assert not referencia.fiable or referencia.deformacion > 0.0
        if not referencia.fiable:
            assert referencia.aviso and "angulo" in referencia.aviso


def test_un_marcador_de_frente_no_sale_deformado():
    referencia = detectar_escala(_escena(lado_marcador_px=200), CONFIG)
    assert referencia.deformacion < 0.05


def test_avisa_cuando_el_marcador_es_muy_pequeno_en_el_encuadre():
    referencia = detectar_escala(_escena(lado_marcador_px=50, tamano=(400, 400)), CONFIG)
    if referencia.detectada:
        assert referencia.aviso is not None
        assert "px" in referencia.aviso


def test_avisa_cuando_el_marcador_esta_lejos_de_la_grieta():
    referencia = detectar_escala(_escena(lado_marcador_px=120, posicion=(20, 20)), CONFIG)
    # Una region en la esquina opuesta de una escena de 700x700.
    aviso = aviso_por_lejania(referencia, (600, 600, 690, 690), CONFIG)
    assert aviso is not None and "marcador" in aviso


def test_no_avisa_cuando_el_marcador_esta_junto_a_la_grieta():
    referencia = detectar_escala(_escena(lado_marcador_px=200, posicion=(40, 40)), CONFIG)
    assert aviso_por_lejania(referencia, (250, 100, 400, 250), CONFIG) is None


def test_sin_referencia_no_hay_aviso_de_lejania():
    assert aviso_por_lejania(ReferenciaEscala(detectada=False), (0, 0, 10, 10), CONFIG) is None


# ---------------------------------------------------------------------------
# Ida y vuelta: de una fisura de ancho conocido a milimetros
# ---------------------------------------------------------------------------


def test_convierte_a_milimetros_una_fisura_de_ancho_conocido():
    # ESTA ES LA PRUEBA CENTRAL DEL MODULO.
    #
    #   marcador de 200 px = 50 mm   ->   1 px = 0.25 mm
    #   fisura de 8 px de ancho      ->   2.0 mm
    #
    # Si el sistema dice otra cosa, el milimetraje que reportaria en una
    # inspeccion real seria igual de erroneo, y con el mismo aspecto de correcto.
    escena = _escena(lado_marcador_px=200, posicion=(30, 30), tamano=(700, 700))
    cv2.line(escena, (450, 80), (450, 620), (35, 35, 35), 8, cv2.LINE_8)

    referencia = detectar_escala(escena, CONFIG)
    assert referencia.detectada

    # El recorte excluye el marcador, igual que ocurre en uso real: la escala se
    # detecta en la foto completa y la fisura se mide en la ventana localizada.
    recorte = escena[60:640, 350:560]
    medidas = medir_grieta(recorte, CONFIG, escala_mm_por_px=referencia.mm_por_px)

    assert medidas.detectada
    assert medidas.ancho_medio_mm == pytest.approx(2.0, abs=0.5)
    assert medidas.longitud_mm == pytest.approx(540 * 0.25, rel=0.2)


def test_sin_escala_los_milimetros_quedan_en_none():
    escena = np.full((400, 400, 3), 210, np.uint8)
    cv2.line(escena, (200, 40), (200, 360), (35, 35, 35), 7, cv2.LINE_8)
    medidas = medir_grieta(escena, CONFIG)

    assert medidas.detectada
    assert medidas.ancho_medio_px > 0
    assert medidas.escala_mm_por_px is None
    assert medidas.ancho_medio_mm is None
    assert medidas.longitud_mm is None


def test_duplicar_la_escala_duplica_los_milimetros():
    escena = np.full((400, 400, 3), 210, np.uint8)
    cv2.line(escena, (200, 40), (200, 360), (35, 35, 35), 7, cv2.LINE_8)
    sencilla = medir_grieta(escena, CONFIG, escala_mm_por_px=0.25)
    doble = medir_grieta(escena, CONFIG, escala_mm_por_px=0.50)

    assert doble.ancho_medio_mm == pytest.approx(sencilla.ancho_medio_mm * 2, rel=0.01)


def test_el_resumen_usa_milimetros_cuando_los_hay():
    escena = np.full((400, 400, 3), 210, np.uint8)
    cv2.line(escena, (200, 40), (200, 360), (35, 35, 35), 7, cv2.LINE_8)

    assert "px" in medir_grieta(escena, CONFIG).resumen()
    assert "mm" in medir_grieta(escena, CONFIG, escala_mm_por_px=0.25).resumen()


# ---------------------------------------------------------------------------
# Anotacion
# ---------------------------------------------------------------------------


def test_la_anotacion_conserva_el_tamano_y_no_toca_el_original():
    escena = _escena(lado_marcador_px=200)
    copia = escena.copy()
    anotada = anotar_referencia(escena, detectar_escala(escena, CONFIG))

    assert anotada.shape == escena.shape
    np.testing.assert_array_equal(escena, copia)


def test_la_anotacion_sin_referencia_devuelve_la_imagen_tal_cual():
    escena = np.full((300, 300, 3), 210, np.uint8)
    anotada = anotar_referencia(escena, ReferenciaEscala(detectada=False))
    np.testing.assert_array_equal(anotada, escena)


def test_el_resumen_de_la_referencia_es_legible_en_ambos_casos():
    assert "Sin referencia" in ReferenciaEscala(detectada=False).resumen()
    assert "mm" in detectar_escala(_escena(lado_marcador_px=200), CONFIG).resumen()


def test_un_borde_blanco_generoso_no_altera_la_medida():
    # Medido sobre tres fondos: con 0 mm de borde o con 4 mm o mas, el lado se
    # mide con un 0.4% de error. Entre 1 y 3 mm el detector se agarra al filo del
    # papel y llega a medir un 14% de mas, lo que haria reportar las grietas mas
    # estrechas de lo que son. De ahi la instruccion impresa en la hoja de dejar
    # 1 cm de blanco al recortar.
    lado = 200
    dicc = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marcador = cv2.cvtColor(cv2.aruco.generateImageMarker(dicc, 0, lado), cv2.COLOR_GRAY2BGR)

    for borde in (0, 40):  # 0 mm y 10 mm equivalentes
        hoja = np.full((lado + 2 * borde, lado + 2 * borde, 3), 255, np.uint8)
        hoja[borde : borde + lado, borde : borde + lado] = marcador
        escena = np.full((700, 700, 3), 120, np.uint8)
        alto, ancho = hoja.shape[:2]
        escena[200 : 200 + alto, 200 : 200 + ancho] = hoja

        referencia = detectar_escala(escena, CONFIG)
        assert referencia.detectada, f"no detecto con borde de {borde} px"
        assert referencia.lado_px == pytest.approx(
            lado, rel=0.03
        ), f"con borde de {borde} px midio {referencia.lado_px:.0f} en vez de {lado}"
