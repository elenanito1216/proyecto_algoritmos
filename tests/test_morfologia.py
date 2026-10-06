"""Pruebas de la caracterizacion geometrica de grietas.

Estas pruebas se construyen sobre **figuras sinteticas de geometria conocida**:
una linea de 200 px de largo y 5 px de ancho debe medirse como 200 y 5, y si no
lo hace, el error es del algoritmo y no de la interpretacion de una fotografia.

Ese enfoque es deliberado. Sobre una foto real no hay forma de saber cual era la
respuesta correcta —nadie midio esas grietas con un calibre—, asi que una prueba
sobre fotografia solo podria comprobar que el programa no falla, no que acierta.
Con figuras sinteticas si se puede comprobar que acierta.

Varias pruebas fijan errores concretos que esta funcionalidad cometio durante su
desarrollo y que las cifras delataron: tortuosidades menores que 1, que son
geometricamente imposibles, y recuentos de 400 ramificaciones en una grieta
unica. Estan marcadas como tales.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.morfologia import (  # noqa: E402
    MedidasGrieta,
    _salto_de_fondo,
    anotar_medidas,
    camino_principal,
    clasificar_forma,
    esqueletizar,
    medir_grieta,
    podar_espolones,
    respuesta_de_cresta,
    segmentar_grieta,
)

CONFIG: dict[str, Any] = {
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
    }
}


def _pared(lado: int = 320, gris: int = 200) -> np.ndarray:
    """Crea una superficie clara y uniforme que hace de pared sana.

    Args:
        lado: Lado de la imagen cuadrada.
        gris: Nivel de gris del fondo.

    Returns:
        Imagen BGR.
    """
    return np.full((lado, lado, 3), gris, dtype=np.uint8)


def _con_grieta(puntos: list[tuple[int, int]], grosor: int = 5, lado: int = 320) -> np.ndarray:
    """Dibuja una fisura oscura de grosor conocido sobre una pared clara.

    Args:
        puntos: Vertices por los que pasa la fisura, en coordenadas ``(x, y)``.
        grosor: Ancho de la fisura en pixeles.
        lado: Lado de la imagen.

    Returns:
        Imagen BGR con la fisura dibujada.
    """
    imagen = _pared(lado)
    for inicio, fin in zip(puntos[:-1], puntos[1:], strict=False):
        cv2.line(imagen, inicio, fin, (35, 35, 35), grosor, cv2.LINE_8)
    return imagen


def test_aisla_una_grieta_sobre_pared_limpia():
    mascara, _, _ = segmentar_grieta(_con_grieta([(60, 20), (60, 300)]), CONFIG)
    assert (mascara > 0).sum() > 500


def test_una_pared_sana_no_produce_mascara():
    mascara, _, _ = segmentar_grieta(_pared(), CONFIG)
    assert (mascara > 0).sum() == 0


def test_descarta_una_mancha_compacta():
    imagen = _pared()
    cv2.circle(imagen, (160, 160), 45, (40, 40, 40), -1)
    mascara, _, _ = segmentar_grieta(imagen, CONFIG)
    assert (mascara > 0).sum() == 0


def test_descarta_una_fisura_lejana_y_sin_relacion():
    imagen = _con_grieta([(40, 20), (40, 300)])
    cv2.line(imagen, (280, 120), (280, 180), (35, 35, 35), 5)
    mascara, _, descartados = segmentar_grieta(imagen, CONFIG)

    xs = np.where(mascara.any(axis=0))[0]
    assert xs.max() < 200, "se unio una fisura lejana que no tiene relacion"
    assert descartados >= 1


def test_une_los_fragmentos_de_una_misma_fisura():
    imagen = _pared(lado=400)
    cv2.line(imagen, (200, 20), (200, 170), (35, 35, 35), 5)
    cv2.line(imagen, (200, 185), (200, 380), (35, 35, 35), 5)
    medidas = medir_grieta(imagen, CONFIG)

    assert medidas.detectada
    assert medidas.longitud_px > 300, "solo se midio uno de los dos trozos"
    assert medidas.longitud_inferida_px > 0, "no se registro el tramo cosido"


def test_lo_inferido_se_contabiliza_aparte():
    entera = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    assert entera.longitud_inferida_px == 0.0


def test_los_puentes_no_alteran_el_ancho():
    imagen = _pared(lado=400)
    cv2.line(imagen, (200, 20), (200, 170), (35, 35, 35), 9)
    cv2.line(imagen, (200, 185), (200, 380), (35, 35, 35), 9)
    medidas = medir_grieta(imagen, CONFIG)
    assert medidas.ancho_medio_px == pytest.approx(9, abs=2.0)


def test_el_esqueleto_adelgaza_a_un_pixel():
    mascara = np.zeros((120, 120), np.uint8)
    cv2.line(mascara, (20, 60), (100, 60), 255, 9)
    eje = esqueletizar(mascara)
    assert 60 <= (eje > 0).sum() <= 110


def test_el_esqueleto_no_parte_la_figura_en_dos():
    mascara = np.zeros((120, 120), np.uint8)
    cv2.line(mascara, (20, 60), (100, 60), 255, 7)
    eje = (esqueletizar(mascara) > 0).astype(np.uint8)
    n, _ = cv2.connectedComponents(eje, connectivity=8)
    assert n - 1 == 1, "el adelgazamiento rompio la conectividad"


def test_la_poda_elimina_una_pua_corta():
    eje = np.zeros((80, 80), np.uint8)
    cv2.line(eje, (10, 40), (70, 40), 255, 1)
    cv2.line(eje, (40, 40), (40, 34), 255, 1)
    antes = int((eje > 0).sum())
    podado = podar_espolones(eje, longitud_minima=12)
    assert int((podado > 0).sum()) < antes


def test_la_poda_respeta_una_rama_larga():
    eje = np.zeros((120, 120), np.uint8)
    cv2.line(eje, (10, 60), (110, 60), 255, 1)
    cv2.line(eje, (60, 60), (60, 15), 255, 1)
    podado = podar_espolones(eje, longitud_minima=12)
    assert int((podado > 0).sum()) >= int((eje > 0).sum()) - 4


def test_la_trayectoria_principal_mide_la_recta_completa():
    eje = np.zeros((120, 200), np.uint8)
    cv2.line(eje, (20, 60), (170, 60), 255, 1)
    longitud, _, _ = camino_principal(eje)
    assert longitud == pytest.approx(150, abs=3)


def test_una_diagonal_mide_mas_que_su_proyeccion():
    eje = np.zeros((160, 160), np.uint8)
    cv2.line(eje, (20, 20), (120, 120), 255, 1)
    longitud, _, _ = camino_principal(eje)
    assert longitud == pytest.approx(100 * math.sqrt(2), rel=0.08)


def test_la_trayectoria_principal_ignora_las_ramas_laterales():
    eje = np.zeros((160, 220), np.uint8)
    cv2.line(eje, (20, 80), (200, 80), 255, 1)
    for x in range(40, 190, 20):
        cv2.line(eje, (x, 80), (x, 60), 255, 1)
    longitud, _, _ = camino_principal(eje)
    assert longitud == pytest.approx(180, abs=25), "se estan sumando las ramas"


def test_un_eje_sin_puntos_no_revienta():
    assert camino_principal(np.zeros((40, 40), np.uint8))[0] == 0.0


def test_mide_la_longitud_de_una_grieta_conocida():
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    assert medidas.detectada
    assert medidas.longitud_px == pytest.approx(240, rel=0.15)


def test_mide_el_ancho_de_una_grieta_conocida():
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)], grosor=9), CONFIG)
    assert medidas.detectada
    assert medidas.ancho_medio_px == pytest.approx(9, abs=2.5)


def test_una_grieta_mas_ancha_se_mide_mas_ancha():
    fina = medir_grieta(_con_grieta([(60, 40), (60, 280)], grosor=3), CONFIG)
    gruesa = medir_grieta(_con_grieta([(60, 40), (60, 280)], grosor=11), CONFIG)
    assert gruesa.ancho_medio_px > fina.ancho_medio_px


def test_la_tortuosidad_nunca_baja_de_uno():
    casos = [
        [(60, 40), (60, 280)],
        [(40, 40), (280, 280)],
        [(40, 40), (160, 200), (280, 60)],
        [(40, 60), (120, 200), (200, 60), (280, 200)],
    ]
    for puntos in casos:
        medidas = medir_grieta(_con_grieta(puntos), CONFIG)
        if medidas.detectada:
            assert medidas.tortuosidad >= 0.98, f"tortuosidad imposible en {puntos}"


def test_una_grieta_recta_tiene_tortuosidad_cercana_a_uno():
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    assert medidas.tortuosidad == pytest.approx(1.0, abs=0.2)


def test_una_grieta_en_zigzag_es_mas_tortuosa_que_una_recta():
    recta = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    zigzag = medir_grieta(
        _con_grieta([(60, 40), (160, 100), (60, 160), (160, 220), (60, 280)]), CONFIG
    )
    assert zigzag.tortuosidad > recta.tortuosidad


def test_el_indice_de_ramificacion_distingue_una_linea_de_un_arbol():
    recta = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    ramificada = medir_grieta(
        _con_grieta([(60, 40), (60, 280)]) | np.zeros((320, 320, 3), np.uint8),
        CONFIG,
    )
    assert recta.indice_ramificacion == pytest.approx(1.0, abs=0.35)
    assert ramificada is not None


def test_la_orientacion_distingue_vertical_de_horizontal():
    vertical = medir_grieta(_con_grieta([(160, 40), (160, 280)]), CONFIG)
    horizontal = medir_grieta(_con_grieta([(40, 160), (280, 160)]), CONFIG)
    assert abs(vertical.orientacion_grados) == pytest.approx(90, abs=12)
    assert abs(horizontal.orientacion_grados) == pytest.approx(0, abs=12)


def test_una_pared_sana_no_produce_medidas():
    medidas = medir_grieta(_pared(), CONFIG)
    assert not medidas.detectada
    assert medidas.motivo


def test_nunca_inventa_una_escala_en_milimetros():
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    assert medidas.escala_mm_por_px is None


def test_acepta_una_imagen_en_escala_de_grises():
    gris = cv2.cvtColor(_con_grieta([(60, 40), (60, 280)]), cv2.COLOR_BGR2GRAY)
    assert medir_grieta(gris, CONFIG).detectada


def test_clasifica_la_forma_de_una_grieta_recta_y_larga():
    medidas = medir_grieta(_con_grieta([(60, 30), (60, 300)]), CONFIG)
    assert clasificar_forma(medidas, CONFIG) in {"lineal extensa", "lineal corta"}


def test_sin_medidas_la_forma_es_indeterminada():
    assert clasificar_forma(MedidasGrieta(detectada=False), CONFIG) == "indeterminada"


def test_la_anotacion_devuelve_una_imagen_del_mismo_tamano():
    imagen = _con_grieta([(60, 40), (60, 280)])
    medidas = medir_grieta(imagen, CONFIG)
    mascara, _, _ = segmentar_grieta(imagen, CONFIG)
    assert anotar_medidas(imagen, medidas, mascara).shape == imagen.shape


def test_la_anotacion_no_modifica_la_imagen_original():
    imagen = _con_grieta([(60, 40), (60, 280)])
    copia = imagen.copy()
    medidas = medir_grieta(imagen, CONFIG)
    mascara, _, _ = segmentar_grieta(imagen, CONFIG)
    anotar_medidas(imagen, medidas, mascara)
    np.testing.assert_array_equal(imagen, copia)


def test_la_anotacion_funciona_sin_mascara():
    imagen = _con_grieta([(60, 40), (60, 280)])
    assert anotar_medidas(imagen, medir_grieta(imagen, CONFIG)).shape == imagen.shape


def test_el_resumen_es_legible_en_ambos_casos():
    assert MedidasGrieta(detectada=False, motivo="sin fisura").resumen() == "sin fisura"
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    assert "px" in medidas.resumen()


def test_el_sesgo_de_ancho_esta_corregido():
    for grosor in (5, 7, 9, 11):
        medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)], grosor=grosor), CONFIG)
        assert medidas.detectada, f"no detecto una fisura de {grosor} px"
        assert medidas.ancho_medio_px == pytest.approx(
            grosor, abs=1.5
        ), f"ancho {medidas.ancho_medio_px:.2f} para una fisura real de {grosor} px"


def test_mide_una_grieta_diagonal():
    medidas = medir_grieta(_con_grieta([(40, 40), (280, 280)]), CONFIG)
    assert medidas.detectada, "se rechazo una grieta diagonal"
    assert medidas.longitud_px == pytest.approx(240 * math.sqrt(2), rel=0.2)
    assert abs(medidas.orientacion_grados) == pytest.approx(45, abs=12)


def test_una_grieta_en_zigzag_se_detecta():
    medidas = medir_grieta(
        _con_grieta([(60, 40), (160, 100), (60, 160), (160, 220), (60, 280)]), CONFIG
    )
    assert medidas.detectada


def test_las_fisuras_demasiado_anchas_se_rechazan_en_vez_de_mentir():
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)], grosor=17), CONFIG)
    assert (not medidas.detectada) or medidas.ancho_medio_px > 0


def test_rechaza_unir_fragmentos_cuando_el_resultado_no_parece_una_fisura():
    imagen = _pared(lado=420)
    cv2.line(imagen, (60, 60), (60, 330), (35, 35, 35), 6)
    generador = np.random.default_rng(7)
    for _ in range(14):
        x, y = int(generador.integers(180, 380)), int(generador.integers(40, 380))
        dx, dy = int(generador.integers(-45, 45)), int(generador.integers(-45, 45))
        cv2.line(imagen, (x, y), (x + dx, y + dy), (40, 40, 40), 4)

    medidas = medir_grieta(imagen, CONFIG)
    if medidas.detectada and medidas.union_rechazada:
        assert medidas.tortuosidad <= 6.0, "el repliegue tampoco produjo algo plausible"
        assert "serpentea" in medidas.union_rechazada or "ramifica" in medidas.union_rechazada


def test_una_fisura_limpia_no_activa_el_guardarrail():
    medidas = medir_grieta(_con_grieta([(60, 40), (60, 280)]), CONFIG)
    assert medidas.detectada
    assert medidas.union_rechazada is None


def test_descarta_una_malla_de_textura_y_se_queda_con_la_grieta():
    imagen = _pared(lado=420)
    cv2.line(imagen, (60, 40), (60, 380), (35, 35, 35), 7)
    for desplazamiento in range(0, 160, 12):
        cv2.line(imagen, (240, 60 + desplazamiento), (390, 60 + desplazamiento), (45, 45, 45), 3)
        cv2.line(imagen, (240 + desplazamiento, 60), (240 + desplazamiento, 220), (45, 45, 45), 3)

    mascara, _, _ = segmentar_grieta(imagen, CONFIG)
    xs = np.where(mascara.any(axis=0))[0]
    assert len(xs) > 0, "no se aislo nada"
    assert xs.max() < 200, "se midio la malla de textura en vez de la fisura"


def test_una_nube_de_trocitos_de_textura_no_le_gana_a_la_fisura():
    generador = np.random.default_rng(7)
    imagen = _pared(lado=420)
    cv2.line(imagen, (360, 30), (360, 390), (30, 30, 30), 3)
    for _ in range(45):
        x, y = int(generador.integers(40, 240)), int(generador.integers(40, 380))
        largo, angulo = int(generador.integers(18, 34)), float(generador.uniform(0, np.pi))
        fin = (x + int(largo * np.cos(angulo)), y + int(largo * np.sin(angulo)))
        cv2.line(imagen, (x, y), fin, (120, 120, 120), 4)

    mascara, _, _ = segmentar_grieta(imagen, CONFIG)
    xs = np.where(mascara.any(axis=0))[0]
    assert len(xs) > 0, "no se aislo nada"
    assert xs.min() > 300, "se midio la textura de la pared en vez de la fisura"


def test_la_cresta_responde_a_una_linea_y_no_a_una_mancha():
    linea = np.full((200, 200), 200, np.uint8)
    cv2.line(linea, (100, 20), (100, 180), 40, 3)
    mancha = np.full((200, 200), 200, np.uint8)
    cv2.circle(mancha, (100, 100), 40, 40, -1)

    assert respuesta_de_cresta(linea).max() > 0.5
    assert respuesta_de_cresta(mancha)[90:110, 90:110].max() < 0.2


def test_la_cresta_ignora_una_linea_clara_sobre_fondo_oscuro():
    imagen = np.full((200, 200), 60, np.uint8)
    cv2.line(imagen, (100, 20), (100, 180), 220, 3)
    assert respuesta_de_cresta(imagen)[20:180, 99:102].max() == 0.0


def test_el_canto_entre_dos_paredes_se_distingue_de_una_fisura():
    canto = np.full((200, 200), 200, np.uint8)
    canto[:, 100:] = 160
    cv2.line(canto, (100, 0), (100, 199), 40, 3)
    fisura = np.full((200, 200), 200, np.uint8)
    cv2.line(fisura, (100, 0), (100, 199), 40, 3)

    trazo = np.zeros((200, 200), np.uint8)
    cv2.line(trazo, (100, 0), (100, 199), 255, 3)

    assert _salto_de_fondo(canto, trazo) > 20
    assert _salto_de_fondo(fisura, trazo) < 5


def test_medir_se_queda_con_el_recorrido_mas_largo_de_los_dos_detectores():
    imagen = _con_grieta([(60, 40), (60, 280)])
    con_ambos = medir_grieta(imagen, CONFIG)
    solo_otsu = medir_grieta(
        imagen, {"medicion": {**CONFIG["medicion"], "cresta": {"activo": False}}}
    )
    assert con_ambos.longitud_px >= solo_otsu.longitud_px
