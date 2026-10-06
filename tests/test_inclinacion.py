"""Pruebas del modulo de inclinometria y orientacion de grietas.

Estrategia: se generan imagenes **sinteticas** con lineas de angulo conocido.
Es la unica forma de tener ground truth exacto para verificar el estimador; las
fotografias reales se usan despues en ``scripts/validar_inclinacion.py``, que
mide fidelidad diferencial mediante rotaciones controladas.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pytest

from src.vision.inclinacion import (
    ResultadoInclinacion,
    angulo_desviacion_vertical,
    angulo_respecto_horizontal,
    clasificar_orientacion_grieta,
    estimar_inclinacion,
    mediana_ponderada,
    rotar_imagen,
    validar_con_rotaciones,
)

TOLERANCIA_GRADOS = 1.5


def _imagen_con_linea(
    angulo_grados: float, lado: int = 400, grosor: int = 4, n_lineas: int = 3
) -> np.ndarray:
    """Genera una imagen con lineas rectas a un angulo conocido de la vertical.

    Se dibujan varias lineas paralelas porque el estimador exige un minimo de
    segmentos coherentes antes de declarar la medida fiable, igual que haria con
    los dos bordes de una columna real.

    Args:
        angulo_grados: Desviacion respecto a la vertical, positiva hacia la
            derecha (misma convencion que ``angulo_desviacion_vertical``).
        lado: Tamano de la imagen cuadrada en pixeles.
        grosor: Grosor de las lineas dibujadas.
        n_lineas: Cuantas lineas paralelas dibujar.

    Returns:
        Imagen BGR con las lineas en blanco sobre fondo negro.
    """
    lienzo = np.zeros((lado, lado, 3), dtype=np.uint8)
    radianes = math.radians(angulo_grados)
    media_altura = lado * 0.40

    for indice in range(n_lineas):
        cx = lado // 2 + (indice - n_lineas // 2) * int(lado * 0.15)
        dx = media_altura * math.tan(radianes)
        superior = (int(cx + dx), int(lado / 2 - media_altura))
        inferior = (int(cx - dx), int(lado / 2 + media_altura))
        cv2.line(lienzo, inferior, superior, (255, 255, 255), grosor)

    return lienzo


def test_linea_perfectamente_vertical_da_cero_grados() -> None:
    assert angulo_desviacion_vertical(100, 200, 100, 10) == pytest.approx(0.0, abs=1e-9)


def test_el_signo_indica_hacia_donde_se_inclina() -> None:
    assert angulo_desviacion_vertical(100, 200, 150, 10) > 0
    assert angulo_desviacion_vertical(100, 200, 50, 10) < 0


def test_el_angulo_no_depende_del_orden_de_los_extremos() -> None:
    directo = angulo_desviacion_vertical(100, 200, 130, 20)
    invertido = angulo_desviacion_vertical(130, 20, 100, 200)
    assert directo == pytest.approx(invertido, abs=1e-9)


@pytest.mark.parametrize("esperado", [0.0, 5.0, -5.0, 15.0, -30.0, 45.0])
def test_el_angulo_reproduce_la_geometria_construida(esperado: float) -> None:
    altura = 200.0
    dx = altura * math.tan(math.radians(esperado))
    obtenido = angulo_desviacion_vertical(0.0, altura, dx, 0.0)
    assert obtenido == pytest.approx(esperado, abs=1e-6)


def test_angulo_respecto_horizontal_esta_acotado() -> None:
    assert angulo_respecto_horizontal(0, 0, 100, 0) == pytest.approx(0.0, abs=1e-9)
    assert angulo_respecto_horizontal(0, 100, 0, 0) == pytest.approx(90.0, abs=1e-9)
    assert angulo_respecto_horizontal(0, 100, 100, 0) == pytest.approx(45.0, abs=1e-9)
    assert angulo_respecto_horizontal(0, 0, 100, 100) == pytest.approx(45.0, abs=1e-9)


def test_angulo_respecto_horizontal_con_segmento_degenerado() -> None:
    assert angulo_respecto_horizontal(50, 50, 50, 50) == 0.0


def test_mediana_ponderada_ignora_un_atipico_corto() -> None:
    assert mediana_ponderada([1.0, 2.0, 100.0], [50.0, 50.0, 1.0]) == pytest.approx(2.0)


def test_mediana_ponderada_con_muestra_vacia() -> None:
    assert mediana_ponderada([], []) == 0.0


def test_mediana_ponderada_con_pesos_nulos_cae_a_la_mediana_simple() -> None:
    assert mediana_ponderada([1.0, 3.0, 5.0], [0.0, 0.0, 0.0]) == pytest.approx(3.0)


@pytest.mark.parametrize("angulo", [0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0])
def test_estimacion_recupera_el_angulo_conocido(angulo: float, config: dict[str, Any]) -> None:
    resultado = estimar_inclinacion(_imagen_con_linea(angulo), config, aplicar_guardarrailes=False)
    assert resultado.fiable, f"No se detectaron lineas fiables para {angulo} grados"
    assert resultado.angulo_grados == pytest.approx(angulo, abs=TOLERANCIA_GRADOS)


def test_imagen_sin_bordes_no_lanza_excepcion(config: dict[str, Any]) -> None:
    lienzo = np.full((300, 300, 3), 128, dtype=np.uint8)
    resultado = estimar_inclinacion(lienzo, config)
    assert isinstance(resultado, ResultadoInclinacion)
    assert resultado.fiable is False
    assert resultado.angulo_grados is None
    assert resultado.mensaje


def test_solo_lineas_horizontales_no_produce_angulo(config: dict[str, Any]) -> None:
    lienzo = np.zeros((300, 300, 3), dtype=np.uint8)
    for y in (80, 150, 220):
        cv2.line(lienzo, (20, y), (280, y), (255, 255, 255), 4)
    resultado = estimar_inclinacion(lienzo, config)
    assert resultado.fiable is False
    assert resultado.n_lineas_detectadas > 0
    assert resultado.n_lineas_validas == 0


def test_una_linea_espuria_no_arrastra_la_estimacion(config: dict[str, Any]) -> None:
    lienzo = _imagen_con_linea(3.0)
    cv2.line(lienzo, (30, 380), (170, 30), (255, 255, 255), 3)
    resultado = estimar_inclinacion(lienzo, config)
    assert resultado.fiable
    assert resultado.angulo_grados == pytest.approx(3.0, abs=TOLERANCIA_GRADOS)


def test_la_confianza_crece_con_el_numero_de_lineas(config: dict[str, Any]) -> None:
    pocas = estimar_inclinacion(_imagen_con_linea(4.0, n_lineas=1), config)
    muchas = estimar_inclinacion(_imagen_con_linea(4.0, n_lineas=5), config)
    assert muchas.confianza > pocas.confianza


def test_la_imagen_original_no_se_modifica_al_anotar(config: dict[str, Any]) -> None:
    from src.vision.inclinacion import anotar_imagen

    original = _imagen_con_linea(5.0)
    copia = original.copy()
    anotar_imagen(original, estimar_inclinacion(original, config))
    assert np.array_equal(original, copia)


def test_desaplome_inverosimil_se_declara_no_fiable(config: dict[str, Any]) -> None:
    resultado = estimar_inclinacion(_imagen_con_linea(25.0, lado=600, n_lineas=4), config)
    assert resultado.fiable is False
    assert resultado.angulo_grados is not None, "el angulo se sigue reportando, solo que no fiable"
    assert "inverosimil" in resultado.mensaje.lower()


def test_el_guardarrail_no_toca_las_medidas_plausibles(config: dict[str, Any]) -> None:
    resultado = estimar_inclinacion(_imagen_con_linea(3.0, lado=600, n_lineas=4), config)
    assert resultado.fiable
    assert resultado.angulo_grados == pytest.approx(3.0, abs=TOLERANCIA_GRADOS)


def test_segmentos_de_poca_extension_vertical_se_rechazan(config: dict[str, Any]) -> None:
    lienzo = np.zeros((600, 600, 3), dtype=np.uint8)
    for k in range(4):
        x = 250 + k * 25
        cv2.line(lienzo, (x, 300), (x, 380), (255, 255, 255), 3)
    resultado = estimar_inclinacion(lienzo, config, min_longitud=40)
    assert resultado.fiable is False
    assert "alto de la imagen" in resultado.mensaje


def test_el_limite_de_verosimilitud_es_configurable(config: dict[str, Any]) -> None:
    import copy

    permisiva = copy.deepcopy(config)
    permisiva["inclinacion"]["desaplome_maximo_plausible_grados"] = 40.0

    imagen = _imagen_con_linea(25.0, lado=600, n_lineas=4)
    assert estimar_inclinacion(imagen, config).fiable is False
    assert estimar_inclinacion(imagen, permisiva).fiable is True


def test_los_umbrales_de_guardarrail_estan_en_config(config: dict[str, Any]) -> None:
    cfg = config["inclinacion"]
    assert cfg["desaplome_maximo_plausible_grados"] > 0
    assert 0 < cfg["dispersion_maxima_grados"] < 90
    assert 0 < cfg["min_extension_vertical"] <= 1.0
    assert cfg["desaplome_maximo_plausible_grados"] > config["riesgo"]["desaplome_severo_grados"]


def test_la_validacion_por_rotaciones_no_la_bloquea_el_guardarrail(
    config: dict[str, Any],
) -> None:
    informe = validar_con_rotaciones(
        _imagen_con_linea(0.0, lado=600, n_lineas=4), config, angulos_prueba=(10.0,)
    )
    assert informe["n_casos_fiables"] == informe["n_casos"] == 2
    assert informe["error_medio_grados"] < TOLERANCIA_GRADOS


@pytest.mark.parametrize("alpha", [2.0, 5.0, 10.0])
def test_rotar_desplaza_el_angulo_lo_esperado(alpha: float, config: dict[str, Any]) -> None:
    base = _imagen_con_linea(0.0, lado=600, n_lineas=4)
    estimacion_base = estimar_inclinacion(base, config, aplicar_guardarrailes=False)
    assert estimacion_base.fiable

    rotada = rotar_imagen(base, alpha)
    estimacion_rotada = estimar_inclinacion(rotada, config, aplicar_guardarrailes=False)
    assert estimacion_rotada.fiable

    esperado = estimacion_base.angulo_grados - alpha
    assert estimacion_rotada.angulo_grados == pytest.approx(esperado, abs=TOLERANCIA_GRADOS)


def test_validacion_por_rotaciones_reporta_error_pequeno(config: dict[str, Any]) -> None:
    informe = validar_con_rotaciones(
        _imagen_con_linea(0.0, lado=600, n_lineas=4), config, angulos_prueba=(2.0, 5.0)
    )
    assert informe["n_casos_fiables"] > 0
    assert informe["error_medio_grados"] is not None
    assert informe["error_medio_grados"] < TOLERANCIA_GRADOS


def test_validacion_exige_una_referencia_fiable(config: dict[str, Any]) -> None:
    lienzo = np.full((300, 300, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="referencia"):
        validar_con_rotaciones(lienzo, config)


def _imagen_con_grieta(angulo_desde_horizontal: float, lado: int = 400) -> np.ndarray:
    """Genera una imagen con una grieta recta a un angulo conocido.

    Args:
        angulo_desde_horizontal: Angulo en grados, 0 = horizontal, 90 = vertical.
        lado: Tamano de la imagen cuadrada.

    Returns:
        Imagen BGR con la grieta en blanco sobre fondo negro.
    """
    lienzo = np.zeros((lado, lado, 3), dtype=np.uint8)
    radianes = math.radians(angulo_desde_horizontal)
    largo = lado * 0.40
    cx = cy = lado // 2
    dx = largo * math.cos(radianes)
    dy = largo * math.sin(radianes)
    cv2.line(
        lienzo,
        (int(cx - dx), int(cy + dy)),
        (int(cx + dx), int(cy - dy)),
        (255, 255, 255),
        3,
    )
    return lienzo


@pytest.mark.parametrize(
    ("angulo", "esperada"),
    [
        (0.0, "horizontal"),
        (10.0, "horizontal"),
        (45.0, "diagonal"),
        (90.0, "vertical"),
        (80.0, "vertical"),
    ],
)
def test_clasificacion_de_orientacion(angulo: float, esperada: str, config: dict[str, Any]) -> None:
    resultado = clasificar_orientacion_grieta(_imagen_con_grieta(angulo), config)
    assert resultado.orientacion == esperada


def test_orientacion_sin_bordes_es_indeterminada(config: dict[str, Any]) -> None:
    lienzo = np.full((300, 300, 3), 90, dtype=np.uint8)
    resultado = clasificar_orientacion_grieta(lienzo, config)
    assert resultado.orientacion == "indeterminada"
    assert resultado.angulo_grados is None


def test_el_promedio_axial_no_sufre_el_envolvimiento(config: dict[str, Any]) -> None:
    resultado = clasificar_orientacion_grieta(_imagen_con_grieta(1.0), config)
    assert resultado.orientacion == "horizontal"
    assert resultado.angulo_grados is not None
    assert resultado.angulo_grados < 15.0
