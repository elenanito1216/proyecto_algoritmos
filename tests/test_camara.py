"""Pruebas del modulo de captura y estabilizacion temporal.

Se prueba todo lo que no necesita una webcam conectada: el suavizado por
mediana, el recorte central y la medicion de tasa de fotogramas. La apertura del
dispositivo no se prueba aqui porque depende del hardware del equipo, y una
prueba que solo pasa cuando hay camara enchufada no es una prueba.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.camara import (
    MedidorFPS,
    SuavizadorTemporal,
    es_fuente_de_red,
    normalizar_url_celular,
    recortar_centro,
    rectangulo_centro,
)

# --------------------------------------------------------------------------- #
# Suavizado temporal
# --------------------------------------------------------------------------- #


def test_suavizador_vacio_devuelve_none() -> None:
    assert SuavizadorTemporal(5).valor() is None


def test_suavizador_devuelve_la_mediana() -> None:
    s = SuavizadorTemporal(5)
    for v in (0.1, 0.9, 0.5):
        s.agregar(v)
    assert s.valor() == pytest.approx(0.5)


def test_el_suavizador_absorbe_un_fotograma_atipico() -> None:
    # Este es el caso que motiva el modulo: cuatro lecturas coherentes por
    # debajo del umbral y un fotograma movido que dispara a 0.99. La mediana
    # debe ignorarlo; una media lo habria subido por encima de 0.5.
    s = SuavizadorTemporal(5)
    for v in (0.10, 0.12, 0.99, 0.11, 0.13):
        s.agregar(v)
    assert s.valor() == pytest.approx(0.12)
    assert s.valor() < 0.5


def test_la_ventana_descarta_los_valores_viejos() -> None:
    s = SuavizadorTemporal(3)
    for v in (0.9, 0.9, 0.9, 0.1, 0.1, 0.1):
        s.agregar(v)
    assert len(s) == 3
    assert s.valor() == pytest.approx(0.1)


def test_los_valores_ausentes_no_contaminan_la_ventana() -> None:
    # None significa "no se pudo medir", no "vale cero". Tratarlo como cero
    # arrastraria la mediana hacia abajo y falsearia el desaplome.
    s = SuavizadorTemporal(5)
    s.agregar(2.0)
    s.agregar(None)
    s.agregar(2.2)
    assert len(s) == 2
    assert s.valor() == pytest.approx(2.1)


def test_una_ventana_solo_con_ausentes_sigue_siendo_none() -> None:
    s = SuavizadorTemporal(3)
    for _ in range(4):
        s.agregar(None)
    assert s.valor() is None
    assert len(s) == 0


def test_los_valores_no_finitos_se_descartan() -> None:
    s = SuavizadorTemporal(3)
    s.agregar(0.5)
    s.agregar(float("nan"))
    s.agregar(float("inf"))
    assert len(s) == 1
    assert s.valor() == pytest.approx(0.5)


def test_la_estabilidad_distingue_ventana_serena_de_ruidosa() -> None:
    serena = SuavizadorTemporal(5)
    ruidosa = SuavizadorTemporal(5)
    for v in (0.50, 0.51, 0.49, 0.50, 0.51):
        serena.agregar(v)
    for v in (0.10, 0.90, 0.20, 0.80, 0.50):
        ruidosa.agregar(v)
    assert serena.estabilidad() < 0.05
    assert ruidosa.estabilidad() > 0.20


def test_lleno_solo_es_cierto_al_completar_la_ventana() -> None:
    s = SuavizadorTemporal(3)
    s.agregar(1.0)
    assert not s.lleno()
    s.agregar(1.0)
    s.agregar(1.0)
    assert s.lleno()


def test_reiniciar_vacia_la_ventana() -> None:
    s = SuavizadorTemporal(3)
    s.agregar(1.0)
    s.reiniciar()
    assert s.valor() is None
    assert len(s) == 0


@pytest.mark.parametrize("ventana", [0, -1])
def test_una_ventana_no_positiva_es_error(ventana: int) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        SuavizadorTemporal(ventana)


# --------------------------------------------------------------------------- #
# Recorte central
# --------------------------------------------------------------------------- #


def test_el_recorte_es_cuadrado_y_esta_centrado() -> None:
    imagen = np.zeros((480, 640, 3), dtype=np.uint8)
    imagen[220:260, 300:340] = 255  # marca en el centro

    recorte = recortar_centro(imagen, 0.5)
    assert recorte.shape[0] == recorte.shape[1] == 240
    assert recorte.max() == 255, "la marca central debe sobrevivir al recorte"


def test_una_fraccion_de_uno_da_el_cuadrado_central_completo() -> None:
    recorte = recortar_centro(np.zeros((480, 640, 3), dtype=np.uint8), 1.0)
    assert recorte.shape[:2] == (480, 480)


def test_el_recorte_nunca_queda_vacio() -> None:
    recorte = recortar_centro(np.zeros((480, 640, 3), dtype=np.uint8), 0.001)
    assert recorte.shape[0] >= 1 and recorte.shape[1] >= 1


@pytest.mark.parametrize("fraccion", [0.0, -0.5])
def test_una_fraccion_no_positiva_es_error(fraccion: float) -> None:
    with pytest.raises(ValueError, match="> 0"):
        recortar_centro(np.zeros((10, 10, 3), dtype=np.uint8), fraccion)


def test_el_rectangulo_coincide_con_lo_que_recorta() -> None:
    # El rectangulo se dibuja sobre el video para que el usuario vea que region
    # analiza el modelo: si no coincidiera con el recorte real, la interfaz
    # estaria mintiendo.
    imagen = np.zeros((480, 640, 3), dtype=np.uint8)
    for fraccion in (0.3, 0.6, 1.0):
        x0, y0, x1, y1 = rectangulo_centro(imagen, fraccion)
        recorte = recortar_centro(imagen, fraccion)
        assert (y1 - y0, x1 - x0) == recorte.shape[:2]


# --------------------------------------------------------------------------- #
# Medidor de FPS
# --------------------------------------------------------------------------- #


def test_el_medidor_arranca_en_cero() -> None:
    medidor = MedidorFPS()
    assert medidor.fps() == 0.0
    assert medidor.ms_por_fotograma() == 0.0


def test_una_sola_marca_no_permite_estimar() -> None:
    medidor = MedidorFPS()
    medidor.marcar()
    assert medidor.fps() == 0.0


def test_con_varias_marcas_estima_una_tasa_positiva() -> None:
    medidor = MedidorFPS()
    for _ in range(5):
        medidor.marcar()
    assert medidor.fps() > 0.0
    assert medidor.ms_por_fotograma() > 0.0


# --------------------------------------------------------------------------- #
# Fuentes de red (camara del celular)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fuente", [0, 1, 2])
def test_un_indice_no_es_fuente_de_red(fuente: int) -> None:
    assert es_fuente_de_red(fuente) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.40:8080/video",
        "https://192.168.1.40:8080/video",
        "rtsp://192.168.1.40:554/live",
        "RTMP://192.168.1.40/stream",
    ],
)
def test_las_urls_de_stream_se_reconocen(url: str) -> None:
    assert es_fuente_de_red(url) is True


def test_una_cadena_que_no_es_url_no_es_fuente_de_red() -> None:
    # "0" escrito como texto sigue siendo un indice de dispositivo, no una URL.
    assert es_fuente_de_red("0") is False


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        # Lo que muestra en pantalla la app IP Webcam: host y puerto, sin mas.
        ("192.168.1.40:8080", "http://192.168.1.40:8080/video"),
        # Sin puerto: se asume el 8080 de IP Webcam.
        ("192.168.1.40", "http://192.168.1.40:8080/video"),
        # Ya completa: no se toca. Puerto 4747 es el de DroidCam.
        ("http://192.168.1.40:4747/video", "http://192.168.1.40:4747/video"),
        # Con ruta pero sin esquema.
        ("192.168.1.40:8080/video", "http://192.168.1.40:8080/video"),
        # Con esquema y puerto pero sin ruta.
        ("http://192.168.1.40:8080", "http://192.168.1.40:8080/video"),
        # Espacios de un copiar y pegar descuidado.
        ("  192.168.1.40:8080  ", "http://192.168.1.40:8080/video"),
    ],
)
def test_la_url_se_completa_como_la_escriba_el_usuario(entrada: str, esperado: str) -> None:
    assert normalizar_url_celular(entrada) == esperado


def test_una_entrada_vacia_devuelve_cadena_vacia() -> None:
    assert normalizar_url_celular("") == ""
    assert normalizar_url_celular("   ") == ""


def test_el_puerto_por_defecto_es_configurable() -> None:
    assert normalizar_url_celular("192.168.1.40", 4747) == "http://192.168.1.40:4747/video"


def test_normalizar_es_idempotente() -> None:
    # Aplicarlo dos veces no debe anadir otra vez "/video" ni el puerto.
    una = normalizar_url_celular("192.168.1.40:8080")
    assert normalizar_url_celular(una) == una
