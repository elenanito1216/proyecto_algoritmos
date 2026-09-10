"""Pruebas de la capa de degradacion de escala.

Esta capa es peligrosa de una forma concreta: si no hiciera nada, el
entrenamiento seguiria funcionando, terminaria sin errores y produciria un
modelo perfectamente valido — solo que sin la propiedad que se buscaba. El fallo
seria invisible y costaria horas de CPU antes de notarse.

Por eso las pruebas no comprueban que la capa "se ejecute", sino que **el tensor
de salida ha perdido detalle de verdad**: misma forma, mismo rango, menos alta
frecuencia. Y que en inferencia no toca nada, que es la otra mitad del contrato.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.data.loader import construir_capa_aumento, construir_capa_degradacion  # noqa: E402


def _lote_con_detalle_fino(n: int = 4, lado: int = 160) -> np.ndarray:
    """Genera un lote con un patron de rayas de un pixel de ancho.

    Un tablero de ajedrez de 1 px es el peor caso para un redimensionado: es la
    frecuencia mas alta que la imagen puede contener, asi que es lo primero que
    desaparece al reducir. Sirve de detector sensible de perdida de detalle.

    Args:
        n: Numero de imagenes del lote.
        lado: Lado de cada imagen.

    Returns:
        Tensor ``(n, lado, lado, 3)`` en ``[0, 1]``.
    """
    fila = np.indices((lado, lado)).sum(axis=0) % 2
    imagen = np.stack([fila] * 3, axis=-1).astype(np.float32)
    return np.repeat(imagen[None, ...], n, axis=0)


def _energia_alta_frecuencia(lote: np.ndarray) -> float:
    """Mide cuanto detalle fino conserva un lote.

    Se usa la media del valor absoluto de la diferencia entre pixeles vecinos:
    en un patron de rayas de 1 px vale casi 1, y cae hacia 0 conforme el
    redimensionado promedia las rayas entre si.

    Args:
        lote: Tensor ``(N, H, W, C)``.

    Returns:
        Energia media de alta frecuencia.
    """
    return float(np.abs(np.diff(np.asarray(lote), axis=2)).mean())


def test_en_inferencia_no_modifica_nada():
    # La otra mitad del contrato: una capa de aumento que actuara en inferencia
    # haria que la misma imagen diera predicciones distintas en cada llamada.
    capa = construir_capa_degradacion(4.0, 1.0)
    lote = _lote_con_detalle_fino()
    salida = capa(tf.constant(lote), training=False)
    np.testing.assert_allclose(np.asarray(salida), lote, rtol=0, atol=0)


def test_en_entrenamiento_destruye_detalle_fino():
    capa = construir_capa_degradacion(4.0, 1.0)
    lote = _lote_con_detalle_fino()
    salida = np.asarray(capa(tf.constant(lote), training=True))

    assert _energia_alta_frecuencia(salida) < _energia_alta_frecuencia(lote) / 2


def test_conserva_la_forma_del_lote():
    # El tensor tiene que seguir encajando en el modelo: la degradacion es un
    # viaje de ida y vuelta, no un cambio de resolucion de entrada.
    capa = construir_capa_degradacion(4.0, 1.0)
    lote = _lote_con_detalle_fino(n=3, lado=160)
    assert tuple(np.asarray(capa(tf.constant(lote), training=True)).shape) == (3, 160, 160, 3)


def test_conserva_el_rango_de_valores():
    capa = construir_capa_degradacion(4.0, 1.0)
    salida = np.asarray(capa(tf.constant(_lote_con_detalle_fino()), training=True))
    assert salida.min() >= 0.0
    assert salida.max() <= 1.0


def test_un_factor_mayor_degrada_mas():
    # El factor concreto se sortea uniformemente en [1, factor_maximo], asi que
    # comparar un unico sorteo de cada capa seria una prueba inestable: un sorteo
    # afortunado de la capa "fuerte" puede caer en 1.1 y quedar por encima de uno
    # de la "suave". Lo que la capa garantiza es una propiedad de la DISTRIBUCION,
    # y por eso se compara la mediana de varias realizaciones.
    lote = tf.constant(_lote_con_detalle_fino())

    def energia_mediana(factor_maximo: float, repeticiones: int = 15) -> float:
        capa = construir_capa_degradacion(factor_maximo, 1.0)
        return float(
            np.median(
                [
                    _energia_alta_frecuencia(np.asarray(capa(lote, training=True)))
                    for _ in range(repeticiones)
                ]
            )
        )

    assert energia_mediana(8.0) < energia_mediana(1.5)


def test_probabilidad_cero_deja_el_lote_intacto():
    capa = construir_capa_degradacion(4.0, 0.0)
    lote = _lote_con_detalle_fino()
    np.testing.assert_allclose(np.asarray(capa(tf.constant(lote), training=True)), lote)


def test_factor_uno_deja_el_lote_intacto():
    # Es el caso que produce `--degradacion-escala 1.0`: desactivar sin tener que
    # tocar el YAML.
    capa = construir_capa_degradacion(1.0, 1.0)
    lote = _lote_con_detalle_fino()
    np.testing.assert_allclose(np.asarray(capa(tf.constant(lote), training=True)), lote)


def test_la_probabilidad_intermedia_deja_pasar_algunos_lotes():
    # Con probabilidad 0.5 debe haber lotes tocados y lotes intactos. Si siempre
    # degradara, el modelo perderia la capacidad de leer grietas nitidas.
    capa = construir_capa_degradacion(4.0, 0.5)
    lote = _lote_con_detalle_fino()
    energias = [
        _energia_alta_frecuencia(np.asarray(capa(tf.constant(lote), training=True)))
        for _ in range(40)
    ]
    intacto = _energia_alta_frecuencia(lote)
    assert any(e >= intacto - 1e-6 for e in energias), "nunca deja pasar un lote intacto"
    assert any(e < intacto / 2 for e in energias), "nunca degrada"


def test_se_integra_en_la_capa_de_aumento_solo_si_esta_activa():
    base = {"preproceso": {"aumento": {"activo": True}}, "proyecto": {"semilla": 42}}
    sin_ella = construir_capa_aumento(base)
    assert all(c.name != "degradacion_escala" for c in sin_ella.layers)

    con_ella = construir_capa_aumento(
        {
            **base,
            "preproceso": {
                "aumento": {
                    "activo": True,
                    "degradacion_escala": {
                        "activo": True,
                        "factor_maximo": 4.0,
                        "probabilidad": 0.5,
                    },
                }
            },
        }
    )
    assert any(c.name == "degradacion_escala" for c in con_ella.layers)


def test_va_antes_que_los_ajustes_fotometricos():
    # El orden codifica una hipotesis fisica: la camara capturo la escena a menor
    # resolucion, y el contraste y el brillo actuan sobre lo que la camara
    # entrego. Invertirlo simularia algo que no ocurre.
    capa = construir_capa_aumento(
        {
            "proyecto": {"semilla": 42},
            "preproceso": {
                "aumento": {
                    "activo": True,
                    "contraste": 0.15,
                    "brillo": 0.15,
                    "degradacion_escala": {
                        "activo": True,
                        "factor_maximo": 4.0,
                        "probabilidad": 0.5,
                    },
                }
            },
        }
    )
    nombres = [c.name for c in capa.layers]
    assert nombres.index("degradacion_escala") < min(
        i for i, n in enumerate(nombres) if "contrast" in n.lower() or "brightness" in n.lower()
    )


def test_la_capa_se_puede_serializar():
    # Sin get_config, guardar el modelo entrenado fallaria al final del
    # entrenamiento, despues de gastar todo el tiempo de CPU.
    capa = construir_capa_degradacion(4.0, 0.5)
    configuracion = capa.get_config()
    assert configuracion["factor_maximo"] == 4.0
    assert configuracion["probabilidad"] == 0.5
