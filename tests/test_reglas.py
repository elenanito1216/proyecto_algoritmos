"""Pruebas del motor de reglas de riesgo.

El motor es una funcion pura, asi que se puede probar exhaustivamente sin
TensorFlow, sin imagenes y sin disco. Eso es precisamente por lo que se diseno
asi: la parte del sistema que emite el juicio es la que mas falta hace poder
verificar.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.risk.reglas import NIVELES, EvaluacionRiesgo, evaluar_riesgo


def _codigos(evaluacion: EvaluacionRiesgo) -> set[str]:
    """Devuelve el conjunto de codigos de regla disparados."""
    return {r.codigo for r in evaluacion.reglas}


@pytest.mark.parametrize("probabilidad", [-0.01, 1.01, 2.0, -5.0])
def test_probabilidad_fuera_de_rango_lanza_error(
    probabilidad: float, config_riesgo_minima: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluar_riesgo(probabilidad, config_riesgo_minima)


@pytest.mark.parametrize("probabilidad", [0.0, 0.5, 1.0])
def test_probabilidad_en_los_extremos_es_valida(
    probabilidad: float, config_riesgo_minima: dict[str, Any]
) -> None:
    evaluacion = evaluar_riesgo(probabilidad, config_riesgo_minima)
    assert evaluacion.nivel in NIVELES


def test_sin_grieta_ni_desaplome_es_riesgo_bajo(config_riesgo_minima: dict[str, Any]) -> None:
    evaluacion = evaluar_riesgo(0.05, config_riesgo_minima, elemento="muro_divisorio")
    assert evaluacion.nivel == "Bajo"
    assert evaluacion.severidad == 0
    assert "R0" in _codigos(evaluacion)


def test_sin_grieta_no_dispara_reglas_de_orientacion(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(
        0.10, config_riesgo_minima, elemento="columna", orientacion_grieta="diagonal"
    )
    assert "R3" not in _codigos(evaluacion)
    assert evaluacion.nivel == "Bajo"


def test_grieta_en_elemento_no_critico_es_riesgo_medio(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(0.70, config_riesgo_minima, elemento="muro_divisorio")
    assert evaluacion.nivel == "Medio"
    assert "R1" in _codigos(evaluacion)


def test_probabilidad_alta_dispara_evidencia_fuerte(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(0.95, config_riesgo_minima, elemento="muro_divisorio")
    assert "R2" in _codigos(evaluacion)


def test_grieta_en_elemento_critico_dispara_regla_de_criticidad(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(0.70, config_riesgo_minima, elemento="columna")
    assert "R7" in _codigos(evaluacion)


@pytest.mark.parametrize(
    ("elemento", "orientacion"),
    [
        ("muro_portante", "diagonal"),
        ("columna", "horizontal"),
        ("columna", "diagonal"),
        ("viga", "diagonal"),
    ],
)
def test_orientacion_grave_eleva_a_riesgo_alto(
    elemento: str, orientacion: str, config_riesgo_minima: dict[str, Any]
) -> None:
    evaluacion = evaluar_riesgo(
        0.80, config_riesgo_minima, elemento=elemento, orientacion_grieta=orientacion
    )
    assert evaluacion.nivel == "Alto"
    assert "R3" in _codigos(evaluacion)


def test_grieta_vertical_en_muro_no_es_grave(config_riesgo_minima: dict[str, Any]) -> None:
    evaluacion = evaluar_riesgo(
        0.70, config_riesgo_minima, elemento="muro_portante", orientacion_grieta="vertical"
    )
    assert "R3" not in _codigos(evaluacion)
    assert "R4" in _codigos(evaluacion)


def test_orientacion_indeterminada_no_dispara_reglas_de_patron(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(
        0.70, config_riesgo_minima, elemento="columna", orientacion_grieta="indeterminada"
    )
    assert "R3" not in _codigos(evaluacion)
    assert "R4" not in _codigos(evaluacion)


def test_orientacion_es_insensible_a_mayusculas(config_riesgo_minima: dict[str, Any]) -> None:
    a = evaluar_riesgo(
        0.80, config_riesgo_minima, elemento="Columna", orientacion_grieta="DIAGONAL"
    )
    b = evaluar_riesgo(
        0.80, config_riesgo_minima, elemento="columna", orientacion_grieta="diagonal"
    )
    assert a.nivel == b.nivel == "Alto"


def test_desaplome_severo_eleva_a_riesgo_alto(config_riesgo_minima: dict[str, Any]) -> None:
    evaluacion = evaluar_riesgo(
        0.05,
        config_riesgo_minima,
        elemento="columna",
        angulo_desaplome=3.5,
        confianza_inclinacion=8,
    )
    assert evaluacion.nivel == "Alto"
    assert "R5" in _codigos(evaluacion)


def test_desaplome_de_atencion_es_riesgo_medio(config_riesgo_minima: dict[str, Any]) -> None:
    evaluacion = evaluar_riesgo(
        0.05,
        config_riesgo_minima,
        elemento="columna",
        angulo_desaplome=1.4,
        confianza_inclinacion=6,
    )
    assert evaluacion.nivel == "Medio"
    assert "R6" in _codigos(evaluacion)


def test_el_signo_del_desaplome_es_irrelevante(config_riesgo_minima: dict[str, Any]) -> None:
    derecha = evaluar_riesgo(
        0.05, config_riesgo_minima, angulo_desaplome=2.5, confianza_inclinacion=5
    )
    izquierda = evaluar_riesgo(
        0.05, config_riesgo_minima, angulo_desaplome=-2.5, confianza_inclinacion=5
    )
    assert derecha.nivel == izquierda.nivel == "Alto"


def test_desaplome_con_baja_confianza_no_altera_el_nivel(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(
        0.05, config_riesgo_minima, angulo_desaplome=5.0, confianza_inclinacion=1
    )
    assert evaluacion.nivel == "Bajo"
    assert "R5" not in _codigos(evaluacion)
    assert evaluacion.advertencias, "Debe avisar de que la medida no es fiable"


def test_desaplome_ausente_genera_advertencia(config_riesgo_minima: dict[str, Any]) -> None:
    evaluacion = evaluar_riesgo(0.90, config_riesgo_minima, angulo_desaplome=None)
    assert any("no se pudo medir" in a.lower() for a in evaluacion.advertencias)


def test_grieta_mas_desaplome_medio_escala_a_alto(config_riesgo_minima: dict[str, Any]) -> None:
    solo_grieta = evaluar_riesgo(0.70, config_riesgo_minima, elemento="muro_divisorio")
    solo_desaplome = evaluar_riesgo(
        0.05,
        config_riesgo_minima,
        elemento="muro_divisorio",
        angulo_desaplome=1.4,
        confianza_inclinacion=6,
    )
    ambas = evaluar_riesgo(
        0.70,
        config_riesgo_minima,
        elemento="muro_divisorio",
        angulo_desaplome=1.4,
        confianza_inclinacion=6,
    )

    assert solo_grieta.nivel == "Medio"
    assert solo_desaplome.nivel == "Medio"
    assert ambas.nivel == "Alto"
    assert "R8" in _codigos(ambas)


def test_el_nivel_es_la_maxima_severidad(config_riesgo_minima: dict[str, Any]) -> None:
    evaluacion = evaluar_riesgo(
        0.95,
        config_riesgo_minima,
        elemento="columna",
        orientacion_grieta="diagonal",
        angulo_desaplome=4.0,
        confianza_inclinacion=10,
    )
    assert evaluacion.severidad == max(r.severidad for r in evaluacion.reglas)
    assert evaluacion.nivel == "Alto"


def test_reglas_criticas_devuelve_solo_las_del_nivel_final(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(
        0.95, config_riesgo_minima, elemento="columna", orientacion_grieta="diagonal"
    )
    criticas = evaluacion.reglas_criticas()
    assert criticas
    assert all(r.severidad == evaluacion.severidad for r in criticas)


def test_toda_regla_trae_justificacion_de_ingenieria(
    config_riesgo_minima: dict[str, Any],
) -> None:
    evaluacion = evaluar_riesgo(
        0.95,
        config_riesgo_minima,
        elemento="columna",
        orientacion_grieta="horizontal",
        angulo_desaplome=2.5,
        confianza_inclinacion=7,
    )
    for regla in evaluacion.reglas:
        assert regla.justificacion.strip(), f"La regla {regla.codigo} no explica su criterio"
        assert regla.detalle.strip(), f"La regla {regla.codigo} no reporta los valores"


def test_la_salida_es_serializable_a_json(config_riesgo_minima: dict[str, Any]) -> None:
    import json

    evaluacion = evaluar_riesgo(0.75, config_riesgo_minima, elemento="viga")
    texto = json.dumps(evaluacion.a_dict(), ensure_ascii=False)
    assert "reglas" in json.loads(texto)


def test_mas_probabilidad_nunca_reduce_el_riesgo(config_riesgo_minima: dict[str, Any]) -> None:
    niveles = [
        evaluar_riesgo(p / 20, config_riesgo_minima, elemento="columna").severidad
        for p in range(21)
    ]
    assert niveles == sorted(niveles)


def test_mas_desaplome_nunca_reduce_el_riesgo(config_riesgo_minima: dict[str, Any]) -> None:
    niveles = [
        evaluar_riesgo(
            0.05, config_riesgo_minima, angulo_desaplome=g / 4, confianza_inclinacion=5
        ).severidad
        for g in range(0, 20)
    ]
    assert niveles == sorted(niveles)


def test_el_config_real_produce_los_tres_niveles(config: dict[str, Any]) -> None:
    bajo = evaluar_riesgo(0.02, config, elemento="muro_divisorio")
    medio = evaluar_riesgo(0.70, config, elemento="muro_divisorio")
    alto = evaluar_riesgo(0.95, config, elemento="columna", orientacion_grieta="horizontal")
    assert (bajo.nivel, medio.nivel, alto.nivel) == ("Bajo", "Medio", "Alto")


def test_los_umbrales_del_config_real_son_coherentes(config: dict[str, Any]) -> None:
    cfg = config["riesgo"]
    assert 0 < cfg["umbral_grieta"] < cfg["umbral_grieta_alta"] <= 1.0
    assert 0 < cfg["desaplome_atencion_grados"] < cfg["desaplome_severo_grados"]
    assert cfg["min_confianza_inclinacion"] >= 1
