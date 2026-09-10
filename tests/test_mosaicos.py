"""Pruebas del analisis por mosaicos.

Lo que se comprueba aqui son las propiedades de las que depende que el troceado
sea correcto, no el valor concreto del F1: la cobertura completa de la imagen, el
comportamiento en los bordes, la degradacion controlada cuando la imagen es
pequena y la semantica de la agregacion.

El error mas probable de esta funcionalidad es silencioso: dejar una franja del
muro sin analizar porque la dimension no era multiplo del paso. No lanzaria
ninguna excepcion; simplemente no se veria la grieta que estuviese ahi. Por eso
la cobertura se verifica pixel a pixel y no por conteo de ventanas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models.inferencia import (
    ETIQUETAS_MODELO,
    Predictor,
    ResultadoMosaicos,
    elegir_modelo,
    generar_ventanas,
    predecir_por_mosaicos,
)

CONFIG: dict[str, Any] = {
    "preproceso": {"alto": 160, "ancho": 160, "canales": 3},
    "app": {
        "mosaicos": {
            "lado_px": 480,
            "solape": 0.5,
            "agregacion": "maximo",
            "maximo_mosaicos": 64,
        }
    },
}


class PredictorFalso(Predictor):
    """Predictor determinista que no carga ningun modelo.

    Devuelve el brillo medio de cada imagen del lote, normalizado a ``[0, 1]``.
    Asi una prueba puede construir una imagen con una region clara y saber de
    antemano que ventanas deben dar probabilidad alta.
    """

    def __init__(self) -> None:
        """Inicializa el predictor y el registro de lotes recibidos."""
        super().__init__(ruta=None, formato="falso")  # type: ignore[arg-type]
        self.lotes_vistos: list[int] = []

    def _inferir(self, lote: np.ndarray) -> np.ndarray:
        self.lotes_vistos.append(len(lote))
        return lote.reshape(len(lote), -1).mean(axis=1)

    def descripcion(self) -> dict[str, Any]:
        """Metadatos minimos para cumplir el contrato de Predictor."""
        return {"formato": "falso", "archivo": "-", "tamano_mb": 0.0}


# ---------------------------------------------------------------------------
# generar_ventanas
# ---------------------------------------------------------------------------


def test_cubre_la_imagen_entera_sin_dejar_franjas():
    # 1000 no es multiplo de 240 (el paso con lado 480 y solape 0.5): es
    # justamente el caso en el que una implementacion ingenua dejaria fuera la
    # franja final.
    alto, ancho, lado = 1000, 1000, 480
    ventanas = generar_ventanas(alto, ancho, lado, solape=0.5)

    cubierto = np.zeros((alto, ancho), dtype=bool)
    for x0, y0, x1, y1 in ventanas:
        cubierto[y0:y1, x0:x1] = True

    assert cubierto.all(), "quedaron pixeles sin analizar"


def test_las_ventanas_caben_dentro_de_la_imagen():
    alto, ancho, lado = 733, 1291, 320
    for x0, y0, x1, y1 in generar_ventanas(alto, ancho, lado, solape=0.5):
        assert 0 <= x0 < x1 <= ancho
        assert 0 <= y0 < y1 <= alto


def test_todas_las_ventanas_son_cuadradas_del_lado_pedido():
    for x0, y0, x1, y1 in generar_ventanas(1600, 1200, 480, solape=0.5):
        assert (x1 - x0, y1 - y0) == (480, 480)


def test_imagen_menor_que_el_mosaico_devuelve_una_sola_ventana():
    # Trocear no aportaria nada y solo anadiria coste: se degrada al caso normal.
    assert generar_ventanas(200, 300, 480, solape=0.5) == [(0, 0, 300, 200)]


def test_imagen_del_tamano_exacto_del_mosaico_no_se_trocea():
    assert generar_ventanas(480, 480, 480, solape=0.5) == [(0, 0, 480, 480)]


def test_mas_solape_produce_mas_ventanas():
    poco = generar_ventanas(1600, 1200, 480, solape=0.1)
    mucho = generar_ventanas(1600, 1200, 480, solape=0.7)
    assert len(mucho) > len(poco)


def test_el_solape_se_recorta_al_rango_valido():
    # Un solape de 1.0 daria paso 0 y un bucle infinito. Se recorta a 0.9.
    ventanas = generar_ventanas(1600, 1200, 480, solape=1.0)
    assert 0 < len(ventanas) < 10_000


def test_solape_negativo_se_trata_como_cero():
    sin_solape = generar_ventanas(960, 960, 480, solape=0.0)
    assert generar_ventanas(960, 960, 480, solape=-0.5) == sin_solape


def test_nunca_devuelve_lista_vacia():
    for alto, ancho in [(1, 1), (5, 2000), (2000, 5), (1600, 1200)]:
        assert generar_ventanas(alto, ancho, 480, 0.5), f"vacia para {alto}x{ancho}"


# ---------------------------------------------------------------------------
# predecir_por_mosaicos
# ---------------------------------------------------------------------------


def _imagen_con_region_clara(
    alto: int, ancho: int, region: tuple[int, int, int, int]
) -> np.ndarray:
    """Devuelve una imagen oscura con un rectangulo blanco.

    Args:
        alto: Alto en pixeles.
        ancho: Ancho en pixeles.
        region: ``(x0, y0, x1, y1)`` del rectangulo claro.

    Returns:
        Imagen RGB de enteros sin signo.
    """
    imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)
    x0, y0, x1, y1 = region
    imagen[y0:y1, x0:x1] = 255
    return imagen


def test_el_maximo_detecta_una_region_pequena_que_la_media_diluiria():
    # Este es el argumento de diseno completo, en una prueba: una region clara
    # que ocupa el 2% de la imagen. Con agregacion por media queda enterrada
    # entre las ventanas oscuras; con maximo, se ve.
    imagen = _imagen_con_region_clara(1600, 1200, (100, 100, 300, 300))
    predictor = PredictorFalso()

    por_maximo = predecir_por_mosaicos(predictor, imagen, CONFIG, agregacion="maximo")
    por_media = predecir_por_mosaicos(predictor, imagen, CONFIG, agregacion="media")

    assert por_maximo.probabilidad > por_media.probabilidad * 3


def test_senala_la_ventana_donde_esta_la_evidencia():
    imagen = _imagen_con_region_clara(1600, 1200, (0, 0, 480, 480))
    resultado = predecir_por_mosaicos(PredictorFalso(), imagen, CONFIG)

    x0, y0, x1, y1 = resultado.ventanas[resultado.indice_maximo]
    # El centro de la ventana ganadora debe caer dentro de la region clara.
    assert 0 <= (x0 + x1) // 2 <= 480
    assert 0 <= (y0 + y1) // 2 <= 480


def test_hay_una_probabilidad_por_ventana():
    resultado = predecir_por_mosaicos(PredictorFalso(), np.zeros((1600, 1200, 3), np.uint8), CONFIG)
    assert len(resultado.probabilidades) == len(resultado.ventanas)


def test_todas_las_ventanas_se_infieren_en_un_solo_lote():
    # Agrupar no es una optimizacion opcional: el coste fijo por llamada al
    # modelo domina, y una llamada por ventana multiplicaria el tiempo.
    predictor = PredictorFalso()
    resultado = predecir_por_mosaicos(predictor, np.zeros((1600, 1200, 3), np.uint8), CONFIG)
    assert predictor.lotes_vistos == [len(resultado.ventanas)]


def test_la_cota_de_mosaicos_agranda_el_lado_en_vez_de_descartar_zonas():
    imagen = np.zeros((3000, 3000, 3), dtype=np.uint8)
    resultado = predecir_por_mosaicos(
        PredictorFalso(), imagen, CONFIG, lado=160, maximo_mosaicos=20
    )

    assert len(resultado.ventanas) <= 20
    assert resultado.lado > 160, "el lado deberia haber crecido"

    # Y lo esencial: seguir cubriendo la imagen completa.
    cubierto = np.zeros((3000, 3000), dtype=bool)
    for x0, y0, x1, y1 in resultado.ventanas:
        cubierto[y0:y1, x0:x1] = True
    assert cubierto.all()


def test_imagen_pequena_equivale_a_inferir_sobre_la_imagen_entera():
    imagen = _imagen_con_region_clara(200, 300, (0, 0, 150, 100))
    predictor = PredictorFalso()

    resultado = predecir_por_mosaicos(predictor, imagen, CONFIG)
    directa, _ = predictor.predecir_imagen(imagen, CONFIG)

    assert len(resultado.ventanas) == 1
    assert resultado.probabilidad == pytest.approx(directa, abs=1e-5)


def test_una_imagen_uniforme_da_la_misma_probabilidad_en_todas_las_ventanas():
    imagen = np.full((1600, 1200, 3), 128, dtype=np.uint8)
    resultado = predecir_por_mosaicos(PredictorFalso(), imagen, CONFIG)
    assert resultado.probabilidades.std() < 1e-5


def test_devuelve_un_resultado_del_tipo_esperado():
    resultado = predecir_por_mosaicos(PredictorFalso(), np.zeros((1600, 1200, 3), np.uint8), CONFIG)
    assert isinstance(resultado, ResultadoMosaicos)
    assert resultado.milisegundos >= 0.0
    assert 0.0 <= resultado.probabilidad <= 1.0


def test_los_parametros_del_yaml_se_usan_cuando_no_se_pasan_explicitos():
    config = {**CONFIG, "app": {"mosaicos": {**CONFIG["app"]["mosaicos"], "lado_px": 640}}}
    resultado = predecir_por_mosaicos(PredictorFalso(), np.zeros((1600, 1200, 3), np.uint8), config)
    assert resultado.lado == 640


def test_el_argumento_explicito_manda_sobre_el_yaml():
    resultado = predecir_por_mosaicos(
        PredictorFalso(), np.zeros((1600, 1200, 3), np.uint8), CONFIG, lado=320
    )
    assert resultado.lado == 320


def test_acepta_imagenes_en_escala_de_grises():
    # preparar_imagen replica los canales; el troceado no debe romperse antes.
    imagen = np.zeros((1600, 1200), dtype=np.uint8)
    resultado = predecir_por_mosaicos(PredictorFalso(), imagen, CONFIG)
    assert len(resultado.ventanas) > 1


# ---------------------------------------------------------------------------
# elegir_modelo
#
# La aplicacion ya no tiene selector: elige ella. Eso convierte esta funcion en
# codigo critico, porque un fallo aqui no da error, solo hace que el analisis se
# ejecute silenciosamente con un modelo peor del que corresponde.
# ---------------------------------------------------------------------------

TODOS = {"keras": True, "tflite": True, "ensemble": True}
CONFIG_MODELOS: dict[str, Any] = {"app": {"modelos": {"foto": "keras", "video": "tflite"}}}


def test_la_foto_prioriza_el_acierto_y_el_video_la_latencia():
    assert elegir_modelo(CONFIG_MODELOS, "foto", TODOS) == "keras"
    assert elegir_modelo(CONFIG_MODELOS, "video", TODOS) == "tflite"


def test_sin_ningun_artefacto_devuelve_none_en_vez_de_fallar():
    vacio = dict.fromkeys(TODOS, False)
    assert elegir_modelo(CONFIG_MODELOS, "foto", vacio) is None
    assert elegir_modelo(CONFIG_MODELOS, "video", vacio) is None


def test_el_repliegue_de_foto_prefiere_el_ensemble_antes_que_tflite():
    # Si falta MobileNetV2, para fotos es preferible el ensemble (recall 0.97)
    # a TFLite (recall 0.87): la foto admite el tiempo extra, y §4.1 dice que el
    # error que no hay que cometer es el falso negativo.
    sin_keras = {**TODOS, "keras": False}
    assert elegir_modelo(CONFIG_MODELOS, "foto", sin_keras) == "ensemble"


def test_el_repliegue_de_video_prefiere_keras_antes_que_el_ensemble():
    # Al reves que en foto: sin TFLite, el video ya va lento, y el ensemble solo
    # anadiria un 11% mas de latencia sin resolver nada.
    sin_tflite = {**TODOS, "tflite": False}
    assert elegir_modelo(CONFIG_MODELOS, "video", sin_tflite) == "keras"


def test_solo_queda_un_artefacto_y_los_dos_modos_lo_usan():
    solo_tflite = {"keras": False, "ensemble": False, "tflite": True}
    assert elegir_modelo(CONFIG_MODELOS, "foto", solo_tflite) == "tflite"
    assert elegir_modelo(CONFIG_MODELOS, "video", solo_tflite) == "tflite"


def test_respeta_lo_que_diga_el_yaml_por_encima_del_repliegue():
    config = {"app": {"modelos": {"foto": "ensemble", "video": "keras"}}}
    assert elegir_modelo(config, "foto", TODOS) == "ensemble"
    assert elegir_modelo(config, "video", TODOS) == "keras"


def test_un_yaml_sin_la_seccion_sigue_dando_una_eleccion_sensata():
    assert elegir_modelo({}, "foto", TODOS) == "keras"
    assert elegir_modelo({}, "video", TODOS) == "tflite"


def test_un_modo_desconocido_no_inventa_un_modelo():
    assert elegir_modelo(CONFIG_MODELOS, "termico", TODOS) is None


def test_hay_etiqueta_legible_para_cada_formato():
    for formato in ("keras", "tflite", "ensemble"):
        assert ETIQUETAS_MODELO[formato]
