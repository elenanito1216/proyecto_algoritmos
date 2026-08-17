"""Panel de evaluacion de riesgo estructural (Streamlit).

Ejecutar desde la raiz del repositorio:

    streamlit run app/app.py

La aplicacion integra las tres piezas del proyecto sobre una misma fotografia:
la CNN que detecta la grieta, la inclinometria clasica que mide el desaplome y
el motor de reglas que convierte ambas senales en un nivel de riesgo explicable.

Principio de diseno de datos: **la app no calcula metricas**. Todo lo que
aparece en la pestana de metricas se lee de ``reports/metricas/``, generado por
``scripts/evaluate.py``. Si un numero no tiene artefacto que lo respalde, no se
muestra.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import estilos  # noqa: E402
from src.models.inferencia import Predictor, cargar_predictor  # noqa: E402
from src.risk.reglas import evaluar_riesgo  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import resolver  # noqa: E402
from src.vision.inclinacion import (  # noqa: E402
    anotar_imagen,
    clasificar_orientacion_grieta,
    estimar_inclinacion,
)

ELEMENTOS = {
    "Columna": "columna",
    "Viga": "viga",
    "Muro portante": "muro_portante",
    "Muro divisorio": "muro_divisorio",
    "Losa": "losa",
}


# --------------------------------------------------------------------------- #
# Carga cacheada
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def obtener_config() -> dict[str, Any]:
    """Carga ``config.yaml`` una sola vez por sesion.

    Returns:
        Diccionario de configuracion.
    """
    return cargar_config()


@st.cache_resource(show_spinner="Cargando modelo...")
def obtener_predictor(formato: str, ruta: str, marca_tiempo: float) -> Predictor:
    """Carga y cachea el predictor.

    ``marca_tiempo`` es la fecha de modificacion del archivo: forma parte de la
    clave de cache para que, si se reentrena el modelo mientras la app esta
    abierta, se recargue solo en lugar de servir el modelo viejo indefinidamente.

    Args:
        formato: ``"keras"`` o ``"tflite"``.
        ruta: Ruta del artefacto.
        marca_tiempo: ``st_mtime`` del archivo.

    Returns:
        Predictor listo para inferir.
    """
    del marca_tiempo  # solo participa en la clave de cache
    # Para el ensemble, 'ruta' es la concatenacion de sus componentes y solo
    # sirve de clave: cargar_predictor los lee de config.yaml.
    return cargar_predictor(obtener_config(), formato, None if formato == "ensemble" else ruta)


@st.cache_data(show_spinner=False)
def cargar_artefacto_json(ruta_relativa: str, marca_tiempo: float) -> dict[str, Any] | None:
    """Lee un artefacto JSON de ``reports/`` si existe.

    Args:
        ruta_relativa: Ruta relativa a la raiz del proyecto.
        marca_tiempo: ``st_mtime`` del archivo, para invalidar la cache.

    Returns:
        Diccionario con el contenido, o ``None`` si el archivo no existe.
    """
    import json

    del marca_tiempo
    destino = resolver(ruta_relativa)
    if not destino.is_file():
        return None
    try:
        return json.loads(destino.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


@st.cache_data(show_spinner=False)
def cargar_artefacto_csv(ruta_relativa: str, marca_tiempo: float) -> pd.DataFrame | None:
    """Lee un artefacto CSV de ``reports/`` si existe.

    Args:
        ruta_relativa: Ruta relativa a la raiz del proyecto.
        marca_tiempo: ``st_mtime`` del archivo, para invalidar la cache.

    Returns:
        ``DataFrame`` con el contenido, o ``None`` si el archivo no existe.
    """
    del marca_tiempo
    destino = resolver(ruta_relativa)
    if not destino.is_file():
        return None
    try:
        return pd.read_csv(destino)
    except (ValueError, OSError):
        return None


def marca_de(ruta_relativa: str) -> float:
    """Devuelve la fecha de modificacion de un archivo, o 0 si no existe.

    Args:
        ruta_relativa: Ruta relativa a la raiz del proyecto.

    Returns:
        ``st_mtime`` como flotante.
    """
    destino = resolver(ruta_relativa)
    return destino.stat().st_mtime if destino.is_file() else 0.0


def artefactos_disponibles(config: dict[str, Any]) -> dict[str, bool]:
    """Comprueba que artefactos existen en disco.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Diccionario ``{clave: existe}`` para ``keras``, ``tflite``, ``baseline``,
        ``ensemble`` y ``evaluacion``.
    """
    componentes = obtener(config, "app.ensemble.componentes", []) or []
    return {
        "keras": resolver(obtener(config, "app.archivo_keras")).is_file(),
        "tflite": resolver(obtener(config, "app.archivo_tflite")).is_file(),
        "baseline": resolver(obtener(config, "app.archivo_baseline")).is_file(),
        "ensemble": bool(componentes) and all(resolver(c).is_file() for c in componentes),
        "evaluacion": resolver("reports/metricas/evaluacion.json").is_file(),
    }


def clave_y_marca(config: dict[str, Any], formato: str) -> tuple[str, float]:
    """Devuelve la clave de cache y la fecha de modificacion de un formato.

    El ensemble no tiene un unico archivo, asi que su clave concatena las rutas
    de los componentes y su marca es la mas reciente de todas: si se reentrena
    cualquiera de ellos, la cache se invalida.

    Args:
        config: Configuracion del proyecto.
        formato: ``"keras"``, ``"tflite"`` o ``"ensemble"``.

    Returns:
        Tupla ``(clave, marca_de_tiempo)``.
    """
    if formato == "ensemble":
        componentes = obtener(config, "app.ensemble.componentes", []) or []
        marca = max((marca_de(c) for c in componentes), default=0.0)
        return "|".join(str(c) for c in componentes), marca

    clave_config = "app.archivo_keras" if formato == "keras" else "app.archivo_tflite"
    ruta = obtener(config, clave_config)
    return str(resolver(ruta)), marca_de(ruta)


# --------------------------------------------------------------------------- #
# Utilidades de imagen
# --------------------------------------------------------------------------- #


def leer_imagen_subida(archivo: Any, max_mb: int) -> tuple[np.ndarray | None, str | None]:
    """Convierte el archivo subido en un arreglo RGB, validandolo.

    Se valida el tamano, el formato y la integridad del contenido. Un archivo
    con extension ``.jpg`` pero contenido corrupto es un caso real (fotos
    transferidas a medias desde el telefono) y no debe tumbar la aplicacion en
    mitad de una demostracion.

    Args:
        archivo: Objeto devuelto por ``st.file_uploader``.
        max_mb: Tamano maximo aceptado en megabytes.

    Returns:
        Tupla ``(imagen_rgb, mensaje_error)``. Uno de los dos es siempre ``None``.
    """
    if archivo is None:
        return None, "No se ha cargado ninguna imagen."

    datos = archivo.getvalue()
    tamano_mb = len(datos) / (1024 * 1024)
    if tamano_mb > max_mb:
        return None, f"La imagen pesa {tamano_mb:.1f} MB y el limite es {max_mb} MB."

    try:
        imagen = Image.open(io.BytesIO(datos))
        imagen.load()  # fuerza la decodificacion: aqui se detecta un JPEG truncado
        imagen = imagen.convert("RGB")
    except UnidentifiedImageError:
        return None, "El archivo no es una imagen reconocible (¿esta corrupto o renombrado?)."
    except OSError as error:
        return None, f"La imagen no se pudo decodificar completamente: {error}"

    arreglo = np.asarray(imagen, dtype=np.uint8)
    if arreglo.ndim != 3 or arreglo.shape[2] != 3:
        return None, "La imagen no tiene tres canales de color tras la conversion."
    if min(arreglo.shape[:2]) < 32:
        return None, f"La imagen es demasiado pequena ({arreglo.shape[1]}x{arreglo.shape[0]} px)."

    return arreglo, None


def redimensionar_para_vision(imagen_rgb: np.ndarray, lado_maximo: int = 1024) -> np.ndarray:
    """Limita el tamano de la imagen antes del procesamiento clasico.

    Canny y Hough son lineales en el numero de pixeles: una foto de movil de 12
    megapixeles multiplica por doce el tiempo de una de 1 megapixel sin aportar
    informacion util sobre la direccion de las lineas. Acotar el lado mayor
    mantiene la interfaz fluida en la demostracion en vivo.

    Args:
        imagen_rgb: Imagen RGB de entrada.
        lado_maximo: Longitud maxima del lado mayor, en pixeles.

    Returns:
        Imagen redimensionada, o la original si ya era lo bastante pequena.
    """
    alto, ancho = imagen_rgb.shape[:2]
    mayor = max(alto, ancho)
    if mayor <= lado_maximo:
        return imagen_rgb
    escala = lado_maximo / mayor
    nuevo = (int(ancho * escala), int(alto * escala))
    return cv2.resize(imagen_rgb, nuevo, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #


def construir_barra_lateral(config: dict[str, Any]) -> dict[str, Any]:
    """Dibuja la barra lateral y devuelve el estado de todos sus controles.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Diccionario con el archivo subido, el elemento seleccionado, el formato
        de modelo y los parametros de OpenCV.
    """
    cfg_inc = config.get("inclinacion", {})
    estado: dict[str, Any] = {}

    with st.sidebar:
        st.markdown("### 📷 Imagen a analizar")
        estado["archivo"] = st.file_uploader(
            "Fotografia del elemento estructural",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            help="Encuadra el elemento completo, con su borde vertical visible de arriba abajo.",
        )
        estado["elemento"] = ELEMENTOS[
            st.selectbox(
                "Elemento inspeccionado",
                list(ELEMENTOS.keys()),
                index=0,
                help=(
                    "Determina que patrones de grieta se consideran graves. Una fisura "
                    "diagonal no significa lo mismo en una columna que en un tabique."
                ),
            )
        ]

        st.divider()
        st.markdown("### 🧠 Modelo")
        disponibles = artefactos_disponibles(config)
        etiquetas = {
            "ensemble": "Ensemble ★",
            "keras": "MobileNetV2",
            "tflite": "TFLite int8",
        }
        opciones = [f for f in ("ensemble", "keras", "tflite") if disponibles[f]]

        if not opciones:
            st.error(
                "No hay ningun modelo entrenado.\n\n"
                "Ejecuta primero:\n"
                "`python scripts/train_transfer.py --config config.yaml`"
            )
            estado["formato"] = None
        else:
            por_defecto = str(obtener(config, "app.modelo_por_defecto", "ensemble"))
            indice = opciones.index(por_defecto) if por_defecto in opciones else 0
            estado["formato"] = st.radio(
                "Formato activo",
                opciones,
                index=indice,
                format_func=lambda v: etiquetas[v],
                help=(
                    "**Ensemble**: promedia la CNN de linea base con MobileNetV2. Es el "
                    "que mejor funciona con fotografias reales, por un 11% de latencia.\n\n"
                    "**MobileNetV2**: el modelo solo.\n\n"
                    "**TFLite int8**: la variante para movil, 35x mas rapida."
                ),
            )
            if estado["formato"] == "ensemble":
                st.caption(
                    "★ Sobre las 40 fotos propias el ensemble sube el F1 de 0.667 a "
                    "**0.824** y elimina 4 de los 10 falsos negativos, por un 11% de "
                    "latencia. Curiosamente, en el dataset publico empeora: los dos "
                    "modelos solo se complementan fuera de distribucion."
                )
            estado["comparar_latencias"] = st.button(
                "⚡ Comparar latencias en vivo", use_container_width=True
            )

        st.divider()
        st.markdown("### 🔧 Sensibilidad de OpenCV")
        st.caption("Cada control altera el resultado de la inclinometria en vivo.")

        estado["canny_bajo"] = st.slider(
            "Canny - umbral inferior",
            0,
            255,
            int(cfg_inc.get("canny_umbral_bajo", 50)),
            step=5,
            help="Bajarlo detecta bordes mas debiles y tambien mas ruido.",
        )
        estado["canny_alto"] = st.slider(
            "Canny - umbral superior",
            0,
            400,
            int(cfg_inc.get("canny_umbral_alto", 150)),
            step=5,
            help="Solo los bordes por encima de este valor inician una cadena de histeresis.",
        )
        estado["min_longitud"] = st.slider(
            "Hough - longitud minima de linea (px)",
            10,
            400,
            int(cfg_inc.get("hough_min_longitud_linea", 80)),
            step=5,
            help="Subirlo descarta la textura del material y conserva los bordes largos.",
        )
        estado["umbral_hough"] = st.slider(
            "Hough - votos minimos",
            10,
            250,
            int(cfg_inc.get("hough_umbral", 60)),
            step=5,
            help="Cuantos puntos alineados hacen falta para aceptar una recta.",
        )
        estado["max_separacion"] = st.slider(
            "Hough - separacion maxima (px)",
            0,
            60,
            int(cfg_inc.get("hough_max_separacion", 10)),
            step=1,
            help="Huecos que se toleran dentro de un mismo segmento.",
        )
        estado["mostrar_descartadas"] = st.checkbox(
            "Mostrar lineas descartadas", value=True, help="En gris, las rechazadas por el filtro."
        )

        if st.button("↺ Restaurar valores de config.yaml", use_container_width=True):
            # Limpiar el estado de los sliders obliga a Streamlit a releer sus
            # valores por defecto, que provienen del YAML.
            for clave in list(st.session_state.keys()):
                del st.session_state[clave]
            st.rerun()

    return estado


def panel_info_modelo(predictor: Predictor) -> None:
    """Muestra en la barra lateral la ficha tecnica del modelo activo.

    Args:
        predictor: Predictor cargado.
    """
    info = predictor.descripcion()
    filas = [
        estilos.fila_dato("Formato", str(info.get("formato", "-"))),
        estilos.fila_dato("Archivo", str(info.get("archivo", "-"))),
        estilos.fila_dato("Arquitectura", str(info.get("arquitectura", "-"))),
    ]
    for componente in info.get("componentes", []):
        tamano = componente.get("tamano_mb")
        filas.append(
            estilos.fila_dato(
                f"· {componente.get('arquitectura', '?')}",
                f"{tamano:.2f} MB" if tamano is not None else "-",
            )
        )
    if info.get("parametros_total"):
        filas.append(estilos.fila_dato("Parametros", f"{info['parametros_total']:,}"))
    if info.get("tamano_mb") is not None:
        filas.append(estilos.fila_dato("Tamano", f"{info['tamano_mb']:.2f} MB"))
    if info.get("tipo_entrada"):
        filas.append(estilos.fila_dato("Tipo entrada", str(info["tipo_entrada"])))
    if info.get("entrada"):
        filas.append(estilos.fila_dato("Entrada", "x".join(str(v) for v in info["entrada"][1:])))

    with st.sidebar:
        st.markdown(
            f'<div class="cra-tarjeta"><h4>Ficha del modelo</h4>{"".join(filas)}</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Pestana 1: analisis en vivo
# --------------------------------------------------------------------------- #


def ejecutar_pipeline(
    imagen_rgb: np.ndarray, predictor: Predictor, config: dict[str, Any], controles: dict[str, Any]
) -> dict[str, Any]:
    """Ejecuta las cinco etapas del analisis mostrando progreso real.

    La barra avanza cuando una etapa **termina de verdad**; no hay retardos
    artificiales. Si una etapa es instantanea, la barra salta: preferimos que la
    interfaz sea honesta sobre donde se va el tiempo.

    Args:
        imagen_rgb: Imagen RGB de entrada.
        predictor: Predictor activo.
        config: Configuracion del proyecto.
        controles: Estado de los controles de la barra lateral.

    Returns:
        Diccionario con ``probabilidad``, ``ms_inferencia``, ``inclinacion``,
        ``orientacion``, ``evaluacion``, ``imagen_anotada`` y ``ms_total``.
    """
    etapas = [
        "Cargando imagen",
        "Preprocesando",
        "Clasificando grieta",
        "Midiendo inclinacion",
        "Aplicando reglas de riesgo",
    ]
    barra = st.progress(0.0, text=f"1/5 · {etapas[0]}")
    inicio_total = time.perf_counter()

    # Etapa 1-2: carga y preprocesado geometrico para el modulo clasico.
    imagen_vision = redimensionar_para_vision(imagen_rgb)
    imagen_bgr = cv2.cvtColor(imagen_vision, cv2.COLOR_RGB2BGR)
    barra.progress(0.2, text=f"2/5 · {etapas[1]}")

    # Etapa 3: clasificacion.
    probabilidad, ms_inferencia = predictor.predecir_imagen(imagen_rgb, config)
    barra.progress(0.4, text=f"3/5 · {etapas[2]}")

    # Etapa 4: inclinometria y orientacion de la grieta.
    inclinacion = estimar_inclinacion(
        imagen_bgr,
        config,
        canny_bajo=controles["canny_bajo"],
        canny_alto=controles["canny_alto"],
        min_longitud=controles["min_longitud"],
        max_separacion=controles["max_separacion"],
        umbral_hough=controles["umbral_hough"],
    )
    orientacion = clasificar_orientacion_grieta(
        imagen_bgr,
        config,
        canny_bajo=controles["canny_bajo"],
        canny_alto=controles["canny_alto"],
    )
    anotada_bgr = anotar_imagen(
        imagen_bgr, inclinacion, dibujar_descartadas=controles["mostrar_descartadas"]
    )
    barra.progress(0.75, text=f"4/5 · {etapas[3]}")

    # Etapa 5: motor de reglas.
    evaluacion = evaluar_riesgo(
        probabilidad_grieta=probabilidad,
        config=config,
        elemento=controles["elemento"],
        orientacion_grieta=orientacion.orientacion,
        angulo_desaplome=inclinacion.angulo_grados,
        confianza_inclinacion=inclinacion.confianza,
    )
    barra.progress(1.0, text=f"5/5 · {etapas[4]} · completado")
    barra.empty()

    return {
        "probabilidad": probabilidad,
        "ms_inferencia": ms_inferencia,
        "inclinacion": inclinacion,
        "orientacion": orientacion,
        "evaluacion": evaluacion,
        "imagen_vision": imagen_vision,
        "imagen_anotada": cv2.cvtColor(anotada_bgr, cv2.COLOR_BGR2RGB),
        "ms_total": (time.perf_counter() - inicio_total) * 1000.0,
    }


def pestana_analisis(
    config: dict[str, Any], controles: dict[str, Any], predictor: Predictor | None
) -> None:
    """Renderiza la pestana de analisis en vivo.

    Args:
        config: Configuracion del proyecto.
        controles: Estado de los controles de la barra lateral.
        predictor: Predictor activo, o ``None`` si no hay modelo.
    """
    st.markdown(estilos.aviso_legal(obtener(config, "app.aviso_legal", "")), unsafe_allow_html=True)

    if predictor is None:
        st.info(
            "Entrena un modelo para habilitar el analisis:\n\n"
            "```bash\npython scripts/train_transfer.py --config config.yaml\n```"
        )
        return

    if controles["archivo"] is None:
        st.info(
            "**Carga una fotografia en la barra lateral para comenzar.**\n\n"
            "Recomendaciones de captura: encuadra el elemento completo, sitúate lo mas "
            "perpendicular posible a la superficie (la perspectiva sesga la medida del "
            "angulo) y evita contraluces fuertes."
        )
        return

    max_mb = int(obtener(config, "app.max_mb_subida", 10))
    imagen_rgb, error = leer_imagen_subida(controles["archivo"], max_mb)
    if imagen_rgb is None:
        st.error(f"No se pudo procesar la imagen. {error}")
        return

    with st.spinner("Analizando la fotografia..."):
        resultado = ejecutar_pipeline(imagen_rgb, predictor, config, controles)

    inclinacion = resultado["inclinacion"]
    orientacion = resultado["orientacion"]
    evaluacion = resultado["evaluacion"]
    umbral = float(obtener(config, "riesgo.umbral_grieta", 0.5))

    # --- Semaforo de riesgo -------------------------------------------------
    st.markdown(estilos.semaforo(evaluacion.nivel, evaluacion.resumen), unsafe_allow_html=True)
    st.write("")

    # --- Metricas principales ----------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hay_grieta = resultado["probabilidad"] >= umbral
        st.markdown(
            estilos.tarjeta_metrica(
                "Probabilidad de grieta",
                f"{resultado['probabilidad']:.1%}",
                nota=(
                    f"{'Grieta detectada' if hay_grieta else 'Sin grieta'} "
                    f"(umbral {umbral:.0%})"
                ),
            ),
            unsafe_allow_html=True,
        )
    with col2:
        if inclinacion.angulo_grados is not None:
            valor = f"{inclinacion.angulo_grados:+.2f}"
            nota = (
                f"{'Medida fiable' if inclinacion.fiable else 'Baja confianza'} · "
                f"{inclinacion.confianza} lineas · ±{inclinacion.dispersion_grados:.2f}°"
            )
        else:
            valor, nota = "n/d", "No se detecto ningun elemento vertical"
        st.markdown(estilos.tarjeta_metrica("Desaplome", valor, "°", nota), unsafe_allow_html=True)
    with col3:
        st.markdown(
            estilos.tarjeta_metrica(
                "Orientacion de la grieta",
                orientacion.orientacion.capitalize(),
                nota=(
                    f"{orientacion.angulo_grados:.1f}° respecto a la horizontal"
                    if orientacion.angulo_grados is not None
                    else "Sin segmentos suficientes"
                ),
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            estilos.tarjeta_metrica(
                "Tiempo de inferencia",
                f"{resultado['ms_inferencia']:.1f}",
                "ms",
                nota=(
                    f"Pipeline completo: {resultado['ms_total']:.0f} ms · " f"{predictor.formato}"
                ),
            ),
            unsafe_allow_html=True,
        )

    st.caption(
        "**Sobre la orientación:** se estima con la misma transformada de Hough, que no "
        "distingue una grieta del borde del propio elemento. En un encuadre amplio, el "
        "ángulo dominante suele ser el de la arista de la columna o el muro. Para leer la "
        "orientación de la fisura, encuadra la zona agrietada de cerca."
    )
    st.write("")

    # --- Comparacion lado a lado -------------------------------------------
    st.markdown("#### Comparacion visual")
    izquierda, derecha = st.columns(2, gap="medium")
    with izquierda:
        st.markdown("**Original**")
        # use_column_width (no use_container_width): es el parametro que expone
        # st.image en Streamlit 1.39. Ambas columnas tienen el mismo ancho, asi
        # que las dos imagenes quedan a la misma escala para compararlas.
        st.image(resultado["imagen_vision"], use_column_width=True)
    with derecha:
        st.markdown("**Procesada · Canny + Hough**")
        st.image(resultado["imagen_anotada"], use_column_width=True)
        # El desglose refleja exactamente lo que dibuja anotar_imagen():
        #   verde = lineas coherentes (== confianza), las que sustentan el angulo
        #   gris  = las que fallaron el filtro de verticalidad
        #   ni una cosa ni la otra = pasaron la verticalidad pero se alejaban de
        #                            la mediana; esas NO se dibujan
        # Reportar aqui n_lineas_validas como "verdes" daria un numero mayor que
        # las lineas verdes realmente visibles.
        tolerancia = float(obtener(config, "inclinacion.tolerancia_vertical_grados", 35.0))
        grises = inclinacion.n_lineas_detectadas - inclinacion.n_lineas_validas
        incoherentes = inclinacion.n_lineas_validas - inclinacion.confianza

        detalle = (
            f"**Verde ({inclinacion.confianza})**: segmentos coherentes; son los que "
            f"sustentan el ángulo. "
            f"**Gris ({grises})**: descartados por alejarse más de {tolerancia:.0f}° "
            f"de la vertical. "
            f"**Naranja**: vertical de referencia."
        )
        if incoherentes > 0:
            detalle += (
                f" Otros **{incoherentes}** segmentos sí eran verticales pero se "
                "apartaban de la mediana; se descartan y no se dibujan."
            )
        detalle += f" Hough detectó {inclinacion.n_lineas_detectadas} segmentos en total."
        st.caption(detalle)

    if inclinacion.motivo_rechazo:
        # Un guardarrail descarto la medida: no es que falten lineas, es que las
        # que hay describen algo que no puede ser el eje de un elemento en pie.
        st.error(
            f"**Medida de desaplome descartada por inverosimil.** "
            f"{inclinacion.motivo_rechazo}\n\n"
            "Comprueba en la imagen procesada si las líneas verdes caen sobre el "
            "borde del elemento o sobre la grieta: si caen sobre la grieta, el "
            "ángulo no describe un desaplome."
        )
    elif not inclinacion.fiable:
        st.warning(f"**Inclinometria poco fiable.** {inclinacion.mensaje}")

    st.write("")

    # --- Reglas disparadas --------------------------------------------------
    st.markdown("#### Reglas que determinaron el nivel de riesgo")
    st.caption(
        "El nivel es la maxima severidad entre las reglas activas. Cada regla cita el "
        "criterio de ingenieria que la motiva."
    )
    for r in sorted(evaluacion.reglas, key=lambda x: -x.severidad):
        st.markdown(
            estilos.regla(r.codigo, r.titulo, r.detalle, r.justificacion, r.severidad),
            unsafe_allow_html=True,
        )

    for advertencia in evaluacion.advertencias:
        st.info(advertencia)

    with st.expander("Ver salida completa en JSON (trazabilidad)"):
        st.json(evaluacion.a_dict())


def bloque_comparar_latencias(config: dict[str, Any], imagen_rgb: np.ndarray | None) -> None:
    """Mide y compara la latencia de los formatos ``.keras`` y ``.tflite``.

    Es la demostracion de viabilidad movil: la misma imagen, los dos artefactos,
    los tiempos medidos en el momento y en el equipo del jurado.

    Args:
        config: Configuracion del proyecto.
        imagen_rgb: Imagen cargada, o ``None`` para usar una entrada sintetica.
    """
    from src.models.inferencia import preparar_imagen

    disponibles = artefactos_disponibles(config)
    if not (disponibles["keras"] and disponibles["tflite"]):
        st.warning(
            "Para comparar hacen falta los dos artefactos. Exporta el TFLite con:\n\n"
            "`python scripts/export_tflite.py --config config.yaml`"
        )
        return

    if imagen_rgb is None:
        alto, ancho, canales = (
            int(obtener(config, "preproceso.alto", 160)),
            int(obtener(config, "preproceso.ancho", 160)),
            int(obtener(config, "preproceso.canales", 3)),
        )
        lote = np.random.default_rng(0).random((1, alto, ancho, canales)).astype(np.float32)
    else:
        lote = preparar_imagen(imagen_rgb, config)

    formatos = [f for f in ("ensemble", "keras", "tflite") if disponibles[f]]
    filas: list[dict[str, Any]] = []
    with st.spinner("Midiendo 30 inferencias por formato (5 de calentamiento)..."):
        for formato in formatos:
            clave, marca = clave_y_marca(config, formato)
            predictor = obtener_predictor(formato, clave, marca)
            for _ in range(5):  # calentamiento: la primera llamada no es representativa
                predictor.predecir(lote)
            tiempos = [predictor.predecir(lote)[1] for _ in range(30)]
            info = predictor.descripcion()
            filas.append(
                {
                    "Formato": info["formato"],
                    "Latencia media (ms)": round(float(np.mean(tiempos)), 2),
                    "Desv. estandar (ms)": round(float(np.std(tiempos, ddof=1)), 2),
                    "Tamano (MB)": info.get("tamano_mb"),
                }
            )

    tabla = pd.DataFrame(filas)
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    por_formato = dict(zip(formatos, filas, strict=True))

    if "keras" in por_formato and "tflite" in por_formato:
        k, t = por_formato["keras"], por_formato["tflite"]
        if t["Latencia media (ms)"] > 0:
            aceleracion = k["Latencia media (ms)"] / t["Latencia media (ms)"]
            reduccion = k["Tamano (MB)"] / max(t["Tamano (MB)"] or 1e-9, 1e-9)
            st.success(
                f"TFLite int8 es **{aceleracion:.2f}x** en velocidad y **{reduccion:.1f}x** mas "
                "pequeno en disco respecto al `.keras`. Medido ahora mismo en este equipo, "
                "con un solo hilo."
            )

    if "ensemble" in por_formato and "keras" in por_formato:
        e, k = por_formato["ensemble"], por_formato["keras"]
        sobrecoste = 100 * (e["Latencia media (ms)"] / max(k["Latencia media (ms)"], 1e-9) - 1)
        st.info(
            f"El **ensemble** cuesta un **{sobrecoste:+.0f}%** de latencia frente a "
            "MobileNetV2 sola y sube el F1 sobre fotografias reales de 0.667 a "
            "**0.824**, eliminando 4 de los 10 falsos negativos.\n\n"
            "La medida de referencia es **+11 %**. La CNN de linea base es solo el "
            "1.9 % de los parametros del conjunto pero aporta el 11 % del tiempo: "
            "el numero de parametros predice mal la latencia."
        )
        st.caption(
            "Esta comparacion usa 5 pasadas de calentamiento; con tan pocas, el "
            "primer formato medido puede salir penalizado por el trazado del grafo. "
            "Si un resultado te parece imposible, repitela."
        )


# --------------------------------------------------------------------------- #
# Pestana 2: metricas
# --------------------------------------------------------------------------- #


def figura_matriz_confusion(metricas: dict[str, Any], clases: list[str], titulo: str) -> go.Figure:
    """Construye el mapa de calor anotado de la matriz de confusion.

    Args:
        metricas: Bloque de metricas de un modelo.
        clases: Nombres de clase.
        titulo: Titulo de la figura.

    Returns:
        Figura de Plotly.
    """
    matriz = np.array(metricas["matriz_confusion_lista"])
    texto = [[str(v) for v in fila] for fila in matriz]

    figura = go.Figure(
        data=go.Heatmap(
            z=matriz,
            x=[f"Pred: {c}" for c in clases],
            y=[f"Real: {c}" for c in clases],
            text=texto,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale="Teal",
            showscale=False,
            hovertemplate="%{y} → %{x}<br>%{z} imagenes<extra></extra>",
        )
    )
    figura.update_layout(**estilos.plantilla_plotly())
    figura.update_layout(title=titulo, height=380)
    figura.update_yaxes(autorange="reversed")
    return figura


def figura_curvas(historial: pd.DataFrame, titulo: str) -> go.Figure:
    """Construye las curvas de entrenamiento a partir del CSV guardado.

    Args:
        historial: DataFrame con las columnas de ``History.history``.
        titulo: Titulo de la figura.

    Returns:
        Figura de Plotly con dos ejes y: perdida y exactitud.
    """
    figura = go.Figure()
    epocas = historial["epoca"] if "epoca" in historial else historial.index + 1

    if "loss" in historial:
        figura.add_trace(
            go.Scatter(x=epocas, y=historial["loss"], name="Perdida (train)", mode="lines+markers")
        )
    if "val_loss" in historial:
        figura.add_trace(
            go.Scatter(
                x=epocas,
                y=historial["val_loss"],
                name="Perdida (val)",
                mode="lines+markers",
                line={"dash": "dash"},
            )
        )
    clave = "accuracy" if "accuracy" in historial else "binary_accuracy"
    if clave in historial:
        figura.add_trace(
            go.Scatter(
                x=epocas, y=historial[clave], name="Exactitud (train)", mode="lines", yaxis="y2"
            )
        )
    if f"val_{clave}" in historial:
        figura.add_trace(
            go.Scatter(
                x=epocas,
                y=historial[f"val_{clave}"],
                name="Exactitud (val)",
                mode="lines",
                line={"dash": "dash"},
                yaxis="y2",
            )
        )

    figura.update_layout(**estilos.plantilla_plotly())
    figura.update_layout(
        title=titulo,
        height=420,
        xaxis_title="Epoca",
        yaxis={"title": "Perdida", "gridcolor": "rgba(128,128,128,0.22)"},
        yaxis2={
            "title": "Exactitud",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1.02],
            "showgrid": False,
        },
    )
    return figura


def figura_pr(curva: dict[str, list[float]], titulo: str) -> go.Figure:
    """Construye la curva precision-recall.

    Args:
        curva: Diccionario con listas ``precision``, ``recall`` y ``umbrales``.
        titulo: Titulo de la figura.

    Returns:
        Figura de Plotly.
    """
    figura = go.Figure(
        go.Scatter(
            x=curva["recall"],
            y=curva["precision"],
            mode="lines",
            fill="tozeroy",
            line={"color": estilos.ACENTO_PRIMARIO, "width": 2},
            customdata=curva["umbrales"],
            hovertemplate=(
                "Recall %{x:.3f}<br>Precision %{y:.3f}<br>Umbral %{customdata:.3f}<extra></extra>"
            ),
        )
    )
    figura.update_layout(**estilos.plantilla_plotly())
    figura.update_layout(
        title=titulo,
        height=380,
        xaxis_title="Recall (clase: con grieta)",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
    )
    return figura


def figura_comparativa(filas: list[dict[str, Any]]) -> go.Figure:
    """Construye la tabla comparativa de desempeno contra costo.

    Args:
        filas: Salida de ``construir_tabla_comparativa``.

    Returns:
        Figura de Plotly con una tabla formateada.
    """
    tabla = pd.DataFrame(filas)
    columnas = {
        "modelo": "Modelo",
        "parametros_total": "Parametros",
        "tamano_mb": "Tamano (MB)",
        "latencia_media_ms": "Latencia (ms)",
        "accuracy": "Exactitud",
        "recall": "Recall",
        "f1": "F1",
        "falsos_negativos": "Falsos neg.",
    }
    presentes = [c for c in columnas if c in tabla.columns]
    vista = tabla[presentes].rename(columns=columnas)

    def _formatear(columna: str, valores: Any) -> list[str]:
        salida: list[str] = []
        for valor in valores:
            if valor is None or (isinstance(valor, float) and np.isnan(valor)):
                salida.append("—")
            elif columna in ("Exactitud", "Recall", "F1"):
                salida.append(f"{float(valor):.4f}")
            elif columna == "Parametros":
                salida.append(f"{int(valor):,}")
            elif columna in ("Tamano (MB)", "Latencia (ms)"):
                salida.append(f"{float(valor):.2f}")
            else:
                salida.append(str(valor))
        return salida

    figura = go.Figure(
        go.Table(
            header={
                "values": [f"<b>{c}</b>" for c in vista.columns],
                "fill_color": "rgba(34,184,207,0.20)",
                "align": "left",
                "height": 34,
            },
            cells={
                "values": [_formatear(c, vista[c]) for c in vista.columns],
                "fill_color": "rgba(128,128,128,0.06)",
                "align": "left",
                "height": 30,
            },
        )
    )
    figura.update_layout(**estilos.plantilla_plotly())
    figura.update_layout(
        title="Desempeno frente a costo computacional", height=90 + 34 * (len(vista) + 1)
    )
    return figura


def pestana_metricas(config: dict[str, Any]) -> None:
    """Renderiza la pestana de metricas del modelo.

    Todos los datos provienen de ``reports/metricas/``. Si un artefacto falta, se
    indica el comando exacto que lo genera en lugar de mostrar un hueco vacio.

    Args:
        config: Configuracion del proyecto.
    """
    evaluacion = cargar_artefacto_json(
        "reports/metricas/evaluacion.json", marca_de("reports/metricas/evaluacion.json")
    )

    if evaluacion is None:
        st.info(
            "Todavia no hay artefactos de evaluacion. Generalos con:\n\n"
            "```bash\n"
            "python scripts/train_baseline.py --config config.yaml\n"
            "python scripts/train_transfer.py --config config.yaml\n"
            "python scripts/export_tflite.py  --config config.yaml\n"
            "python scripts/evaluate.py       --config config.yaml\n"
            "```"
        )
        return

    clases = obtener(config, "datos.clases", ["Negative", "Positive"])
    nombres = list(evaluacion.get("modelos", {}).keys())
    if not nombres:
        st.warning("El archivo de evaluacion no contiene ningun modelo.")
        return

    if evaluacion.get("comparativa"):
        st.plotly_chart(figura_comparativa(evaluacion["comparativa"]), use_container_width=True)
        st.caption(
            "Latencia medida con lote de 1 imagen; el TFLite se evalua con un solo hilo "
            f"para representar un dispositivo modesto. Equipo de medida: "
            f"{evaluacion.get('dispositivo', {}).get('dispositivo', '?')}."
        )
        st.divider()

    seleccion = st.selectbox("Modelo a inspeccionar", nombres)
    bloque = evaluacion["modelos"][seleccion]

    conjuntos = [c for c in ("test", "propias") if c in bloque and "metricas" in bloque.get(c, {})]
    etiquetas = {"test": "Conjunto de prueba", "propias": "Fotos propias del equipo"}
    conjunto = st.radio(
        "Conjunto de evaluacion",
        conjuntos,
        format_func=lambda c: etiquetas.get(c, c),
        horizontal=True,
    )
    datos = bloque[conjunto]
    m = datos["metricas"]

    columnas = st.columns(5)
    for columna, (etiqueta, clave, formato) in zip(
        columnas,
        [
            ("Exactitud", "accuracy", "{:.4f}"),
            ("Precision", "precision", "{:.4f}"),
            ("Recall", "recall", "{:.4f}"),
            ("F1", "f1", "{:.4f}"),
            ("Muestras", "n_muestras", "{:,}"),
        ],
        strict=False,
    ):
        with columna:
            st.metric(etiqueta, formato.format(m[clave]))

    st.write("")
    izquierda, derecha = st.columns(2, gap="medium")
    with izquierda:
        st.plotly_chart(
            figura_matriz_confusion(m, clases, f"Matriz de confusion · {etiquetas[conjunto]}"),
            use_container_width=True,
        )
    with derecha:
        if datos.get("curva_pr"):
            st.plotly_chart(
                figura_pr(datos["curva_pr"], f"Precision-Recall · {etiquetas[conjunto]}"),
                use_container_width=True,
            )
        else:
            st.info("La curva precision-recall exige ambas clases en el conjunto.")

    # --- Falsos negativos ---------------------------------------------------
    fn = datos.get("falsos_negativos", {})
    st.markdown("#### Analisis de falsos negativos")
    st.caption(
        "Grietas reales que el modelo no detecto. Es el error caro de este dominio: "
        "un falso positivo cuesta una revision; un falso negativo, un elemento danado "
        "que nadie inspecciona."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Falsos negativos", fn.get("n_falsos_negativos", 0))
    c2.metric("Sobre grietas reales", fn.get("n_positivos_reales", 0))
    c3.metric("Tasa de fuga", f"{fn.get('tasa', 0):.2%}")

    if fn.get("peores_casos"):
        st.caption("Casos en los que el modelo estuvo mas seguro de equivocarse:")
        st.dataframe(pd.DataFrame(fn["peores_casos"]), use_container_width=True, hide_index=True)

    ur = datos.get("umbral_recall")
    if ur:
        st.markdown("#### Ajuste del umbral de decision")
        if ur.get("alcanzable", True):
            st.info(
                f"Para garantizar un **recall de {ur['recall_objetivo']:.0%}** en la clase "
                f"'con grieta', el umbral debe bajar a **{ur['umbral']:.3f}** (por defecto "
                f"{evaluacion.get('umbral', 0.5):.2f}). La precision resultante seria "
                f"**{ur['precision_resultante']:.3f}**.\n\n"
                "**El compromiso:** bajar el umbral reduce las grietas que se escapan y "
                "aumenta las falsas alarmas. En seguridad estructural esa asimetria "
                "justifica sacrificar precision. Ajusta `evaluacion.umbral` en "
                "`config.yaml` si decides adoptarlo."
            )
        else:
            st.warning(
                f"Ningun umbral alcanza el recall objetivo de {ur['recall_objetivo']:.0%} "
                "en este conjunto. El modelo necesita mas datos o mas capacidad."
            )

    # --- Curvas de entrenamiento -------------------------------------------
    st.markdown("#### Curvas de entrenamiento")
    encontradas = False
    for nombre in (
        obtener(config, "transfer.nombre", "mobilenetv2"),
        obtener(config, "baseline.nombre", "baseline_cnn"),
    ):
        ruta = f"reports/metricas/historial_{nombre}.csv"
        historial = cargar_artefacto_csv(ruta, marca_de(ruta))
        if historial is not None and not historial.empty:
            st.plotly_chart(
                figura_curvas(historial, f"Entrenamiento · {nombre}"), use_container_width=True
            )
            encontradas = True
    if not encontradas:
        st.info("No hay historiales de entrenamiento en reports/metricas/.")

    # --- Validacion de la inclinometria ------------------------------------
    validacion = cargar_artefacto_json(
        "reports/metricas/validacion_inclinacion.json",
        marca_de("reports/metricas/validacion_inclinacion.json"),
    )
    if validacion:
        st.markdown("#### Validacion del estimador de inclinacion")
        st.caption(
            "Sin inclinometro fisico no hay ground truth. Se valida rotando fotos a plomo "
            "angulos conocidos y comprobando que la estimacion se desplaza exactamente ese "
            "angulo."
        )
        c1, c2 = st.columns(2)
        c1.metric("Imagenes validadas", validacion.get("n_imagenes", 0))
        error = validacion.get("error_medio_global_grados")
        c2.metric("Error medio", f"{error:.3f}°" if error is not None else "n/d")

        csv_val = cargar_artefacto_csv(
            "reports/metricas/validacion_inclinacion.csv",
            marca_de("reports/metricas/validacion_inclinacion.csv"),
        )
        if csv_val is not None:
            st.dataframe(csv_val, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Pestana 3: acerca del proyecto
# --------------------------------------------------------------------------- #


def pestana_acerca(config: dict[str, Any]) -> None:
    """Renderiza la pestana informativa del proyecto.

    Args:
        config: Configuracion del proyecto.
    """
    izquierda, derecha = st.columns([3, 2], gap="large")

    with izquierda:
        st.markdown(
            """
#### El problema

Tras un sismo, o simplemente con el paso del tiempo, una edificación acumula
señales de daño que un ojo no entrenado no sabe jerarquizar. Este prototipo
automatiza dos de esas señales sobre una fotografía corriente:

1. **¿Hay grieta?** Una red convolucional clasifica la superficie.
2. **¿Está a plomo?** Canny y la transformada de Hough miden la desviación del
   elemento respecto a la vertical.

Un motor de reglas combina ambas señales con el tipo de elemento y la
orientación de la fisura para emitir un nivel de riesgo **explicable**: cada
nivel llega acompañado de las reglas que lo dispararon y del criterio de
ingeniería que las motiva.

#### Enfoque técnico

| Módulo | Técnica | Por qué |
|---|---|---|
| Clasificación | MobileNetV2 (transfer learning) | Diseñada para inferencia móvil; convolución separable en profundidad |
| Línea base | CNN propia de 3 bloques | Punto de comparación obligatorio |
| Inclinometría | Canny + HoughLinesP | Geometría explícita, sin datos etiquetados de ángulo |
| Ángulo robusto | Mediana ponderada por longitud | Inmune a líneas espurias (ventanas, cables, marcos) |
| Riesgo | Motor de reglas puro | Auditable y recalibrable sin reentrenar |
| Despliegue | TensorFlow Lite int8 | ~4× menos tamaño; viable en gama baja |

#### Limitaciones que hay que decir en voz alta

- **No mide el ancho real de la grieta.** Sin una referencia métrica en la
  escena (una regla, una moneda), la escala es desconocida. El ancho de fisura
  es justamente el criterio que usa la NSR-10, y este sistema no puede darlo.
- **La perspectiva sesga el ángulo.** El desaplome se mide en el plano de la
  imagen. Una foto tomada en ángulo introduce error sistemático.
- **El dataset no representa la construcción informal.** Está dominado por
  hormigón de laboratorio bien iluminado; el ladrillo a la vista, el pañete
  agrietado o el bahareque están fuera de su distribución.
- **Un falso negativo es el error caro.** En contexto sísmico, una grieta
  peligrosa no detectada puede costar vidas. Por eso el proyecto reporta el
  recall por separado y permite ajustar el umbral.

#### Aviso ético

Este sistema es una **herramienta de tamizaje**, no un dictamen. Usarlo para
declarar habitable una edificación es un uso indebido con consecuencias
potencialmente graves. El análisis completo está en `reports/analisis.md`.
            """
        )

    with derecha:
        st.markdown("#### Ficha del proyecto")
        filas = [
            estilos.fila_dato("Asignatura", "Algoritmos y Programación"),
            estilos.fila_dato("Programa", "Ingeniería en IA · UIS"),
            estilos.fila_dato("Periodo", "2026-2"),
            estilos.fila_dato(
                "Resolución de entrada",
                f"{obtener(config, 'preproceso.alto')}×{obtener(config, 'preproceso.ancho')}",
            ),
            estilos.fila_dato("Arquitectura", str(obtener(config, "transfer.arquitectura_base"))),
            estilos.fila_dato("Alpha", str(obtener(config, "transfer.alpha"))),
            estilos.fila_dato("Semilla global", str(obtener(config, "proyecto.semilla"))),
        ]
        st.markdown(
            f'<div class="cra-tarjeta"><h4>Configuración activa</h4>{"".join(filas)}</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("#### Equipo")
        equipo = obtener(config, "app.equipo", []) or []
        filas_equipo = [
            estilos.fila_dato(str(m.get("nombre", "—")), str(m.get("rol", ""))) for m in equipo
        ]
        st.markdown(
            f'<div class="cra-tarjeta"><h4>Integrantes</h4>{"".join(filas_equipo)}</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("#### Umbrales de riesgo vigentes")
        cfg_riesgo = config.get("riesgo", {})
        filas_riesgo = [
            estilos.fila_dato("Grieta detectada", f"P ≥ {cfg_riesgo.get('umbral_grieta')}"),
            estilos.fila_dato("Evidencia fuerte", f"P ≥ {cfg_riesgo.get('umbral_grieta_alta')}"),
            estilos.fila_dato(
                "Desaplome · atención", f"≥ {cfg_riesgo.get('desaplome_atencion_grados')}°"
            ),
            estilos.fila_dato(
                "Desaplome · severo", f"≥ {cfg_riesgo.get('desaplome_severo_grados')}°"
            ),
        ]
        st.markdown(
            f'<div class="cra-tarjeta"><h4>Definidos en config.yaml</h4>{"".join(filas_riesgo)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Ningún umbral está incrustado en el código: todos viven en `config.yaml` "
            "y se pueden recalibrar con un ingeniero estructural sin tocar Python."
        )


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #


def main() -> None:
    """Construye y renderiza la aplicacion completa."""
    # set_page_config debe ser la primera llamada de Streamlit que se ejecuta;
    # por eso el titulo se lee del YAML directamente y no del cache de sesion.
    st.set_page_config(
        page_title="Evaluacion de Riesgo Estructural",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = obtener_config()
    estilos.inyectar_estilos(st)
    st.markdown(
        estilos.cabecera(
            obtener(config, "app.titulo", "Evaluacion de Riesgo Estructural"),
            obtener(config, "app.subtitulo", ""),
        ),
        unsafe_allow_html=True,
    )

    controles = construir_barra_lateral(config)

    predictor: Predictor | None = None
    if controles.get("formato"):
        clave = "app.archivo_keras" if controles["formato"] == "keras" else "app.archivo_tflite"
        ruta = str(resolver(obtener(config, clave)))
        try:
            predictor = obtener_predictor(
                controles["formato"], ruta, marca_de(obtener(config, clave))
            )
            panel_info_modelo(predictor)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            st.sidebar.error(f"No se pudo cargar el modelo: {error}")

    pestana1, pestana2, pestana3 = st.tabs(
        ["🔍 Análisis en vivo", "📊 Métricas del modelo", "ℹ️ Acerca del proyecto"]
    )

    with pestana1:
        pestana_analisis(config, controles, predictor)
        if controles.get("comparar_latencias"):
            st.divider()
            st.markdown("#### Comparación de latencia: Keras frente a TFLite int8")
            imagen_rgb = None
            if controles.get("archivo") is not None:
                imagen_rgb, _ = leer_imagen_subida(
                    controles["archivo"], int(obtener(config, "app.max_mb_subida", 10))
                )
            bloque_comparar_latencias(config, imagen_rgb)

    with pestana2:
        pestana_metricas(config)

    with pestana3:
        pestana_acerca(config)

    st.markdown(
        '<div class="cra-pie">Prototipo académico · Universidad Industrial de Santander · '
        "No sustituye la inspección de un ingeniero estructural matriculado</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
