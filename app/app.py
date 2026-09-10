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

import contextlib
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
from src.models.inferencia import (  # noqa: E402
    ETIQUETAS_MODELO,
    Predictor,
    ResultadoMosaicos,
    cargar_predictor,
    elegir_modelo,
    predecir_por_mosaicos,
)
from src.risk.reglas import evaluar_riesgo  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import resolver  # noqa: E402
from src.vision.camara import (  # noqa: E402
    LectorAsincrono,
    MedidorFPS,
    SuavizadorTemporal,
    abrir_camara,
    diagnosticar_fotograma,
    listar_camaras,
    normalizar_url_celular,
    recortar_centro,
    rectangulo_centro,
)
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

        # No hay selector de modelo, y esa ausencia es una decision, no una
        # simplificacion. Foto y video imponen restricciones opuestas: analizar
        # una fotografia admite un segundo de calculo y exige el maximo recall;
        # el video tiene 33 ms por fotograma y no admite ninguno. Un unico modelo
        # no puede ser el mejor en ambos, y pedirle al usuario que elija seria
        # trasladarle una decision que la aplicacion puede tomar mejor: la
        # respuesta correcta esta medida en el informe, no depende de su gusto.
        disponibles = artefactos_disponibles(config)
        estado["formato"] = elegir_modelo(config, "foto", disponibles)
        estado["formato_video"] = elegir_modelo(config, "video", disponibles)

        if estado["formato"] is None:
            st.error(
                "No hay ningun modelo entrenado.\n\n"
                "Ejecuta primero:\n"
                "`python scripts/train_transfer.py --config config.yaml`"
            )
        else:
            st.markdown(
                f"- **Fotografias** · {ETIQUETAS_MODELO[estado['formato']]} + mosaicos\n"
                f"- **Video en vivo** · {ETIQUETAS_MODELO.get(estado['formato_video'], '—')}"
            )
            st.caption(
                "La aplicacion elige el modelo segun lo que se este analizando. Para "
                "fotografias manda el acierto: **MobileNetV2 con mosaicos, F1 0.936** "
                "sobre 60 fotos propias. Para video manda la latencia: **TFLite int8, "
                "4.9 ms**, lo unico que sostiene 30 FPS.\n\n"
                "El ensemble se retiro de la interfaz: con mosaicos solo aporta "
                "**+0.015 de F1** —un falso positivo de sesenta— por un 26 % mas de "
                "tiempo. Lo que aportaba era compensar el error de escala que los "
                "mosaicos corrigen en su origen (§4.6). Sigue medible con el boton de "
                "abajo."
            )
            estado["comparar_latencias"] = st.button(
                "⚡ Comparar los tres en vivo", use_container_width=True
            )

        st.divider()
        st.markdown("### 🔍 Escala de analisis")
        estado["mosaicos"] = st.checkbox(
            "Analizar por mosaicos",
            value=bool(obtener(config, "app.mosaicos.activo", True)),
            help=(
                "Trocea la fotografia en ventanas y analiza cada una por separado.\n\n"
                "Una foto de telefono (1600x1200) se reduce 7.5 veces para entrar al "
                "modelo, que aprendio con parches que solo se reducian 1.4 veces. Una "
                "grieta de 3 px queda en 0.4 px: **desaparece antes de que el modelo la "
                "vea**. Trocear devuelve cada region a una escala reconocible."
            ),
        )
        if estado["mosaicos"]:
            estado["lado_mosaico"] = st.select_slider(
                "Lado del mosaico (px)",
                options=[227, 320, 480, 640],
                value=int(obtener(config, "app.mosaicos.lado_px", 480)),
                help=(
                    "Medido sobre 60 fotos propias (F1 / recall / segundos por foto):\n\n"
                    "- sin mosaicos — 0.889 / 0.80 / 0.32 s\n"
                    "- 227 px — 0.918 / 0.93 / 2.27 s\n"
                    "- 320 px — 0.936 / 0.97 / 1.33 s\n"
                    "- **480 px — 0.951 / 0.97 / 0.78 s**\n"
                    "- 640 px — 0.951 / 0.97 / 0.55 s\n\n"
                    "El optimo no esta en 227 px, el tamano de los parches de "
                    "entrenamiento: los mosaicos pequenos dejan la grieta sin contexto y "
                    "multiplican las falsas alarmas."
                ),
            )
            st.caption(
                "Sube el recall de **0.80 a 0.97** sobre fotografias propias (de 6 "
                "grietas perdidas a 1), a cambio de 2 falsas alarmas y algo mas de tiempo."
            )
        else:
            estado["lado_mosaico"] = int(obtener(config, "app.mosaicos.lado_px", 480))

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


def _conviene_trocear(imagen_rgb: np.ndarray, config: dict[str, Any]) -> bool:
    """Decide si trocear la imagen aporta algo.

    Trocear una imagen que ya es pequena no rescata ninguna grieta, porque no
    habia perdida de escala que rescatar, pero si multiplica las ocasiones de dar
    una falsa alarma. El troceado solo se activa cuando la imagen es lo bastante
    grande como para que el redimensionado destruya detalle.

    Args:
        imagen_rgb: Imagen RGB de entrada.
        config: Configuracion del proyecto.

    Returns:
        ``True`` si merece la pena analizarla por ventanas.
    """
    minimo = int(obtener(config, "app.mosaicos.lado_minimo_imagen", 600))
    alto, ancho = np.asarray(imagen_rgb).shape[:2]
    return min(alto, ancho) >= minimo


def marcar_mosaicos(
    imagen_rgb: np.ndarray, mosaicos: ResultadoMosaicos, umbral: float
) -> np.ndarray:
    """Senala sobre la imagen las ventanas donde el modelo vio evidencia.

    Es un efecto colateral valioso del troceado: analizando la imagen entera solo
    se obtiene "hay grieta o no la hay"; analizandola por ventanas se sabe
    **donde**. Para un inspector, esa es la diferencia entre un aviso y una
    indicacion util.

    Solo se dibujan las ventanas que superan el umbral, y se resalta la de mayor
    probabilidad. Dibujar tambien las demas llenaria la fotografia de recuadros y
    escondaria la informacion en lugar de mostrarla.

    Args:
        imagen_rgb: Imagen RGB sobre la que dibujar. No se modifica.
        mosaicos: Resultado del analisis por ventanas.
        umbral: Umbral de decision vigente.

    Returns:
        Copia de la imagen con las ventanas marcadas.
    """
    lienzo = np.asarray(imagen_rgb).copy()
    if mosaicos.indice_maximo < 0:
        return lienzo

    grosor = max(2, int(min(lienzo.shape[:2]) / 250))
    for indice, (x0, y0, x1, y1) in enumerate(mosaicos.ventanas):
        probabilidad = float(mosaicos.probabilidades[indice])
        if probabilidad < umbral:
            continue
        es_maximo = indice == mosaicos.indice_maximo
        color = (220, 30, 30) if es_maximo else (250, 170, 40)
        cv2.rectangle(lienzo, (x0, y0), (x1, y1), color, grosor * (2 if es_maximo else 1))
        if es_maximo:
            cv2.putText(
                lienzo,
                f"{probabilidad:.0%}",
                (x0 + grosor * 3, y0 + grosor * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                grosor * 0.35,
                color,
                max(1, grosor),
                cv2.LINE_AA,
            )
    return lienzo


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
    #
    # Se hace sobre la imagen ENTERA, no sobre 'imagen_vision': el redimensionado
    # a 1024 px que necesita OpenCV ya destruiria parte de la evidencia fina que
    # el analisis por mosaicos pretende rescatar.
    mosaicos = None
    if controles.get("mosaicos") and _conviene_trocear(imagen_rgb, config):
        mosaicos = predecir_por_mosaicos(
            predictor, imagen_rgb, config, lado=controles.get("lado_mosaico")
        )
        probabilidad = mosaicos.probabilidad
        ms_inferencia = mosaicos.milisegundos
    else:
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
        "mosaicos": mosaicos,
        "inclinacion": inclinacion,
        "orientacion": orientacion,
        "evaluacion": evaluacion,
        "imagen_vision": imagen_vision,
        "imagen_rgb": imagen_rgb,
        "imagen_anotada": cv2.cvtColor(anotada_bgr, cv2.COLOR_BGR2RGB),
        "ms_total": (time.perf_counter() - inicio_total) * 1000.0,
    }


def _panel_mosaicos(imagen_rgb: np.ndarray, mosaicos: ResultadoMosaicos, umbral: float) -> None:
    """Muestra donde encontro evidencia el analisis por ventanas.

    Args:
        imagen_rgb: Imagen original a tamano completo.
        mosaicos: Resultado del troceado.
        umbral: Umbral de decision vigente.
    """
    probabilidades = mosaicos.probabilidades
    activas = int((probabilidades >= umbral).sum())
    reduccion = min(np.asarray(imagen_rgb).shape[:2]) / 160.0

    st.markdown("#### Localizacion de la evidencia")
    izquierda, derecha = st.columns([3, 2], gap="medium")

    with izquierda:
        st.image(
            marcar_mosaicos(imagen_rgb, mosaicos, umbral),
            use_column_width=True,
            caption=(
                f"{activas} de {len(mosaicos.ventanas)} ventanas superan el umbral. "
                "En rojo, la de mayor probabilidad."
            ),
        )

    with derecha:
        st.markdown(
            f"La fotografia se analizo en **{len(mosaicos.ventanas)} ventanas de "
            f"{mosaicos.lado}x{mosaicos.lado} px** con un 50 % de solape, en lugar de "
            "reducir la imagen entera a 160 px."
        )
        st.markdown(
            f"- Ventana mas alta: **{probabilidades.max():.1%}**\n"
            f"- Mediana de las ventanas: **{float(np.median(probabilidades)):.1%}**\n"
            f"- Ventanas sobre el umbral: **{activas}**"
        )
        if activas == 0:
            st.caption(
                "Ninguna ventana supera el umbral: la superficie parece sana en toda su "
                "extension, no solo en promedio."
            )
        elif activas == 1:
            st.caption(
                "Dispara una sola ventana. Si la evidencia fuese ruido tenderia a "
                "aparecer dispersa; concentrada en una region es mas creible."
            )
        else:
            st.caption(
                "La evidencia aparece en varias ventanas, lo que es coherente con una "
                "fisura que recorre la superficie."
            )

        st.caption(
            f"**Por que se trocea:** esta foto se reduciria {reduccion:.1f} veces para "
            "entrar al modelo, que aprendio con parches que solo se reducian 1.4 veces. "
            "Una fisura fina no sobrevive a esa reduccion."
        )

    st.write("")


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
        mosaicos = resultado.get("mosaicos")
        nota_grieta = f"{'Grieta detectada' if hay_grieta else 'Sin grieta'} (umbral {umbral:.0%})"
        if mosaicos is not None:
            nota_grieta += f" · maximo de {len(mosaicos.ventanas)} ventanas de {mosaicos.lado} px"
        st.markdown(
            estilos.tarjeta_metrica(
                "Probabilidad de grieta",
                f"{resultado['probabilidad']:.1%}",
                nota=nota_grieta,
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

    # --- Localizacion por mosaicos -----------------------------------------
    if mosaicos is not None:
        _panel_mosaicos(resultado["imagen_rgb"], mosaicos, umbral)

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
            "MobileNetV2 sola (la medida de referencia es **+11 %**). La CNN de linea "
            "base es solo el 1.9 % de los parametros del conjunto pero aporta el 11 % "
            "del tiempo: el numero de parametros predice mal la latencia.\n\n"
            "**Por que ya no se usa.** Sin mosaicos compraba +0.089 de F1 (0.800 → "
            "0.889) con ese 11 %, y era una compra evidente. Con mosaicos compra "
            "**+0.015** (0.936 → 0.951) por un 26 % mas de tiempo: un unico falso "
            "positivo de sesenta. La ganancia nunca fue del ensemble, era del error de "
            "escala que compensaba, y ese error ya esta corregido en su origen (§4.6)."
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
# Pestana 4: camara en vivo
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Buscando camaras...", ttl=300)
def detectar_camaras(max_indice: int) -> list[int]:
    """Sondea que camaras hay conectadas, cacheando el resultado.

    El sondeo abre y cierra cada dispositivo, lo que tarda entre medio segundo y
    varios segundos. Sin cache se repetiria en cada rerun de Streamlit y la
    interfaz seria inusable. El TTL de 5 minutos permite detectar una camara
    conectada despues de arrancar la aplicacion.

    Args:
        max_indice: Ultimo indice a probar (exclusivo).

    Returns:
        Lista de indices disponibles.
    """
    return listar_camaras(max_indice)


@st.cache_resource(show_spinner=False)
def obtener_captura(fuente: int | str, ancho: int, alto: int) -> Any:
    """Abre la camara una sola vez y la mantiene entre reruns.

    Es ``cache_resource`` y no ``cache_data`` a proposito: el objeto de captura
    no es serializable y, sobre todo, **debe sobrevivir a los reruns**. Abrir y
    cerrar el dispositivo en cada fotograma costaria entre 200 y 500 ms, mas que
    todo el pipeline de analisis junto.

    Args:
        fuente: Indice del dispositivo local o URL del stream del celular.
        ancho: Ancho de captura solicitado.
        alto: Alto de captura solicitado.

    Returns:
        Objeto ``cv2.VideoCapture``, abierto o no.
    """
    captura = abrir_camara(fuente, ancho, alto)
    if not captura.isOpened():
        return captura
    # Se envuelve en el hilo lector: sin el, los fotogramas se acumulan en el
    # bufer de la fuente y el retardo entre mover la camara y verlo en pantalla
    # crece sin limite.
    return LectorAsincrono(captura)


def liberar_camara() -> None:
    """Cierra la camara y limpia la cache del recurso.

    Necesario porque ``cache_resource`` mantendria el dispositivo abierto -y el
    piloto de la webcam encendido- aunque el usuario detenga el analisis.
    """
    # Parar el hilo lector ANTES de limpiar la cache: si solo se limpiara la
    # cache, el hilo seguiria vivo leyendo y reteniendo el dispositivo.
    lector = st.session_state.pop("lector_activo", None)
    if lector is not None:
        with contextlib.suppress(Exception):
            lector.detener()
    with contextlib.suppress(Exception):
        obtener_captura.clear()


def _suavizadores(ventana: int) -> tuple[SuavizadorTemporal, SuavizadorTemporal]:
    """Devuelve los suavizadores de la sesion, recreandolos si cambio la ventana.

    Args:
        ventana: Tamano de ventana solicitado en la interfaz.

    Returns:
        Tupla ``(suavizador_probabilidad, suavizador_angulo)``.
    """
    if st.session_state.get("ventana_suavizado") != ventana:
        st.session_state.ventana_suavizado = ventana
        st.session_state.suave_prob = SuavizadorTemporal(ventana)
        st.session_state.suave_angulo = SuavizadorTemporal(ventana)
    return st.session_state.suave_prob, st.session_state.suave_angulo


def procesar_fotograma(
    fotograma_bgr: np.ndarray,
    predictor: Predictor,
    config: dict[str, Any],
    controles: dict[str, Any],
    fraccion: float,
) -> dict[str, Any]:
    """Analiza un fotograma y devuelve el resultado sin suavizar.

    Reparto deliberado del encuadre, que sale del hallazgo de §3.6 del informe:
    los dos modulos necesitan vistas distintas.

    - La **clasificacion** usa el recorte central, porque el modelo se entreno
      con parches en primer plano y una webcam da una vista amplia.
    - La **inclinometria** usa el fotograma completo, porque necesita ver el
      elemento vertical entero para encontrar su arista.

    Visto a la luz de §4.6, el recorte central resulta ser **el equivalente en
    tiempo real del analisis por mosaicos**: ambos evitan que la escena entera se
    reduzca a 160 px y borre la fisura. La diferencia es que aqui se mira una
    sola ventana en lugar de veinticuatro, porque a treinta fotogramas por
    segundo no hay presupuesto para mas. El precio es que solo se analiza el
    centro del encuadre, y por eso se dibuja el recuadro "zona analizada": la
    interfaz no debe dejar creer que se esta mirando todo lo que se ve.

    Args:
        fotograma_bgr: Fotograma capturado, en BGR.
        predictor: Predictor activo.
        config: Configuracion del proyecto.
        controles: Estado de los controles de OpenCV de la barra lateral.
        fraccion: Fraccion del recorte central para clasificar.

    Returns:
        Diccionario con ``probabilidad``, ``ms_inferencia``, ``inclinacion`` y
        ``anotada_rgb`` (el fotograma con las lineas y el recuadro dibujados).
    """
    recorte_bgr = recortar_centro(fotograma_bgr, fraccion)
    recorte_rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
    probabilidad, ms = predictor.predecir_imagen(recorte_rgb, config)

    inclinacion = estimar_inclinacion(
        fotograma_bgr,
        config,
        canny_bajo=controles["canny_bajo"],
        canny_alto=controles["canny_alto"],
        min_longitud=controles["min_longitud"],
        max_separacion=controles["max_separacion"],
        umbral_hough=controles["umbral_hough"],
    )

    anotada = anotar_imagen(
        fotograma_bgr, inclinacion, dibujar_descartadas=controles["mostrar_descartadas"]
    )
    x0, y0, x1, y1 = rectangulo_centro(fotograma_bgr, fraccion)
    cv2.rectangle(anotada, (x0, y0), (x1, y1), (255, 200, 0), 2)
    cv2.putText(
        anotada,
        "zona analizada",
        (x0 + 4, max(y0 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )

    return {
        "probabilidad": probabilidad,
        "ms_inferencia": ms,
        "inclinacion": inclinacion,
        "recorte_bgr": recorte_bgr,
        "anotada_rgb": cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB),
    }


def _tarjetas_vivo(
    prob: float | None,
    angulo: float | None,
    inclinacion: Any,
    fps: float,
    ms: float,
    umbral: float,
    estabilidad: float,
    edad_ms: float,
    columnas: int = 4,
) -> str:
    """Construye el HTML de las tarjetas del modo video.

    Args:
        prob: Probabilidad de grieta suavizada.
        angulo: Desaplome suavizado, o ``None``.
        inclinacion: Resultado de inclinometria del ultimo fotograma.
        fps: Tasa de fotogramas medida.
        ms: Milisegundos de inferencia del ultimo fotograma.
        umbral: Umbral de decision.
        estabilidad: Dispersion temporal de la probabilidad.
        edad_ms: Antiguedad del fotograma mostrado, en milisegundos.
        columnas: Cuantas tarjetas por fila. Se usa 1 cuando van en la columna
            estrecha junto al video, y 4 cuando ocupan el ancho completo.

    Returns:
        Fragmento HTML con cuatro tarjetas.
    """
    texto_prob = f"{prob:.1%}" if prob is not None else "—"
    nota_prob = (
        f"{'Grieta detectada' if (prob or 0) >= umbral else 'Sin grieta'} · "
        f"estabilidad ±{estabilidad:.3f}"
    )
    if angulo is not None and inclinacion.fiable:
        texto_ang, nota_ang = f"{angulo:+.2f}", f"{inclinacion.confianza} lineas coherentes"
    elif inclinacion.motivo_rechazo:
        texto_ang, nota_ang = "—", "Descartado por inverosimil"
    else:
        texto_ang, nota_ang = "—", "Sin elemento vertical"

    return (
        f'<div style="display:grid;grid-template-columns:repeat({max(1, columnas)},1fr);'
        'gap:0.8rem">'
        + estilos.tarjeta_metrica("P(grieta) suavizada", texto_prob, nota=nota_prob)
        + estilos.tarjeta_metrica("Desaplome", texto_ang, "°", nota_ang)
        + estilos.tarjeta_metrica(
            "Fotogramas", f"{fps:.1f}", "FPS", f"Retardo del ultimo: {edad_ms:.0f} ms"
        )
        + estilos.tarjeta_metrica("Inferencia", f"{ms:.1f}", "ms", "Ultimo fotograma")
        + "</div>"
    )


def pestana_camara(
    config: dict[str, Any], controles: dict[str, Any], predictor: Predictor | None
) -> None:
    """Renderiza la pestana de analisis con camara en vivo.

    Args:
        config: Configuracion del proyecto.
        controles: Estado de los controles de la barra lateral.
        predictor: Predictor activo, o ``None`` si no hay modelo.
    """
    cfg = obtener(config, "app.camara", {}) or {}
    st.markdown(estilos.aviso_legal(obtener(config, "app.aviso_legal", "")), unsafe_allow_html=True)

    if predictor is None:
        st.info("Entrena un modelo para habilitar el analisis en vivo.")
        return

    modo = st.radio(
        "Modo de captura",
        ["video", "foto"],
        horizontal=True,
        format_func=lambda v: (
            "🎥 Vídeo continuo" if v == "video" else "📸 Foto a foto (navegador)"
        ),
        help=(
            "**Vídeo continuo** lee la webcam desde el servidor con OpenCV: es la "
            "demo en tiempo real, y funciona en local.\n\n"
            "**Foto a foto** usa la cámara del navegador y analiza una instantánea. "
            "Más lento pero no depende de que OpenCV vea el dispositivo."
        ),
    )

    if modo == "foto":
        # Una instantanea es una fotografia: se analiza con el modelo de foto y
        # con mosaicos, igual que una imagen subida.
        _modo_foto(config, controles, predictor)
        return

    # El video usa su propio modelo. Cargarlo aqui y no en main() evita que abrir
    # la aplicacion cargue en memoria un modelo que quiza no se llegue a usar.
    formato_video = controles.get("formato_video")
    predictor_video = predictor
    if formato_video and formato_video != predictor.formato:
        try:
            clave, marca = clave_y_marca(config, formato_video)
            predictor_video = obtener_predictor(formato_video, clave, marca)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            st.warning(
                f"No se pudo cargar **{ETIQUETAS_MODELO.get(formato_video, formato_video)}** "
                f"para el vídeo ({error}). Se usará "
                f"**{ETIQUETAS_MODELO.get(predictor.formato, predictor.formato)}**, "
                "que es más lento y bajará los FPS."
            )

    _modo_video(config, controles, predictor_video, cfg)


def _modo_foto(config: dict[str, Any], controles: dict[str, Any], predictor: Predictor) -> None:
    """Modo de instantanea con la camara del navegador.

    Es la red de seguridad para la sustentacion: no depende de que OpenCV pueda
    abrir el dispositivo, solo de que el navegador tenga permiso de camara.

    Args:
        config: Configuracion del proyecto.
        controles: Estado de los controles de la barra lateral.
        predictor: Predictor activo.
    """
    captura = st.camera_input("Toma una foto del elemento")
    if captura is None:
        st.caption(
            "El navegador pedirá permiso para usar la cámara. La imagen se procesa "
            "en local y no se envía a ningún servidor externo."
        )
        return

    imagen_rgb, error = leer_imagen_subida(captura, int(obtener(config, "app.max_mb_subida", 10)))
    if imagen_rgb is None:
        st.error(f"No se pudo procesar la captura. {error}")
        return

    with st.spinner("Analizando..."):
        resultado = ejecutar_pipeline(imagen_rgb, predictor, config, controles)

    st.markdown(
        estilos.semaforo(resultado["evaluacion"].nivel, resultado["evaluacion"].resumen),
        unsafe_allow_html=True,
    )
    izquierda, derecha = st.columns(2, gap="medium")
    with izquierda:
        st.image(resultado["imagen_vision"], use_column_width=True)
    with derecha:
        st.image(resultado["imagen_anotada"], use_column_width=True)

    for regla in sorted(resultado["evaluacion"].reglas, key=lambda r: -r.severidad):
        st.markdown(
            estilos.regla(
                regla.codigo, regla.titulo, regla.detalle, regla.justificacion, regla.severidad
            ),
            unsafe_allow_html=True,
        )


def _modo_video(
    config: dict[str, Any],
    controles: dict[str, Any],
    predictor: Predictor,
    cfg: dict[str, Any],
) -> None:
    """Modo de video continuo leyendo la webcam con OpenCV.

    Args:
        config: Configuracion del proyecto.
        controles: Estado de los controles de la barra lateral.
        predictor: Predictor activo.
        cfg: Seccion ``app.camara`` de la configuracion.
    """
    recomendado = str(cfg.get("modelo_recomendado", "tflite"))
    if predictor.formato == recomendado:
        st.caption(
            f"Modelo en uso: **{ETIQUETAS_MODELO.get(predictor.formato, predictor.formato)}** "
            "(4.9 ms por fotograma). El vídeo se analiza con un modelo distinto al de las "
            "fotografías, y a propósito: aquí hay 33 ms por fotograma para sostener 30 FPS, "
            "y ningún otro cabe en ese presupuesto. El precio es menos recall por fotograma, "
            "que el suavizado temporal compensa en parte agregando varias observaciones de "
            "la misma escena."
        )
    else:
        st.warning(
            f"El vídeo está usando **{ETIQUETAS_MODELO.get(predictor.formato, predictor.formato)}**, "
            f"que tarda ~{'190' if predictor.formato == 'ensemble' else '170'} ms por "
            "fotograma: irá a unos 5 FPS. Lo esperado es **TFLite int8** (4.9 ms). "
            "Expórtalo con `python scripts/export_tflite.py --config config.yaml` para "
            "recuperar el vídeo fluido."
        )

    # --- Seleccion de la fuente de video ------------------------------------
    fuente_tipo = st.radio(
        "Fuente de vídeo",
        ["equipo", "celular"],
        horizontal=True,
        index=0 if str(cfg.get("fuente", "equipo")) == "equipo" else 1,
        format_func=lambda v: (
            "💻 Webcam del equipo" if v == "equipo" else "📱 Cámara del celular (por red)"
        ),
    )

    if fuente_tipo == "celular":
        with st.expander(
            "Cómo conectar el celular", expanded=not st.session_state.get("url_celular")
        ):
            st.markdown(
                """
**1.** Instala en el celular una app de cámara IP:

| App | Sistema | Puerto |
|---|---|---|
| **IP Webcam** | Android | 8080 |
| **DroidCam** | Android / iOS | 4747 |

**2.** Conecta el celular **a la misma red Wi-Fi** que este equipo.

**3.** Abre la app y pulsa *Iniciar servidor*. Te mostrará una dirección
del tipo `http://192.168.1.40:8080`.

**4.** Cópiala abajo. Puedes pegar solo `192.168.1.40:8080`: se completa sola.

La captura la hace el teléfono de forma nativa, así que **no hace falta HTTPS**
ni instalar nada en el PC. El vídeo viaja por tu red local; no sale a internet.
                """
            )

        texto_url = st.text_input(
            "Dirección del stream",
            value=st.session_state.get("url_celular", str(cfg.get("url_celular", ""))),
            placeholder="192.168.1.40:8080",
            help="Se admite la forma abreviada; se completa a http://IP:puerto/video.",
        )
        url = normalizar_url_celular(texto_url)
        st.session_state.url_celular = texto_url

        if not url:
            st.info("Escribe la dirección que muestra la app del celular para continuar.")
            return
        st.caption(f"Se conectará a `{url}`")
        fuente: int | str = url
    else:
        # El sondeo de dispositivos se hace SOLO cuando el usuario lo pide: abrir
        # cada indice enciende brevemente el piloto de la webcam, y hacerlo por el
        # mero hecho de abrir la pestana seria una sorpresa desagradable.
        if "camaras_detectadas" not in st.session_state:
            st.info(
                "Para empezar hay que localizar los dispositivos conectados. La búsqueda "
                "abre y cierra cada cámara, así que verás su piloto encenderse un instante."
            )
            if st.button("🔎 Buscar cámaras"):
                st.session_state.camaras_detectadas = detectar_camaras(3)
                st.rerun()
            return

        camaras = st.session_state.camaras_detectadas
        if not camaras:
            st.error(
                "**No se detectó ninguna cámara.** Comprueba que está conectada, que "
                "ninguna otra aplicación la esté usando y que el sistema le da permiso.\n\n"
                "Puedes usar el modo **Foto a foto**, o conectar la cámara del celular."
            )
            if st.button("🔄 Volver a buscar"):
                del st.session_state["camaras_detectadas"]
                detectar_camaras.clear()
                st.rerun()
            return

        fuente = st.selectbox(
            "Cámara",
            camaras,
            index=(
                camaras.index(int(cfg.get("indice", 0)))
                if int(cfg.get("indice", 0)) in camaras
                else 0
            ),
            format_func=lambda i: f"Dispositivo {i}",
        )
    # El tamano de la vista se queda visible; los otros dos controles van
    # plegados. Cada bloque de ajustes que ocupa alto empuja hacia abajo el video
    # y el veredicto, que son las dos cosas que hay que mirar mientras se inspecciona.
    # Los ajustes se tocan una vez; el video y el riesgo, todo el rato.
    columna_a, columna_b = st.columns([2, 3])
    with columna_a:
        # Un stream de telefono llega en vertical (1200x1600) y a ancho completo
        # desplaza el semaforo de riesgo fuera de la pantalla.
        ancho_vista = st.select_slider(
            "Tamaño del vídeo",
            options=[320, 400, 480, 560, 640],
            value=int(cfg.get("ancho_vista", 400)),
            help=(
                "Ancho en píxeles de la previsualización. No afecta al análisis: "
                "el modelo siempre recibe el recorte a resolución completa."
            ),
        )
    with columna_b:
        st.caption(
            "Si el vídeo tapa el semáforo de riesgo, baja este valor. La imagen es solo "
            "para encuadrar: el clasificador recibe siempre el recorte a resolución "
            "completa, así que reducirla **no** empeora la detección."
        )

    with st.expander("⚙ Ajustes de análisis"):
        fraccion = st.slider(
            "Zona analizada",
            0.2,
            1.0,
            float(cfg.get("fraccion_recorte", 0.6)),
            0.05,
            help=(
                "Fracción central del fotograma que se pasa al clasificador. El "
                "modelo se entrenó con primeros planos: sin recorte, una grieta "
                "queda reducida a dos o tres píxeles."
            ),
        )
        ventana = st.slider(
            "Suavizado temporal",
            1,
            15,
            int(cfg.get("ventana_suavizado", 7)),
            help=(
                "Fotogramas sobre los que se toma la mediana. Con 1 verás el "
                "parpadeo que motiva esta función."
            ),
        )

    activa = bool(st.session_state.get("camara_activa", False))
    boton_izq, boton_der = st.columns(2)
    with boton_izq:
        if st.button("▶ Iniciar", use_container_width=True, disabled=activa):
            st.session_state.camara_activa = True
            st.rerun()
    with boton_der:
        if st.button("⏹ Detener", use_container_width=True, disabled=not activa):
            st.session_state.camara_activa = False
            liberar_camara()
            st.rerun()

    # Video y veredicto lado a lado, no apilados. Apilados obligaban a hacer
    # scroll para pasar de "que estoy enfocando" a "que riesgo tiene", que son
    # justamente las dos cosas que hay que leer juntas mientras se mueve la
    # camara: si no se ven a la vez, no se puede saber que encuadre produjo que
    # resultado.
    columna_video, columna_datos = st.columns([3, 2], gap="medium")
    with columna_video:
        marcador_video = st.empty()
        marcador_avisos = st.empty()
    with columna_datos:
        marcador_semaforo = st.empty()
        marcador_tarjetas = st.empty()

    if not activa:
        marcador_video.info(
            "Pulsa **Iniciar** para abrir la cámara. Apunta al elemento de forma que "
            "su borde vertical quede dentro del encuadre y la superficie a inspeccionar "
            "dentro del recuadro naranja."
        )
        return

    captura = obtener_captura(fuente, int(cfg.get("ancho", 640)), int(cfg.get("alto", 480)))
    if not captura.isOpened():
        st.session_state.camara_activa = False
        liberar_camara()
        marcador_video.error(
            f"No se pudo abrir la fuente `{fuente}`. Si es el celular, comprueba que la "
            "app sigue emitiendo y que ambos estan en la misma red Wi-Fi. Si es la "
            "webcam, puede estar en uso por otra "
            "aplicación. Cierra Zoom, Teams o el navegador y vuelve a intentarlo."
        )
        return

    st.session_state.lector_activo = captura
    suave_prob, suave_angulo = _suavizadores(ventana)
    medidor = st.session_state.setdefault("medidor_fps", MedidorFPS())
    umbral = float(obtener(config, "riesgo.umbral_grieta", 0.5))
    # Bucle CONTINUO, sin st.rerun() periodico.
    #
    # La version anterior capturaba en rafagas de un segundo y despues llamaba a
    # st.rerun(). Funcionaba, pero cada rerun repinta la pagina entera: el video
    # parpadeaba una vez por segundo y los textos cambiaban demasiado deprisa
    # para poder leerlos.
    #
    # Streamlit comprueba si hay una interaccion pendiente cada vez que se
    # actualiza un elemento, asi que este bucle **si es interrumpible**: al
    # pulsar Detener, la llamada a marcador_video.image() lanza la excepcion de
    # rerun y el script se reinicia. No hace falta trocear la captura.
    intervalo_texto = float(cfg.get("segundos_entre_textos", 0.4))
    ultimo_texto = 0.0
    fotogramas = 0

    while st.session_state.get("camara_activa"):
        t_lectura = time.perf_counter()
        leido, fotograma, edad_ms = captura.leer()
        if not leido:
            # El hilo aun no tiene el primer fotograma.
            time.sleep(0.02)
            if time.perf_counter() - t_lectura > 5.0:
                break
            continue

        # Solo se diagnostica el primer fotograma de la rafaga: basta para
        # detectar el problema y no cuesta nada en los siguientes.
        if fotogramas == 0:
            # Se pasa la ANTIGUEDAD, no el tiempo de lectura: con el hilo
            # lector, read() devuelve al instante y ya no mide nada util.
            problema = diagnosticar_fotograma(fotograma, edad_ms)
            if problema:
                st.session_state.camara_activa = False
                liberar_camara()
                marcador_video.error(f"**Problema de captura.** {problema}")
                return

        if not leido or fotograma is None:
            break

        # Espejo: mover la camara a la derecha debe mover la imagen a la derecha.
        fotograma = cv2.flip(fotograma, 1)
        resultado = procesar_fotograma(fotograma, predictor, config, controles, fraccion)

        prob = suave_prob.agregar(resultado["probabilidad"])
        inclinacion = resultado["inclinacion"]
        angulo = suave_angulo.agregar(inclinacion.angulo_grados if inclinacion.fiable else None)
        medidor.marcar()
        fotogramas += 1

        # Esta llamada es tambien el punto donde Streamlit puede interrumpir el
        # bucle si el usuario pulsa Detener.
        # width en pixeles y no use_column_width: la anchura de la columna varia
        # con el navegador, y un stream vertical a ancho completo empuja el
        # veredicto fuera de la pantalla.
        marcador_video.image(resultado["anotada_rgb"], width=ancho_vista)
        st.session_state.edad_fotograma = edad_ms

        # La orientacion se mide sobre el RECORTE, nunca sobre el fotograma
        # anotado: este ultimo lleva las lineas verdes de Hough dibujadas
        # encima, y volver a pasarle Canny detectaria esas lineas en vez de la
        # grieta. Ademas solo se calcula cuando hay grieta, porque es lo unico
        # que el motor de reglas usa (R3/R4 exigen hay_grieta) y cada llamada es
        # una pasada completa de Canny+Hough: ~3 ms por fotograma que no se
        # gastan cuando no hacen falta.
        if (prob or 0.0) >= umbral:
            orientacion_txt = clasificar_orientacion_grieta(
                resultado["recorte_bgr"],
                config,
                canny_bajo=controles["canny_bajo"],
                canny_alto=controles["canny_alto"],
            ).orientacion
        else:
            orientacion_txt = "indeterminada"

        evaluacion = evaluar_riesgo(
            probabilidad_grieta=float(prob or 0.0),
            config=config,
            elemento=controles["elemento"],
            orientacion_grieta=orientacion_txt,
            angulo_desaplome=angulo,
            confianza_inclinacion=inclinacion.confianza if inclinacion.fiable else 0,
        )

        # El video se actualiza en cada fotograma; los textos, varias veces por
        # segundo. Repintar las tarjetas y el semaforo a 15 Hz los vuelve
        # ilegibles y ademas cuesta mas que el propio analisis.
        ahora = time.perf_counter()
        if ahora - ultimo_texto < intervalo_texto:
            continue
        ultimo_texto = ahora

        marcador_tarjetas.markdown(
            _tarjetas_vivo(
                prob,
                angulo,
                inclinacion,
                medidor.fps(),
                resultado["ms_inferencia"],
                umbral,
                suave_prob.estabilidad(),
                edad_ms,
                columnas=1,
            ),
            unsafe_allow_html=True,
        )

        marcador_semaforo.markdown(
            estilos.semaforo(evaluacion.nivel, evaluacion.resumen), unsafe_allow_html=True
        )

        # Siempre el MISMO tipo de elemento. Alternar entre caption y warning
        # cambia la altura del bloque y hace saltar todo lo que hay debajo, que
        # es la otra mitad de la sensacion de parpadeo.
        if not suave_prob.lleno():
            aviso = f"⏳ Estabilizando: {len(suave_prob)}/{ventana} fotogramas en la ventana."
        elif inclinacion.motivo_rechazo:
            aviso = f"⚠️ Desaplome descartado: {inclinacion.motivo_rechazo}"
        else:
            aviso = (
                f"✓ Nivel calculado sobre la mediana de los últimos {ventana} "
                "fotogramas, no sobre el actual."
            )
        marcador_avisos.caption(aviso)

    if fotogramas == 0:
        st.session_state.camara_activa = False
        liberar_camara()
        marcador_video.error(
            "La cámara se abrió pero no entregó ningún fotograma. Comprueba que no "
            "esté siendo usada por otra aplicación."
        )
        return


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
        clave, marca = clave_y_marca(config, controles["formato"])
        try:
            predictor = obtener_predictor(controles["formato"], clave, marca)
            panel_info_modelo(predictor)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            st.sidebar.error(f"No se pudo cargar el modelo: {error}")

    pestana1, pestana2, pestana3, pestana4 = st.tabs(
        [
            "🔍 Análisis en vivo",
            "📹 Cámara en vivo",
            "📊 Métricas del modelo",
            "ℹ️ Acerca del proyecto",
        ]
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
        pestana_camara(config, controles, predictor)

    with pestana3:
        # Mientras la camara esta activa la pagina se recarga cada segundo.
        # Reconstruir cinco figuras de Plotly en cada rerun robaria fotogramas al
        # video sin que nadie las este mirando.
        if st.session_state.get("camara_activa"):
            st.info(
                "Métricas en pausa mientras la cámara está activa, para no robarle "
                "fotogramas al vídeo. Detén la cámara para volver a verlas."
            )
        else:
            pestana_metricas(config)

    with pestana4:
        pestana_acerca(config)

    st.markdown(
        '<div class="cra-pie">Prototipo académico · Universidad Industrial de Santander · '
        "No sustituye la inspección de un ingeniero estructural matriculado</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
