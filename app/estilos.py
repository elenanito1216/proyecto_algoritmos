"""Paleta, CSS personalizado y componentes visuales de la aplicacion.

Criterios de diseno
-------------------
1. **Dos colores de acento y nada mas.** Un azul cian para lo interactivo y un
   ambar para lo que reclama atencion. Todo lo demas son neutros. Mas colores en
   un panel de seguridad significan menos jerarquia visual, no mas informacion.
2. **El tema manda.** Los fondos, textos y bordes se derivan de las variables CSS
   que Streamlit publica segun el tema activo
   (``--background-color``, ``--secondary-background-color``, ``--text-color``).
   Fijar un ``#FFFFFF`` a mano deja la interfaz ilegible en cuanto el usuario
   cambia de tema, y la sustentacion se hace en un equipo ajeno.
3. **El color nunca va solo.** Cada nivel de riesgo lleva icono y texto ademas
   del color. Entre el 5 y el 8% de los hombres tiene alguna deficiencia en la
   vision del rojo y el verde; en un panel que clasifica riesgo estructural, un
   semaforo que solo se distingue por el tono es un fallo de diseno, no un
   detalle estetico.
"""

from __future__ import annotations

from typing import Any

ACENTO_PRIMARIO = "#22B8CF"
ACENTO_SECUNDARIO = "#F59E0B"

COLOR_BAJO = "#16A34A"
COLOR_MEDIO = "#D97706"
COLOR_ALTO = "#DC2626"

ICONO_NIVEL: dict[str, str] = {"Bajo": "✓", "Medio": "▲", "Alto": "✕"}
COLOR_NIVEL: dict[str, str] = {"Bajo": COLOR_BAJO, "Medio": COLOR_MEDIO, "Alto": COLOR_ALTO}
TEXTO_NIVEL: dict[str, str] = {
    "Bajo": "Sin indicios relevantes",
    "Medio": "Requiere seguimiento",
    "Alto": "Requiere revision profesional",
}


def css() -> str:
    """Genera la hoja de estilos de la aplicacion.

    El CSS cubre solo lo que el tema de Streamlit no alcanza: tarjetas con
    sombra, bordes redondeados en imagenes, tipografia de las metricas y el
    semaforo de riesgo. No reimplementa el tema.

    Returns:
        Bloque ``<style>`` listo para inyectar con ``st.markdown``.
    """
    return f"""
<style>
:root {{
    --acento: {ACENTO_PRIMARIO};
    --acento-2: {ACENTO_SECUNDARIO};
    --riesgo-bajo: {COLOR_BAJO};
    --riesgo-medio: {COLOR_MEDIO};
    --riesgo-alto: {COLOR_ALTO};
    --superficie: var(--secondary-background-color, rgba(128,128,128,0.10));
    --borde: rgba(128, 128, 128, 0.28);
    --sombra: 0 1px 2px rgba(0,0,0,0.08), 0 6px 20px rgba(0,0,0,0.10);
    --radio: 14px;
}}

/* El contenido se centra y se limita para que no se estire de forma ilegible
   en un monitor ancho, pero sigue cabiendo entero en un portatil de 1366 px. */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* ---------- Cabecera ---------- */
.cra-cabecera {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.1rem 1.4rem;
    border-radius: var(--radio);
    background: linear-gradient(135deg,
                rgba(34, 184, 207, 0.16) 0%,
                rgba(34, 184, 207, 0.04) 55%,
                transparent 100%);
    border: 1px solid var(--borde);
    border-left: 4px solid var(--acento);
    margin-bottom: 1.2rem;
}}
.cra-cabecera h1 {{
    font-size: 1.55rem;
    line-height: 1.2;
    margin: 0;
    letter-spacing: -0.015em;
    font-weight: 700;
}}
.cra-cabecera p {{
    margin: 0.25rem 0 0 0;
    font-size: 0.92rem;
    opacity: 0.75;
}}
.cra-cabecera .cra-icono {{ font-size: 2.1rem; line-height: 1; }}

/* ---------- Tarjetas ---------- */
.cra-tarjeta {{
    background: var(--superficie);
    border: 1px solid var(--borde);
    border-radius: var(--radio);
    padding: 1.05rem 1.2rem;
    box-shadow: var(--sombra);
    height: 100%;
}}
.cra-tarjeta h4 {{
    margin: 0 0 0.55rem 0;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    opacity: 0.62;
}}
.cra-valor {{
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}}
.cra-unidad {{ font-size: 0.95rem; font-weight: 500; opacity: 0.6; margin-left: 0.2rem; }}
.cra-nota {{ font-size: 0.8rem; opacity: 0.68; margin-top: 0.35rem; line-height: 1.45; }}

/* ---------- Semaforo de riesgo ---------- */
.cra-semaforo {{
    border-radius: var(--radio);
    padding: 1.15rem 1.35rem;
    box-shadow: var(--sombra);
    border: 1px solid var(--borde);
    display: flex;
    align-items: center;
    gap: 1.1rem;
}}
.cra-semaforo .cra-disco {{
    width: 58px; height: 58px;
    min-width: 58px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.7rem; font-weight: 700; color: #ffffff;
}}
.cra-semaforo .cra-nivel {{ font-size: 1.45rem; font-weight: 700; line-height: 1.15; }}
.cra-semaforo .cra-glosa {{ font-size: 0.88rem; opacity: 0.78; margin-top: 0.15rem; }}
.cra-bajo   {{ border-left: 5px solid var(--riesgo-bajo);  background: rgba(22,163,74,0.10); }}
.cra-medio  {{ border-left: 5px solid var(--riesgo-medio); background: rgba(217,119,6,0.10); }}
.cra-alto   {{ border-left: 5px solid var(--riesgo-alto);  background: rgba(220,38,38,0.10); }}

/* ---------- Reglas disparadas ---------- */
.cra-regla {{
    border-left: 3px solid var(--borde);
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.55rem;
    background: var(--superficie);
    border-radius: 0 10px 10px 0;
}}
.cra-regla.sev-2 {{ border-left-color: var(--riesgo-alto); }}
.cra-regla.sev-1 {{ border-left-color: var(--riesgo-medio); }}
.cra-regla.sev-0 {{ border-left-color: var(--riesgo-bajo); }}
.cra-regla .cra-codigo {{
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.1rem 0.42rem;
    border-radius: 5px;
    background: rgba(128,128,128,0.20);
    margin-right: 0.5rem;
}}
.cra-regla .cra-titulo {{ font-weight: 650; font-size: 0.95rem; }}
.cra-regla .cra-detalle {{ font-size: 0.85rem; opacity: 0.82; margin-top: 0.25rem; line-height: 1.5; }}
.cra-regla .cra-justif {{
    font-size: 0.79rem; opacity: 0.62; margin-top: 0.35rem;
    font-style: italic; line-height: 1.5;
}}

/* ---------- Imagenes ---------- */
div[data-testid="stImage"] img {{
    border-radius: 12px;
    border: 1px solid var(--borde);
    box-shadow: var(--sombra);
}}

/* ---------- Aviso legal permanente ---------- */
.cra-aviso {{
    border: 1px solid rgba(245, 158, 11, 0.45);
    background: rgba(245, 158, 11, 0.11);
    border-radius: 10px;
    padding: 0.65rem 0.95rem;
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 0.6rem 0 1.1rem 0;
}}
.cra-aviso strong {{ color: var(--acento-2); }}

/* ---------- Pestanas ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px 9px 0 0;
    padding: 0.45rem 1.05rem;
    font-weight: 600;
}}

/* ---------- Barra lateral ---------- */
section[data-testid="stSidebar"] .cra-tarjeta {{ padding: 0.8rem 0.9rem; }}
.cra-fila-dato {{
    display: flex; justify-content: space-between; gap: 0.6rem;
    font-size: 0.82rem; padding: 0.22rem 0;
    border-bottom: 1px dashed var(--borde);
}}
.cra-fila-dato:last-child {{ border-bottom: none; }}
.cra-fila-dato .cra-k {{ opacity: 0.66; }}
.cra-fila-dato .cra-v {{ font-weight: 650; font-variant-numeric: tabular-nums; text-align: right; }}

.cra-pie {{
    text-align: center; font-size: 0.76rem; opacity: 0.5;
    margin-top: 2.2rem; padding-top: 0.9rem;
    border-top: 1px solid var(--borde);
}}

/* En pantallas pequenas o proyectadas, las columnas se apilan solas: se evita
   que las tarjetas fuercen scroll horizontal. */
@media (max-width: 900px) {{
    .cra-valor {{ font-size: 1.5rem; }}
    .cra-cabecera h1 {{ font-size: 1.25rem; }}
    .cra-semaforo {{ flex-direction: column; align-items: flex-start; }}
}}
</style>
"""


def inyectar_estilos(st: Any) -> None:
    """Inyecta la hoja de estilos en la pagina.

    Args:
        st: Modulo ``streamlit`` (se pasa como argumento para no importar
            Streamlit en un modulo que tambien se quiere poder inspeccionar sin
            arrancar el servidor).
    """
    st.markdown(css(), unsafe_allow_html=True)


def cabecera(titulo: str, subtitulo: str, icono: str = "🏗️") -> str:
    """Construye la cabecera principal del panel.

    Args:
        titulo: Titulo de la aplicacion.
        subtitulo: Linea descriptiva.
        icono: Emoji decorativo.

    Returns:
        Fragmento HTML.
    """
    return (
        f'<div class="cra-cabecera"><div class="cra-icono">{icono}</div>'
        f"<div><h1>{titulo}</h1><p>{subtitulo}</p></div></div>"
    )


def tarjeta_metrica(titulo: str, valor: str, unidad: str = "", nota: str = "") -> str:
    """Construye una tarjeta de metrica con sombra.

    Args:
        titulo: Encabezado en mayusculas pequenas.
        valor: Valor principal.
        unidad: Sufijo del valor (p. ej. ``"ms"``).
        nota: Texto secundario explicativo.

    Returns:
        Fragmento HTML.
    """
    html_unidad = f'<span class="cra-unidad">{unidad}</span>' if unidad else ""
    html_nota = f'<div class="cra-nota">{nota}</div>' if nota else ""
    return (
        f'<div class="cra-tarjeta"><h4>{titulo}</h4>'
        f'<div class="cra-valor">{valor}{html_unidad}</div>{html_nota}</div>'
    )


def semaforo(nivel: str, resumen: str) -> str:
    """Construye el semaforo de riesgo.

    El color se acompana **siempre** de un icono y de una etiqueta de texto, de
    modo que la informacion sigue siendo completa para una persona con
    deficiencia en la percepcion del color o en una proyeccion desvaida.

    Args:
        nivel: ``"Bajo"``, ``"Medio"`` o ``"Alto"``.
        resumen: Frase de resumen de la evaluacion.

    Returns:
        Fragmento HTML.
    """
    color = COLOR_NIVEL.get(nivel, COLOR_MEDIO)
    icono = ICONO_NIVEL.get(nivel, "▲")
    glosa = TEXTO_NIVEL.get(nivel, "")
    clase = f"cra-{nivel.lower()}"
    return (
        f'<div class="cra-semaforo {clase}">'
        f'<div class="cra-disco" style="background:{color}">{icono}</div>'
        f'<div><div class="cra-nivel" style="color:{color}">Riesgo {nivel}</div>'
        f'<div class="cra-glosa">{glosa}. {resumen}</div></div></div>'
    )


def regla(codigo: str, titulo: str, detalle: str, justificacion: str, severidad: int) -> str:
    """Construye la tarjeta de una regla disparada.

    Args:
        codigo: Identificador de la regla.
        titulo: Enunciado breve.
        detalle: Valores concretos que la activaron.
        justificacion: Criterio de ingenieria que la motiva.
        severidad: 0, 1 o 2.

    Returns:
        Fragmento HTML.
    """
    return (
        f'<div class="cra-regla sev-{severidad}">'
        f'<span class="cra-codigo">{codigo}</span>'
        f'<span class="cra-titulo">{titulo}</span>'
        f'<div class="cra-detalle">{detalle}</div>'
        f'<div class="cra-justif">Criterio: {justificacion}</div></div>'
    )


def aviso_legal(texto: str) -> str:
    """Construye el aviso legal permanente.

    Args:
        texto: Texto del aviso, tomado de ``config.yaml``.

    Returns:
        Fragmento HTML.
    """
    return f'<div class="cra-aviso"><strong>⚠ Aviso:</strong> {texto}</div>'


def fila_dato(clave: str, valor: str) -> str:
    """Construye una fila de dato clave-valor para la barra lateral.

    Args:
        clave: Nombre del dato.
        valor: Valor formateado.

    Returns:
        Fragmento HTML.
    """
    return f'<div class="cra-fila-dato"><span class="cra-k">{clave}</span><span class="cra-v">{valor}</span></div>'


def plantilla_plotly() -> dict[str, Any]:
    """Devuelve los ajustes de layout comunes a todas las figuras de Plotly.

    Fondo blanco solido y texto negro, fijados en la figura y no heredados. Las
    figuras se pintan con ``theme=None`` en ``st.plotly_chart``: con el tema de
    Streamlit activo, la aplicacion reescribia el color de la letra al del tema y
    sobre el fondo blanco quedaba texto blanco, es decir invisible.

    Antes el fondo era transparente para heredar el
    tema, pero sobre fondo oscuro las rejillas y los ejes quedaban al limite de
    lo legible, y una grafica que hay que forzar la vista para leer no cumple su
    funcion. Con fondo propio la figura se lee igual en cualquier tema y se puede
    capturar para el informe sin que salga con el fondo de la aplicacion.

    Returns:
        Diccionario de argumentos para ``figura.update_layout``.
    """
    tinta = "#111111"
    rejilla = {
        "gridcolor": "rgba(17,17,17,0.14)",
        "zerolinecolor": "rgba(17,17,17,0.30)",
        "linecolor": "rgba(17,17,17,0.45)",
        "tickfont": {"color": tinta, "size": 11},
        "title": {"font": {"color": tinta, "size": 12}},
    }
    return {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"color": tinta, "size": 12},
        "xaxis": dict(rejilla),
        "yaxis": dict(rejilla),
        "margin": {"l": 55, "r": 25, "t": 55, "b": 50},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": tinta},
        },
        "hoverlabel": {"font_size": 12},
    }
