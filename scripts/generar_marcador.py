"""Genera la hoja imprimible con el marcador de escala.

Para que sirve
--------------
El sistema puede medir una grieta en pixeles, pero para convertirla a milimetros
necesita un objeto de tamano conocido en la escena (ver ``src/vision/escala.py``
y §5.1 del informe). Este programa produce ese objeto: una hoja con un marcador
ArUco impreso a un tamano exacto, lista para llevar a la inspeccion.

El detalle que decide si funciona
---------------------------------
El marcador solo sirve si su tamano impreso es **exactamente** el declarado. Por
eso la imagen se genera a una resolucion calculada para el tamano fisico pedido,
y la hoja lleva impresa una regla de comprobacion: basta medirla con un
escalimetro para confirmar que la impresora no reescalo nada.

Ese es el fallo silencioso de este metodo. Si se imprime "ajustar a la pagina",
el marcador sale mas pequeno, el sistema calcula una equivalencia erronea y
**todas las medidas en milimetros salen mal sin que nada lo delate**: las cifras
tienen el mismo aspecto correcto. Hay que imprimir al 100 %, y hay que
comprobarlo.

Uso tipico:
    python scripts/generar_marcador.py
    python scripts/generar_marcador.py --lado-mm 80 --id 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import asegurar_directorio, resolver  # noqa: E402

MM_POR_PULGADA = 25.4


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(description="Genera la hoja del marcador de escala.")
    parser.add_argument("--config", default="config.yaml", help="Ruta del YAML de configuracion.")
    parser.add_argument(
        "--lado-mm",
        type=float,
        default=None,
        help="Lado del marcador en milimetros. Por defecto, el de config.yaml.",
    )
    parser.add_argument("--id", type=int, default=0, help="Identificador del marcador.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolucion de impresion.")
    parser.add_argument(
        "--salida", default="reports/marcador_escala.png", help="Ruta relativa del PNG."
    )
    return parser


def milimetros_a_pixeles(milimetros: float, dpi: int) -> int:
    """Convierte una medida fisica al numero de pixeles que la representa.

    Args:
        milimetros: Medida en milimetros.
        dpi: Puntos por pulgada de la impresion.

    Returns:
        Numero de pixeles, redondeado.
    """
    return int(round(milimetros / MM_POR_PULGADA * dpi))


def dibujar_regla(lienzo: np.ndarray, x: int, y: int, largo_mm: float, dpi: int) -> None:
    """Dibuja una regla de comprobacion de 50 mm con marcas cada 10 mm.

    Es la defensa contra el fallo silencioso de este metodo: si la impresora
    reescala la hoja, la regla deja de medir lo que dice y se detecta con un
    escalimetro en dos segundos. Sin ella, un marcador mal impreso produce
    medidas erroneas que nada delata.

    Args:
        lienzo: Imagen sobre la que dibujar.
        x: Coordenada horizontal de inicio.
        y: Coordenada vertical de la linea.
        largo_mm: Longitud de la regla en milimetros.
        dpi: Resolucion de impresion.
    """
    largo_px = milimetros_a_pixeles(largo_mm, dpi)
    grosor = max(1, dpi // 150)
    cv2.line(lienzo, (x, y), (x + largo_px, y), (0, 0, 0), grosor)

    for milimetro in range(0, int(largo_mm) + 1, 10):
        posicion = x + milimetros_a_pixeles(milimetro, dpi)
        alto = milimetros_a_pixeles(3.0 if milimetro % 50 else 5.0, dpi)
        cv2.line(lienzo, (posicion, y - alto), (posicion, y), (0, 0, 0), grosor)
        cv2.putText(
            lienzo,
            str(milimetro),
            (posicion - milimetros_a_pixeles(2.0, dpi), y + milimetros_a_pixeles(5.0, dpi)),
            cv2.FONT_HERSHEY_SIMPLEX,
            dpi / 900.0,
            (0, 0, 0),
            grosor,
            cv2.LINE_AA,
        )


def generar_hoja(lado_mm: float, identificador: int, dpi: int, diccionario: str) -> np.ndarray:
    """Compone la hoja completa: marcador, regla e instrucciones.

    Args:
        lado_mm: Lado del marcador en milimetros.
        identificador: Numero del marcador.
        dpi: Resolucion de impresion.
        diccionario: Nombre del diccionario ArUco.

    Returns:
        Imagen BGR de la hoja.

    Raises:
        ValueError: Si el diccionario no existe en esta version de OpenCV.
    """
    if not hasattr(cv2.aruco, diccionario):
        raise ValueError(f"El diccionario '{diccionario}' no existe en OpenCV {cv2.__version__}.")

    dicc = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    lado_px = milimetros_a_pixeles(lado_mm, dpi)
    marcador = cv2.aruco.generateImageMarker(dicc, identificador, lado_px)

    margen = milimetros_a_pixeles(15.0, dpi)
    alto = lado_px + margen * 2 + milimetros_a_pixeles(45.0, dpi)
    ancho = max(lado_px + margen * 2, milimetros_a_pixeles(90.0, dpi))
    hoja = np.full((alto, ancho, 3), 255, dtype=np.uint8)

    # El marcador necesita un borde blanco a su alrededor para que el detector
    # encuentre su contorno; recortarlo al ras lo haria indetectable.
    x0, y0 = (ancho - lado_px) // 2, margen
    hoja[y0 : y0 + lado_px, x0 : x0 + lado_px] = cv2.cvtColor(marcador, cv2.COLOR_GRAY2BGR)

    grosor = max(1, dpi // 150)
    escala_texto = dpi / 800.0
    y = y0 + lado_px + milimetros_a_pixeles(9.0, dpi)

    lineas = [
        f"Marcador #{identificador} - lado {lado_mm:.0f} mm - {diccionario}",
        "IMPRIMIR AL 100%. No usar 'ajustar a la pagina'.",
        "Comprueba con una regla que la escala de abajo mide 50 mm.",
        "Pega la hoja junto a la grieta, en su mismo plano.",
        "Si lo recortas, DEJA 1 cm DE BLANCO alrededor del cuadro.",
        "Un borde fino de 1 a 3 mm hace medir hasta un 13% de mas.",
    ]
    for texto in lineas:
        cv2.putText(
            hoja,
            texto,
            (margen, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_texto,
            (0, 0, 0),
            grosor,
            cv2.LINE_AA,
        )
        y += milimetros_a_pixeles(6.0, dpi)

    dibujar_regla(hoja, margen, y + milimetros_a_pixeles(6.0, dpi), 50.0, dpi)
    return hoja


def guardar_pdf(hoja: np.ndarray, dpi: int, destino: Path) -> Path:
    """Guarda la hoja como PDF de tamano carta con el marcador a escala real.

    Por que un PDF y no solo el PNG
    -------------------------------
    Una imagen no lleva dentro su tamano fisico: cada programa la imprime como le
    parece. La aplicacion Fotos de Windows, por ejemplo, **solo ofrece 'rellenar
    pagina' y 'ajustar a la pagina'**, y ambas reescalan — con lo que un marcador
    declarado de 50 mm puede acabar impreso a 140.

    Un PDF si lleva las medidas fisicas de cada elemento. Al abrirlo en cualquier
    lector y elegir 'Tamano real', el cuadrado sale con el lado que dice tener.

    Args:
        hoja: Imagen BGR de la hoja.
        dpi: Resolucion con la que se genero.
        destino: Ruta del PDF a escribir.

    Returns:
        La ruta del PDF escrito.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alto_px, ancho_px = hoja.shape[:2]
    alto_pulg, ancho_pulg = alto_px / dpi, ancho_px / dpi

    # Hoja carta, con la imagen centrada arriba y a su tamano exacto.
    carta = (8.5, 11.0)
    figura = plt.figure(figsize=carta, dpi=dpi)
    izquierda = (carta[0] - ancho_pulg) / 2.0 / carta[0]
    abajo = 1.0 - (0.6 + alto_pulg) / carta[1]
    ejes = figura.add_axes((izquierda, abajo, ancho_pulg / carta[0], alto_pulg / carta[1]))
    ejes.imshow(hoja[:, :, ::-1])  # BGR -> RGB
    ejes.axis("off")

    figura.savefig(destino, format="pdf")
    plt.close(figura)
    return destino


def main() -> None:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    config = cargar_config(args.config)

    lado_mm = float(args.lado_mm or obtener(config, "escala.lado_marcador_mm", 50.0))
    diccionario = str(obtener(config, "escala.diccionario", "DICT_4X4_50"))

    hoja = generar_hoja(lado_mm, args.id, args.dpi, diccionario)
    destino = resolver(args.salida)
    asegurar_directorio(destino.parent)
    cv2.imwrite(str(destino), hoja)

    pdf = guardar_pdf(hoja, args.dpi, destino.with_suffix(".pdf"))

    print(f"Marcador #{args.id} de {lado_mm:.0f} mm generado.")
    print(f"  PDF (recomendado para imprimir) -> {pdf}")
    print(f"  PNG (para ver en pantalla)      -> {destino}")
    print(
        f"  Tamano impreso: {hoja.shape[1] / args.dpi * MM_POR_PULGADA:.0f} x "
        f"{hoja.shape[0] / args.dpi * MM_POR_PULGADA:.0f} mm en una hoja carta"
    )
    print()
    print("COMO IMPRIMIRLO")
    print("  Abre el PDF y elige 'Tamano real' o 'Escala 100%'.")
    print("  NO uses 'Ajustar a la pagina' ni 'Rellenar pagina': reescalan la hoja.")
    print("  La aplicacion Fotos de Windows NO sirve: solo ofrece esas dos opciones.")
    print()
    print("DESPUES DE IMPRIMIR, comprueba la regla de la hoja con un escalimetro.")
    print(f"  Si marca 50 mm exactos, ya esta: config.yaml declara {lado_mm:.0f} mm.")
    print("  Si marca otra cosa, NO hace falta reimprimir. Mide el LADO NEGRO del")
    print("  cuadrado y escribe ese valor real en config.yaml:")
    print("      escala:")
    print("        lado_marcador_mm: <lo que mida de verdad>")
    print()
    print("Lo que importa no es imprimir a 50 mm exactos, sino que config.yaml diga")
    print("la verdad sobre el marcador que tienes en la mano.")


if __name__ == "__main__":
    main()
