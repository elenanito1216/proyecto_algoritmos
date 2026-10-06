"""Valida el estimador de inclinacion rotando fotos a plomo angulos conocidos.

El problema: no hay ground truth. Nadie midio con inclinometro el desaplome real
de las fotos del dataset, asi que no se puede reportar un error absoluto honesto.

La solucion: **validacion por transformacion conocida**. Se toma una fotografia
de un elemento razonablemente vertical, se rota un angulo exacto y se comprueba
que la estimacion se desplaza justo ese angulo. Mide la fidelidad del metodo sin
necesitar instrumentacion, y es el tipo de verificacion que distingue un
proyecto que mide de uno que solo dibuja lineas bonitas.

Uso tipico:
    python scripts/validar_inclinacion.py --config config.yaml --imagenes data/propias
    python scripts/validar_inclinacion.py --config config.yaml --imagenes data/propias/columna.jpg
    python scripts/validar_inclinacion.py --config config.yaml --angulos 1 2 5 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval import metricas as met  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import asegurar_directorio, resolver  # noqa: E402
from src.vision.inclinacion import (  # noqa: E402
    anotar_imagen,
    clasificar_orientacion_grieta,
    estimar_inclinacion,
    validar_con_rotaciones,
)


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Valida el estimador de angulo mediante rotaciones controladas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al config YAML.")
    parser.add_argument(
        "--imagenes",
        type=str,
        default="data/propias",
        help="Archivo de imagen o directorio con fotos de elementos verticales.",
    )
    parser.add_argument(
        "--angulos",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0],
        help="Angulos de rotacion a evaluar (se prueban tambien sus opuestos).",
    )
    parser.add_argument(
        "--max-imagenes",
        type=int,
        default=200,
        help=(
            "Limite de imagenes a procesar. El valor debe cubrir el directorio entero: "
            "las rutas se recorren en orden alfabetico, asi que un limite bajo procesa "
            "solo las primeras subcarpetas (p. ej. 'Negative/') y puede dar la falsa "
            "impresion de que ninguna foto sirve."
        ),
    )
    parser.add_argument(
        "--guardar-anotadas",
        action="store_true",
        help="Guarda en reports/figuras/ la imagen anotada de cada foto valida.",
    )
    return parser


def _listar_imagenes(ruta: Path, extensiones: list[str]) -> list[Path]:
    """Recolecta todas las imagenes candidatas, en orden alfabetico.

    No aplica ningun limite: el recorte lo hace quien la llama, para poder
    avisar de cuantas imagenes se dejan fuera. Truncar aqui en silencio hacia
    que un limite bajo pareciera "ninguna foto sirve" cuando en realidad el
    script no habia llegado a mirarlas.

    Args:
        ruta: Archivo o directorio.
        extensiones: Extensiones aceptadas.

    Returns:
        Lista de rutas de imagen, ordenada alfabeticamente.
    """
    if ruta.is_file():
        return [ruta]
    if not ruta.is_dir():
        return []
    validas = {e.lower() for e in extensiones}
    return sorted(p for p in ruta.rglob("*") if p.is_file() and p.suffix.lower() in validas)


def main() -> int:
    """Punto de entrada de la validacion de inclinometria.

    Returns:
        Codigo de salida: 0 si al menos una imagen se pudo validar, 1 si ninguna.
    """
    args = construir_parser().parse_args()
    config = cargar_config(args.config)

    ruta = resolver(args.imagenes)
    extensiones = obtener(config, "datos.extensiones", [".jpg", ".png"])
    encontradas = _listar_imagenes(ruta, extensiones)
    imagenes = encontradas[: args.max_imagenes]

    if len(encontradas) > len(imagenes):
        print(
            f"  AVISO: hay {len(encontradas)} imagenes en {ruta.name}/ y solo se "
            f"procesaran las {len(imagenes)} primeras en orden alfabetico.\n"
            f"         Usa --max-imagenes {len(encontradas)} para incluirlas todas."
        )

    if not imagenes:
        print(
            f"No se encontraron imagenes en {ruta}.\n"
            "Coloca al menos una foto de una columna o muro razonablemente vertical\n"
            "en data/propias/ y vuelve a ejecutar."
        )
        return 1

    print("=" * 72)
    print("  VALIDACION DEL ESTIMADOR DE INCLINACION POR ROTACION CONTROLADA")
    print("=" * 72)
    print(f"  Imagenes         : {len(imagenes)}")
    print(f"  Angulos de prueba: +-{', +-'.join(f'{a:g}' for a in args.angulos)} grados")
    print(
        "  Criterio         : al rotar la imagen alpha grados en sentido antihorario,\n"
        "                     la desviacion estimada debe bajar exactamente alpha.\n"
    )

    informes: list[dict[str, Any]] = []
    filas_tabla: list[dict[str, Any]] = []
    errores_globales: list[float] = []

    for imagen_path in imagenes:
        imagen = cv2.imread(str(imagen_path))
        if imagen is None:
            print(f"  [saltada] {imagen_path.name}: no se pudo leer el archivo.")
            continue

        base = estimar_inclinacion(imagen, config)
        if not base.fiable:
            print(f"  [saltada] {imagen_path.name}: {base.mensaje}")
            continue

        informe = validar_con_rotaciones(imagen, config, args.angulos)
        informe["archivo"] = imagen_path.name

        orientacion = clasificar_orientacion_grieta(imagen, config)
        informe["orientacion_grieta"] = {
            "orientacion": orientacion.orientacion,
            "angulo_grados": orientacion.angulo_grados,
            "confianza": orientacion.confianza,
        }
        informes.append(informe)

        print(f"\n  {imagen_path.name}")
        print(
            f"    referencia sin rotar: {informe['referencia_grados']:+.2f} grados "
            f"({informe['confianza_referencia']} lineas)"
        )
        print(f"    {'rotacion':>10} {'esperado':>10} {'estimado':>10} {'error':>8}  fiable")
        for caso in informe["casos"]:
            estimado = caso["angulo_estimado"]
            error = caso["error_absoluto"]
            texto_est = f"{estimado:+.2f}" if estimado is not None else "  n/d"
            texto_err = f"{error:.2f}" if error is not None else " n/d"
            print(
                f"    {caso['rotacion_aplicada']:>+9.1f} {caso['angulo_esperado']:>+10.2f} "
                f"{texto_est:>10} {texto_err:>8}  {'si' if caso['fiable'] else 'NO'}"
            )
            filas_tabla.append(
                {
                    "archivo": imagen_path.name,
                    "rotacion_aplicada": caso["rotacion_aplicada"],
                    "angulo_esperado": caso["angulo_esperado"],
                    "angulo_estimado": estimado,
                    "error_absoluto": error,
                    "fiable": caso["fiable"],
                }
            )

        if informe["error_medio_grados"] is not None:
            errores_globales.append(informe["error_medio_grados"])
            print(
                f"    error medio: {informe['error_medio_grados']:.3f} grados  |  "
                f"maximo: {informe['error_maximo_grados']:.3f} grados  "
                f"({informe['n_casos_fiables']}/{informe['n_casos']} casos fiables)"
            )

        if args.guardar_anotadas:
            destino = asegurar_directorio("reports/figuras") / f"inclinacion_{imagen_path.stem}.png"
            cv2.imwrite(str(destino), anotar_imagen(imagen, base))
            print(f"    anotada -> {destino}")

    if not informes:
        print(
            "\nNinguna imagen produjo una estimacion fiable. Usa fotos donde el borde\n"
            "vertical del elemento sea largo, nitido y contrastado contra el fondo."
        )
        return 1

    error_medio = float(np.mean(errores_globales)) if errores_globales else None
    print("\n" + "=" * 72)
    print(f"  RESUMEN: {len(informes)} imagen(es) validada(s)")
    if error_medio is not None:
        print(f"  Error medio global: {error_medio:.3f} grados")
        print(
            "  Interpretacion: por debajo de ~0.5 grados el metodo distingue con\n"
            "  holgura el umbral de desaplome severo (2.0 grados) del de atencion (1.0)."
        )
    print("=" * 72)

    resumen = {
        "n_imagenes": len(informes),
        "angulos_prueba": args.angulos,
        "error_medio_global_grados": error_medio,
        "informes": informes,
    }
    ruta_json = met.guardar_json(resumen, "reports/metricas/validacion_inclinacion.json")
    ruta_csv = asegurar_directorio("reports/metricas") / "validacion_inclinacion.csv"
    pd.DataFrame(filas_tabla).to_csv(ruta_csv, index=False, encoding="utf-8")

    print(f"\n  JSON -> {ruta_json}")
    print(f"  CSV  -> {ruta_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
