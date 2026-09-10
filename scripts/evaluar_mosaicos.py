"""Compara inferencia sobre la imagen entera frente a inferencia por mosaicos.

La hipotesis que pone a prueba
------------------------------
El informe (§6) atribuia la brecha entre el conjunto de prueba (F1 0.9423) y las
fotografias propias (F1 0.6667 con MobileNetV2, 0.8235 con el ensemble) a un
desplazamiento de dominio: otro material, otra iluminacion, otra camara.

Hay una explicacion mas simple y mas incomoda, porque el fallo seria nuestro:
**la escala**. Los parches de entrenamiento miden 227x227 px de mediana y se
reducen a 160 (factor 1.4x). Las fotos propias miden 1200x1600 y se reducen a
160 (factor 7.5x). Una grieta de 3 px de ancho queda en 2.1 px en el primer caso
y en 0.4 px en el segundo: en el segundo la hemos borrado nosotros, con el
redimensionado, antes de que el modelo la viera.

Si la hipotesis es cierta, trocear la fotografia en ventanas del tamano con el
que el modelo aprendio debe recuperar recall **sin reentrenar nada**. Si es
falsa, el troceado no cambiara los resultados y la culpa sera del dominio.

Que mide
--------
Para cada estrategia (imagen entera y mosaicos de varios lados) reporta el
cuadro de metricas al umbral 0.5, el desglose de errores y el coste por foto.
El coste importa tanto como el acierto: la memoria pide analisis de eficiencia,
y trocear multiplica el numero de inferencias.

Uso tipico:
    python scripts/evaluar_mosaicos.py --config config.yaml
    python scripts/evaluar_mosaicos.py --config config.yaml --lados 227 320 480
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.loader import construir_inventario  # noqa: E402
from src.eval import metricas as met  # noqa: E402
from src.models.inferencia import cargar_predictor, predecir_por_mosaicos  # noqa: E402
from src.utils.config import cargar_config, forma_entrada, obtener  # noqa: E402
from src.utils.rutas import resolver  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(description="Compara imagen entera frente a mosaicos.")
    parser.add_argument("--config", default="config.yaml", help="Ruta del YAML de configuracion.")
    parser.add_argument(
        "--formato",
        default="ensemble",
        choices=["ensemble", "keras", "tflite"],
        help="Predictor a evaluar.",
    )
    parser.add_argument(
        "--lados",
        type=int,
        nargs="+",
        default=[227, 320, 480],
        help="Lados de mosaico a comparar, en pixeles.",
    )
    parser.add_argument(
        "--modelo",
        default=None,
        help=(
            "Ruta de un .keras concreto, para evaluar una variante experimental sin "
            "tocar config.yaml. Solo tiene efecto con --formato keras."
        ),
    )
    parser.add_argument("--solape", type=float, default=0.5, help="Solape entre ventanas.")
    parser.add_argument("--umbral", type=float, default=0.5, help="Umbral de decision.")
    parser.add_argument(
        "--maximo-mosaicos", type=int, default=64, help="Cota de ventanas por imagen."
    )
    parser.add_argument(
        "--salida",
        default="reports/mosaicos.json",
        help="Ruta relativa del JSON de resultados.",
    )
    return parser


def cargar_imagen(ruta: Path) -> np.ndarray:
    """Lee una imagen de disco en RGB.

    Args:
        ruta: Ruta del archivo.

    Returns:
        Array ``(H, W, 3)`` de enteros sin signo.

    Raises:
        OSError: Si el archivo no se puede decodificar.
    """
    import cv2

    bgr = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"No se pudo leer la imagen: {ruta}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resumir_tamanos(rutas: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    """Cuantifica el factor de reduccion que sufre cada imagen al entrar al modelo.

    Es la medicion que motiva todo el script: sin ella, "las fotos son distintas"
    seria una intuicion y no un diagnostico.

    Args:
        rutas: Rutas de las imagenes a medir.
        config: Configuracion del proyecto.

    Returns:
        Diccionario con las medianas de alto, ancho y factor de reduccion.
    """
    alto_modelo, ancho_modelo, _ = forma_entrada(config)
    dimensiones = []
    for ruta in rutas:
        try:
            imagen = cargar_imagen(ruta)
        except OSError:
            continue
        dimensiones.append(imagen.shape[:2])

    if not dimensiones:
        return {"n": 0}

    altos = np.array([d[0] for d in dimensiones])
    anchos = np.array([d[1] for d in dimensiones])
    factores = np.minimum(altos / alto_modelo, anchos / ancho_modelo)
    return {
        "n": len(dimensiones),
        "alto_mediano": int(np.median(altos)),
        "ancho_mediano": int(np.median(anchos)),
        "factor_reduccion_mediano": round(float(np.median(factores)), 2),
    }


def evaluar_estrategia(
    predictor: Any,
    rutas: list[Path],
    etiquetas: np.ndarray,
    config: dict[str, Any],
    lado: int | None,
    solape: float,
    umbral: float,
    maximo_mosaicos: int,
) -> dict[str, Any]:
    """Evalua una estrategia de inferencia sobre todas las imagenes.

    Args:
        predictor: Predictor ya cargado.
        rutas: Rutas de las imagenes.
        etiquetas: Etiquetas verdaderas ``(N,)``.
        config: Configuracion del proyecto.
        lado: Lado del mosaico, o ``None`` para inferir sobre la imagen entera.
        solape: Solape entre ventanas.
        umbral: Umbral de decision.
        maximo_mosaicos: Cota de ventanas por imagen.

    Returns:
        Diccionario con metricas, coste y la lista de errores cometidos.
    """
    probabilidades: list[float] = []
    tiempos: list[float] = []
    ventanas_por_foto: list[int] = []
    # El lado pedido no siempre es el aplicado: si una foto genera mas ventanas
    # que la cota, el mosaico se agranda. Reportarlo evita comparar dos filas de
    # la tabla que en realidad usaron el mismo lado.
    lados_efectivos: list[int] = []

    for ruta in rutas:
        imagen = cargar_imagen(ruta)
        inicio = time.perf_counter()
        if lado is None:
            probabilidad, _ = predictor.predecir_imagen(imagen, config)
            n_ventanas = 1
        else:
            resultado = predecir_por_mosaicos(
                predictor,
                imagen,
                config,
                lado=lado,
                solape=solape,
                agregacion="maximo",
                maximo_mosaicos=maximo_mosaicos,
            )
            probabilidad = resultado.probabilidad
            n_ventanas = len(resultado.ventanas)
            lados_efectivos.append(resultado.lado)
        tiempos.append(time.perf_counter() - inicio)
        probabilidades.append(float(probabilidad))
        ventanas_por_foto.append(n_ventanas)

    y_prob = np.array(probabilidades, dtype=np.float32)
    resumen = met.calcular_metricas(etiquetas, y_prob, umbral=umbral)

    y_pred = (y_prob >= umbral).astype(int)
    fallos = [
        {
            "archivo": ruta.name,
            "verdad": int(verdad),
            "probabilidad": round(float(prob), 4),
            "tipo": "falso_negativo" if verdad == 1 else "falso_positivo",
        }
        for ruta, verdad, prediccion, prob in zip(rutas, etiquetas, y_pred, y_prob, strict=True)
        if verdad != prediccion
    ]

    return {
        "estrategia": "imagen_entera" if lado is None else f"mosaicos_{lado}px",
        "lado": lado,
        "lado_efectivo": int(np.median(lados_efectivos)) if lados_efectivos else None,
        "metricas": resumen,
        "segundos_por_foto": round(float(np.median(tiempos)), 3),
        "segundos_total": round(float(np.sum(tiempos)), 1),
        "ventanas_medianas": int(np.median(ventanas_por_foto)),
        "errores": fallos,
    }


def imprimir_tabla(resultados: list[dict[str, Any]]) -> None:
    """Muestra la comparativa en una tabla legible en terminal.

    Args:
        resultados: Salidas de :func:`evaluar_estrategia`.
    """
    cabecera = (
        f"{'estrategia':<24}{'F1':>8}{'recall':>9}{'prec':>8}"
        f"{'FN':>5}{'FP':>5}{'lado_ef':>9}{'ventanas':>10}{'s/foto':>9}"
    )
    print("\n" + cabecera)
    print("-" * len(cabecera))
    for fila in resultados:
        metricas = fila["metricas"]
        confusion = metricas["matriz_confusion"]
        print(
            f"{fila['estrategia']:<24}{metricas['f1']:>8.4f}{metricas['recall']:>9.2f}"
            f"{metricas['precision']:>8.3f}{confusion['fn']:>5}{confusion['fp']:>5}"
            f"{str(fila['lado_efectivo'] or '-'):>9}"
            f"{fila['ventanas_medianas']:>10}{fila['segundos_por_foto']:>9.2f}"
        )


def main() -> None:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    config = cargar_config(args.config)
    fijar_semillas(int(obtener(config, "proyecto.semilla", 42)))

    directorio = resolver(obtener(config, "rutas.datos_propias", "data/propias"))
    if not directorio.is_dir():
        raise SystemExit(
            f"No existe {directorio}. Este script mide la brecha de dominio sobre "
            "las fotografias propias del equipo; sin ellas no tiene nada que medir."
        )
    inventario = construir_inventario(directorio, config)
    rutas = list(inventario.rutas)
    etiquetas = np.asarray(inventario.etiquetas)

    print(f"Fotografias propias: {len(rutas)}  ({inventario.conteo_por_clase()})")
    tamanos = resumir_tamanos(rutas, config)
    print(
        f"Tamano mediano: {tamanos['alto_mediano']}x{tamanos['ancho_mediano']} px  "
        f"-> factor de reduccion {tamanos['factor_reduccion_mediano']}x"
    )

    predictor = cargar_predictor(config, args.formato, args.modelo)
    print(f"Predictor: {args.formato}\n")

    # Calentamiento: la primera inferencia de Keras incluye la construccion del
    # grafo y falsearia la medida de coste de la primera estrategia evaluada.
    predictor.predecir_imagen(cargar_imagen(rutas[0]), config)

    resultados = [
        evaluar_estrategia(
            predictor,
            rutas,
            etiquetas,
            config,
            None,
            args.solape,
            args.umbral,
            args.maximo_mosaicos,
        )
    ]
    for lado in args.lados:
        resultados.append(
            evaluar_estrategia(
                predictor,
                rutas,
                etiquetas,
                config,
                lado,
                args.solape,
                args.umbral,
                args.maximo_mosaicos,
            )
        )

    imprimir_tabla(resultados)

    mejor = max(resultados, key=lambda r: r["metricas"]["f1"])
    base = resultados[0]
    print(
        f"\nMejor estrategia: {mejor['estrategia']}  "
        f"F1 {base['metricas']['f1']:.4f} -> {mejor['metricas']['f1']:.4f}  "
        f"(recall {base['metricas']['recall']:.2f} -> {mejor['metricas']['recall']:.2f})"
    )
    if mejor["errores"]:
        print("\nErrores que quedan con la mejor estrategia:")
        for fallo in mejor["errores"]:
            print(f"  [{fallo['tipo']:<16}] p={fallo['probabilidad']:.4f}  {fallo['archivo']}")

    destino = met.guardar_json(
        {
            "formato": args.formato,
            "modelo": args.modelo,
            "umbral": args.umbral,
            "solape": args.solape,
            "tamanos": tamanos,
            "resultados": resultados,
        },
        args.salida,
    )
    print(f"\nResultados en {destino}")


if __name__ == "__main__":
    main()
