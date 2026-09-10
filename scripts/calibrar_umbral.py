"""Estima honestamente si mover la regla de decision mejora algo.

Por que hace falta un script aparte
-----------------------------------
Elegir el umbral que maximiza el F1 sobre las mismas 60 fotografias en las que
despues se reporta ese F1 es circular: mide lo bien que el umbral se ajusta a esa
muestra concreta, no lo bien que funcionara con la siguiente fotografia. La §3.8
ya cayo en esa trampa y tuvo que etiquetar su resultado como cota superior.

Aqui todo se mide con **validacion cruzada estratificada**: la regla se elige en
los pliegues de entrenamiento y se aplica al pliegue retenido, que no participo
en la eleccion. Las predicciones fuera de pliegue se acumulan y se miden juntas.
Es la diferencia entre "existe una regla que funciona en estas fotos" y "elegir
la regla asi funcionara en fotos nuevas". El oraculo se reporta igualmente, pero
etiquetado como lo que es: una cota superior inalcanzable, cuya distancia al
resultado cruzado mide el sobreajuste que se habria colado sin este cuidado.

Las dos familias de reglas
--------------------------
El analisis por mosaicos (§4.6) agrega las ventanas por **maximo**, y eso deja
dos formas distintas de decidir:

1. **Umbral sobre el maximo.** La habitual. Su problema es que el maximo de
   veinticuatro ventanas se satura: la mayoria de las fotos con grieta dan
   exactamente 1.0, y las sanas se acumulan cerca de 0.02. El umbral optimo
   acaba empujado a la zona saturada, donde separar bien depende de diferencias
   de diezmilesimas. Este script mide ese margen y lo reporta.
2. **Fraccion minima de ventanas.** Exigir que la grieta aparezca en al menos
   una parte de las ventanas, no en una sola. Es mas robusta porque cuenta
   evidencias en lugar de fiarse de un unico valor extremo, y ademas se puede
   explicar sin hablar de probabilidades: "la fisura debe verse en al menos una
   de cada diez ventanas".

Uso tipico:
    python scripts/calibrar_umbral.py --config config.yaml
    python scripts/calibrar_umbral.py --config config.yaml --formato ensemble
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.loader import construir_inventario  # noqa: E402
from src.eval import metricas as met  # noqa: E402
from src.models.inferencia import cargar_predictor, predecir_por_mosaicos  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import resolver  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402

from evaluar_mosaicos import cargar_imagen  # noqa: E402, isort: skip


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(description="Calibra la regla de decision con CV.")
    parser.add_argument("--config", default="config.yaml", help="Ruta del YAML de configuracion.")
    parser.add_argument(
        "--formato",
        default=None,
        help="Predictor a calibrar. Por defecto, el que la app usa para fotografias.",
    )
    parser.add_argument(
        "--modelo",
        default=None,
        help=(
            "Ruta de un .keras concreto, para evaluar una variante experimental sin "
            "tocar config.yaml. Solo tiene efecto con --formato keras."
        ),
    )
    parser.add_argument("--pliegues", type=int, default=5, help="Numero de pliegues.")
    parser.add_argument(
        "--umbral-ventana",
        type=float,
        default=0.5,
        help="Umbral con el que se decide si una ventana individual ve grieta.",
    )
    parser.add_argument(
        "--salida", default="reports/umbral.json", help="Ruta relativa del JSON de resultados."
    )
    return parser


def candidatos_intermedios(valores: np.ndarray) -> np.ndarray:
    """Devuelve los puntos de corte utiles de un vector de puntuaciones.

    Solo tiene sentido probar umbrales a medio camino entre dos valores
    observados consecutivos: cualquier otro produce exactamente la misma
    particion, asi que evaluarlo seria trabajo perdido.

    Args:
        valores: Puntuaciones observadas.

    Returns:
        Vector de umbrales candidatos. Puede estar vacio si no hay variacion.
    """
    unicos = np.unique(valores)
    if len(unicos) < 2:
        return np.array([0.5], dtype=np.float64)
    return (unicos[:-1] + unicos[1:]) / 2.0


def mejor_umbral(y_true: np.ndarray, puntuacion: np.ndarray) -> float:
    """Busca el umbral que maximiza el F1 sobre los datos dados.

    Ante empates se devuelve el umbral **mas alto**, y no es un detalle menor:
    con distribuciones bimodales como las que produce la agregacion por maximo,
    el F1 tiene dos optimos casi iguales —uno muy bajo que lo acepta casi todo y
    uno muy alto que rechaza las falsas alarmas—. Sin una regla de desempate, la
    eleccion salta entre ambos segun el pliegue y parece inestable cuando en
    realidad esta empatada.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        puntuacion: Puntuacion por muestra ``(N,)``.

    Returns:
        El umbral elegido.
    """
    from sklearn.metrics import f1_score

    candidatos = candidatos_intermedios(puntuacion)
    puntajes = np.array(
        [f1_score(y_true, (puntuacion >= u).astype(int), zero_division=0) for u in candidatos]
    )
    mejores = np.flatnonzero(puntajes >= puntajes.max() - 1e-12)
    return float(candidatos[mejores[-1]])


def evaluar(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calcula las metricas de una prediccion binaria.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        y_pred: Predicciones binarias ``(N,)``.

    Returns:
        Diccionario con ``f1``, ``recall``, ``precision``, ``fn`` y ``fp``.
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    _, fp, fn, _ = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "fn": fn,
        "fp": fp,
    }


def calibrar(
    y_true: np.ndarray, puntuacion: np.ndarray, pliegues: int, semilla: int
) -> dict[str, Any]:
    """Estima fuera de pliegue el desempeno de elegir el umbral por F1.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        puntuacion: Puntuacion por muestra ``(N,)``.
        pliegues: Numero de pliegues estratificados.
        semilla: Semilla del reparto.

    Returns:
        Metricas fuera de pliegue, umbrales elegidos y el resultado del oraculo.
    """
    from sklearn.model_selection import StratifiedKFold

    particion = StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=semilla)
    fuera_de_pliegue = np.zeros(len(y_true), dtype=int)
    umbrales: list[float] = []

    for entrena, prueba in particion.split(puntuacion.reshape(-1, 1), y_true):
        umbral = mejor_umbral(y_true[entrena], puntuacion[entrena])
        umbrales.append(umbral)
        fuera_de_pliegue[prueba] = (puntuacion[prueba] >= umbral).astype(int)

    umbral_oraculo = mejor_umbral(y_true, puntuacion)
    return {
        **evaluar(y_true, fuera_de_pliegue),
        "umbral_mediano": float(np.median(umbrales)),
        "umbrales_por_pliegue": [round(float(u), 6) for u in umbrales],
        "acuerdo_entre_pliegues": float(
            np.mean(np.abs(np.array(umbrales) - np.median(umbrales)) < 0.01)
        ),
        "oraculo": {"umbral": umbral_oraculo, **evaluar(y_true, (puntuacion >= umbral_oraculo))},
    }


def margen_de_separacion(
    y_true: np.ndarray, puntuacion: np.ndarray, umbral: float
) -> dict[str, float]:
    """Mide cuanto espacio libre deja un umbral a cada lado.

    Es la comprobacion que decide si una regla es fiable o solo afortunada. Un
    umbral que acierta en las sesenta fotografias pero pasa a diezmilesimas del
    peor negativo no ha aprendido nada generalizable: ha encontrado un hueco en
    esta muestra concreta.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        puntuacion: Puntuacion por muestra ``(N,)``.
        umbral: Umbral a examinar.

    Returns:
        Diccionario con la distancia al negativo mas alto y al positivo mas bajo
        que el umbral clasifica correctamente.
    """
    negativos = puntuacion[y_true == 0]
    positivos = puntuacion[y_true == 1]
    negativos_bajo = negativos[negativos < umbral]
    positivos_sobre = positivos[positivos >= umbral]
    return {
        "margen_inferior": (
            float(umbral - negativos_bajo.max()) if len(negativos_bajo) else float("nan")
        ),
        "margen_superior": (
            float(positivos_sobre.min() - umbral) if len(positivos_sobre) else float("nan")
        ),
    }


def main() -> None:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    config = cargar_config(args.config)
    semilla = int(obtener(config, "proyecto.semilla", 42))
    fijar_semillas(semilla)

    formato = args.formato or str(obtener(config, "app.modelos.foto", "keras"))
    directorio = resolver(obtener(config, "rutas.datos_propias", "data/propias"))
    if not directorio.is_dir():
        raise SystemExit(f"No existe {directorio}: no hay fotografias propias que calibrar.")

    inventario = construir_inventario(directorio, config)
    y_true = np.asarray(inventario.etiquetas)
    predictor = cargar_predictor(config, formato, args.modelo)
    print(f"{len(inventario.rutas)} fotografias · predictor {formato} · mosaicos\n")

    por_ventana = [
        predecir_por_mosaicos(predictor, cargar_imagen(ruta), config).probabilidades
        for ruta in inventario.rutas
    ]
    maximos = np.array([float(p.max()) for p in por_ventana], dtype=np.float64)
    fracciones = np.array(
        [float((p >= args.umbral_ventana).mean()) for p in por_ventana], dtype=np.float64
    )

    base = evaluar(y_true, (maximos >= 0.5).astype(int))
    cal_maximo = calibrar(y_true, maximos, args.pliegues, semilla)
    cal_fraccion = calibrar(y_true, fracciones, args.pliegues, semilla)

    filas = [
        ("umbral fijo 0.5 (actual)", base, 0.5),
        ("CV · umbral sobre el maximo", cal_maximo, cal_maximo["umbral_mediano"]),
        ("CV · fraccion de ventanas", cal_fraccion, cal_fraccion["umbral_mediano"]),
        ("ORACULO sobre el maximo", cal_maximo["oraculo"], cal_maximo["oraculo"]["umbral"]),
        ("ORACULO sobre la fraccion", cal_fraccion["oraculo"], cal_fraccion["oraculo"]["umbral"]),
    ]
    cabecera = f"{'regla':<30}{'F1':>8}{'recall':>9}{'prec':>8}{'FN':>5}{'FP':>5}{'corte':>10}"
    print(cabecera)
    print("-" * len(cabecera))
    for nombre, datos, corte in filas:
        print(
            f"{nombre:<30}{datos['f1']:>8.4f}{datos['recall']:>9.2f}{datos['precision']:>8.3f}"
            f"{datos['fn']:>5}{datos['fp']:>5}{corte:>10.4f}"
        )

    print("\nEstabilidad del corte entre pliegues")
    for nombre, calibrado in (("maximo", cal_maximo), ("fraccion", cal_fraccion)):
        print(
            f"  {nombre:<9} {calibrado['umbrales_por_pliegue']}  "
            f"acuerdo {calibrado['acuerdo_entre_pliegues']:.0%}"
        )

    print("\nMargen de separacion del corte del oraculo")
    for nombre, puntuacion, calibrado in (
        ("maximo", maximos, cal_maximo),
        ("fraccion", fracciones, cal_fraccion),
    ):
        margen = margen_de_separacion(y_true, puntuacion, calibrado["oraculo"]["umbral"])
        print(
            f"  {nombre:<9} al negativo mas alto {margen['margen_inferior']:.4f} · "
            f"al positivo mas bajo {margen['margen_superior']:.4f}"
        )
    print(
        "  Un margen inferior diminuto significa que el corte pasa rozando una foto sana\n"
        "  concreta de esta muestra: acierta aqui, pero no hay razon para esperar que\n"
        "  acierte en la siguiente."
    )

    destino = met.guardar_json(
        {
            "formato": formato,
            "modelo": args.modelo,
            "n": len(inventario.rutas),
            "pliegues": args.pliegues,
            "umbral_ventana": args.umbral_ventana,
            "umbral_fijo": base,
            "calibrado_maximo": cal_maximo,
            "calibrado_fraccion": cal_fraccion,
        },
        args.salida,
    )
    print(f"\nResultados en {destino}")


if __name__ == "__main__":
    main()
