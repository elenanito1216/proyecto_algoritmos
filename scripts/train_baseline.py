"""Entrena la CNN de linea base desde cero.

Es el punto de comparacion obligatorio del proyecto: sin el, no se puede
afirmar que el transfer learning aporte nada.

Uso tipico:
    # Ensayo rapido del pipeline completo con el 10% de los datos
    python scripts/train_baseline.py --config config.yaml --subset 0.1 --epochs 3

    # Corrida completa
    python scripts/train_baseline.py --config config.yaml

    # Reanudar una corrida interrumpida
    python scripts/train_baseline.py --config config.yaml --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import (  # noqa: E402
    calcular_pesos_clase,
    cargar_particiones,
    pasos_por_epoca,
)
from src.eval import metricas as met  # noqa: E402
from src.eval.complejidad import contar_parametros  # noqa: E402
from src.models.baseline import compilar, construir_baseline  # noqa: E402
from src.models.entrenamiento import (  # noqa: E402
    construir_callbacks,
    guardar_modelo_final,
    intentar_reanudar,
    resumen_entrenamiento,
)
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.dispositivo import detectar_dispositivo  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Entrena la CNN de linea base para deteccion de grietas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Ruta al archivo de configuracion."
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help=(
            "Fraccion del dataset a usar, en (0, 1]. Permite validar el pipeline "
            "completo en minutos antes de lanzar la corrida larga en CPU."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Sobrescribe entrenamiento.epocas."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Sobrescribe entrenamiento.batch_size."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Sobrescribe entrenamiento.tasa_aprendizaje."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda desde el ultimo checkpoint guardado en models/checkpoints/.",
    )
    parser.add_argument(
        "--sin-pesos-clase",
        action="store_true",
        help="Desactiva el balanceo por pesos de clase aunque este activo en el YAML.",
    )
    return parser


def main() -> int:
    """Punto de entrada del entrenamiento de la linea base.

    Returns:
        Codigo de salida del proceso: 0 si termino correctamente.
    """
    args = construir_parser().parse_args()
    config = cargar_config(args.config)

    semilla = int(obtener(config, "proyecto.semilla", 42))
    fijar_semillas(semilla)
    info_dispositivo = detectar_dispositivo()

    nombre = str(obtener(config, "baseline.nombre", "baseline_cnn"))
    epocas = int(args.epochs or obtener(config, "entrenamiento.epocas", 20))
    batch = args.batch_size or obtener(config, "entrenamiento.batch_size", 32)
    lr = float(args.lr or obtener(config, "entrenamiento.tasa_aprendizaje", 1e-3))

    print("\n[1/5] Cargando datos")
    datasets, inventarios = cargar_particiones(config, subset=args.subset, batch_size=batch)

    pasos_train = pasos_por_epoca(inventarios["train"], batch)
    pasos_val = pasos_por_epoca(inventarios["val"], batch)

    pesos = None
    if not args.sin_pesos_clase and obtener(config, "entrenamiento.pesos_clase") == "auto":
        pesos = calcular_pesos_clase(inventarios["train"])
        print(f"  Pesos de clase (balanceo): {pesos}")

    print("\n[2/5] Construyendo modelo")
    epoca_inicial = 0
    modelo = None
    if args.resume:
        modelo, epoca_inicial = intentar_reanudar(config, nombre)
    if modelo is None:
        modelo = compilar(construir_baseline(config), lr)
    modelo.summary()

    parametros = contar_parametros(modelo)
    print(
        f"  Parametros: {parametros['total']:,} "
        f"(entrenables {parametros['entrenables']:,}, "
        f"congelados {parametros['no_entrenables']:,})"
    )

    if epoca_inicial >= epocas:
        print(
            f"\n  El checkpoint ya alcanzo la epoca {epoca_inicial} de {epocas}. "
            "Aumenta --epochs para continuar entrenando."
        )
        return 0

    print(f"\n[3/5] Entrenando ({epocas - epoca_inicial} epocas por delante)")
    callbacks, medidor = construir_callbacks(config, nombre, epocas)
    historia = modelo.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=epocas,
        initial_epoch=epoca_inicial,
        steps_per_epoch=pasos_train,
        validation_steps=pasos_val,
        callbacks=callbacks,
        class_weight=pesos,
        verbose=1,
    )

    print("\n[4/5] Guardando artefactos")
    ruta_modelo = guardar_modelo_final(modelo, config, f"{nombre}.keras")
    print(f"  Modelo   -> {ruta_modelo}")

    ruta_hist = met.guardar_historial(historia.history, f"reports/metricas/historial_{nombre}.csv")
    print(f"  Historial-> {ruta_hist}")

    ruta_fig = met.figura_curvas_entrenamiento(
        historia.history,
        f"Curvas de entrenamiento - linea base ({nombre})",
        f"reports/figuras/curvas_{nombre}.png",
    )
    print(f"  Figura   -> {ruta_fig}")

    resumen = resumen_entrenamiento(
        nombre=nombre,
        historial=historia.history,
        tiempos=medidor.resumen(),
        info_dispositivo=info_dispositivo,
        parametros=parametros,
        extra={
            "subset": args.subset,
            "batch_size": batch,
            "tasa_aprendizaje": lr,
            "pesos_clase": pesos,
            "n_train": len(inventarios["train"]),
            "n_val": len(inventarios["val"]),
            "n_test": len(inventarios["test"]),
        },
    )
    ruta_resumen = met.guardar_json(resumen, f"reports/metricas/entrenamiento_{nombre}.json")
    print(f"  Resumen  -> {ruta_resumen}")

    print("\n[5/5] Evaluacion rapida sobre validacion")
    y_true, y_prob = met.predecir(modelo, datasets["val"].take(pasos_val))
    umbral = float(obtener(config, "evaluacion.umbral", 0.5))
    resultado = met.calcular_metricas(y_true, y_prob, umbral)
    print(
        f"  accuracy={resultado['accuracy']:.4f}  precision={resultado['precision']:.4f}  "
        f"recall={resultado['recall']:.4f}  f1={resultado['f1']:.4f}"
    )

    tiempos = medidor.resumen()
    print(
        f"\n  Costo: {tiempos['epocas_medidas']} epocas, "
        f"{tiempos['segundos_por_epoca_media']:.1f} s/epoca de media, "
        f"{tiempos['segundos_totales'] / 60:.1f} min en total.\n"
    )
    print("Listo. Siguiente paso: python scripts/train_transfer.py --config config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
