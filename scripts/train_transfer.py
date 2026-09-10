"""Entrena MobileNetV2 por transfer learning, en dos etapas.

Etapa 1 - **extraccion de caracteristicas**: la base preentrenada en ImageNet
queda congelada y solo se entrena la cabeza densa. Es rapido y establece un
suelo de desempeno.

Etapa 2 - **fine-tuning**: se descongelan los ultimos bloques de la base y se
reentrena con una tasa de aprendizaje ~100 veces menor. El LR bajo no es un
detalle: con el LR de la etapa 1, los gradientes iniciales de una cabeza recien
entrenada destruirian los pesos de ImageNet en las primeras iteraciones.

El script guarda **ambos** modelos y **ambas** metricas, porque la comparacion
congelado contra fine-tuned es parte de lo que hay que reportar.

Uso tipico:
    python scripts/train_transfer.py --config config.yaml --subset 0.1 --epochs 3
    python scripts/train_transfer.py --config config.yaml
    python scripts/train_transfer.py --config config.yaml --sin-fine-tuning
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
from src.models.baseline import compilar  # noqa: E402
from src.models.entrenamiento import (  # noqa: E402
    construir_callbacks,
    fusionar_historiales,
    guardar_modelo_final,
    resumen_entrenamiento,
)
from src.models.transfer import construir_transfer, descongelar_ultimas_capas  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.dispositivo import detectar_dispositivo  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Entrena MobileNetV2 por transfer learning en dos etapas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al config YAML.")
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Fraccion del dataset en (0, 1] para validar el pipeline en minutos.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Epocas de la etapa 1 (base congelada)."
    )
    parser.add_argument(
        "--epochs-ft", type=int, default=None, help="Epocas de la etapa 2 (fine-tuning)."
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Tamano de lote.")
    parser.add_argument("--lr", type=float, default=None, help="LR de la etapa 1.")
    parser.add_argument("--lr-ft", type=float, default=None, help="LR de la etapa 2 (fine-tuning).")
    parser.add_argument(
        "--sin-fine-tuning",
        action="store_true",
        help="Ejecuta solo la etapa 1. Util para medir cuanto aporta el fine-tuning.",
    )
    parser.add_argument(
        "--sin-pesos-clase", action="store_true", help="Desactiva el balanceo por pesos de clase."
    )
    parser.add_argument(
        "--nombre",
        type=str,
        default=None,
        help=(
            "Nombre del experimento. Determina los nombres de TODOS los artefactos "
            "(modelo, checkpoints, historial, figuras), asi que darle uno distinto es "
            "la forma de probar una variante sin sobrescribir el modelo bueno."
        ),
    )
    parser.add_argument(
        "--degradacion-escala",
        type=float,
        default=None,
        metavar="FACTOR",
        help=(
            "Activa la degradacion de escala con el factor maximo indicado (ver §4.6). "
            "Con 1.0 o menos queda desactivada. Anula lo que diga el YAML."
        ),
    )
    return parser


def main() -> int:
    """Punto de entrada del entrenamiento por transfer learning.

    Returns:
        Codigo de salida del proceso: 0 si termino correctamente.
    """
    args = construir_parser().parse_args()
    config = cargar_config(args.config)

    semilla = int(obtener(config, "proyecto.semilla", 42))
    fijar_semillas(semilla)
    info_dispositivo = detectar_dispositivo()

    nombre = str(args.nombre or obtener(config, "transfer.nombre", "mobilenetv2"))

    # La degradacion de escala se inyecta en la configuracion ya cargada, no se
    # pasa como parametro suelto: asi viaja sola hasta crear_dataset y queda
    # registrada en el resumen del experimento junto al resto de hiperparametros.
    if args.degradacion_escala is not None:
        aumento = config.setdefault("preproceso", {}).setdefault("aumento", {})
        aumento["degradacion_escala"] = {
            "activo": args.degradacion_escala > 1.0,
            "factor_maximo": float(args.degradacion_escala),
            "probabilidad": float(
                obtener(config, "preproceso.aumento.degradacion_escala.probabilidad", 0.5)
            ),
        }

    degradacion = obtener(config, "preproceso.aumento.degradacion_escala", {}) or {}
    if degradacion.get("activo"):
        print(
            f"Degradacion de escala ACTIVA: factor hasta {degradacion.get('factor_maximo')}x "
            f"en el {float(degradacion.get('probabilidad', 0.5)):.0%} de los lotes."
        )
    if nombre != str(obtener(config, "transfer.nombre", "mobilenetv2")):
        print(f"Experimento '{nombre}': los artefactos NO sobrescriben los del modelo por defecto.")
    epocas_1 = int(args.epochs or obtener(config, "entrenamiento.epocas", 20))
    epocas_2 = int(args.epochs_ft or obtener(config, "transfer.fine_tuning.epocas", 10))
    batch = args.batch_size or obtener(config, "entrenamiento.batch_size", 32)
    lr_1 = float(args.lr or obtener(config, "entrenamiento.tasa_aprendizaje", 1e-3))
    lr_2 = float(args.lr_ft or obtener(config, "transfer.fine_tuning.tasa_aprendizaje", 1e-5))
    hacer_ft = not args.sin_fine_tuning and bool(
        obtener(config, "transfer.fine_tuning.activo", True)
    )

    print("\n[1/6] Cargando datos")
    datasets, inventarios = cargar_particiones(config, subset=args.subset, batch_size=batch)

    # train y val se repiten indefinidamente: Keras necesita el numero de pasos.
    pasos_train = pasos_por_epoca(inventarios["train"], batch)
    pasos_val = pasos_por_epoca(inventarios["val"], batch)

    pesos = None
    if not args.sin_pesos_clase and obtener(config, "entrenamiento.pesos_clase") == "auto":
        pesos = calcular_pesos_clase(inventarios["train"])
        print(f"  Pesos de clase (balanceo): {pesos}")

    print("\n[2/6] Construyendo MobileNetV2 (base congelada)")
    modelo, base = construir_transfer(config, base_entrenable=False)
    modelo = compilar(modelo, lr_1)
    modelo.summary()

    params_congelado = contar_parametros(modelo)
    print(
        f"  Parametros: {params_congelado['total']:,} "
        f"(entrenables {params_congelado['entrenables']:,}, "
        f"congelados {params_congelado['no_entrenables']:,})"
    )
    print(
        f"  Solo el {100 * params_congelado['entrenables'] / max(params_congelado['total'], 1):.1f}% "
        "de los parametros se entrena en la etapa 1: por eso es tan rapida en CPU."
    )

    print(f"\n[3/6] Etapa 1 - entrenando la cabeza ({epocas_1} epocas)")
    callbacks_1, medidor_1 = construir_callbacks(config, f"{nombre}_congelado", epocas_1)
    historia_1 = modelo.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=epocas_1,
        steps_per_epoch=pasos_train,
        validation_steps=pasos_val,
        callbacks=callbacks_1,
        class_weight=pesos,
        verbose=1,
    )

    ruta_congelado = guardar_modelo_final(modelo, config, f"{nombre}_congelado.keras")
    print(f"  Modelo etapa 1 -> {ruta_congelado}")

    y_true_v, y_prob_v = met.predecir(modelo, datasets["val"].take(pasos_val))
    umbral = float(obtener(config, "evaluacion.umbral", 0.5))
    metricas_congelado = met.calcular_metricas(y_true_v, y_prob_v, umbral)
    print(
        f"  Validacion (congelado): accuracy={metricas_congelado['accuracy']:.4f}  "
        f"recall={metricas_congelado['recall']:.4f}  f1={metricas_congelado['f1']:.4f}"
    )

    historial_total = historia_1.history
    metricas_ft = None
    medidor_2 = None
    capas_descongeladas = 0
    ruta_final = ruta_congelado

    if hacer_ft:
        n_capas = int(obtener(config, "transfer.fine_tuning.capas_a_descongelar", 30))
        capas_descongeladas = descongelar_ultimas_capas(base, n_capas)
        # Recompilar es obligatorio: cambiar 'trainable' no surte efecto hasta que
        # se reconstruye el grafo de entrenamiento con el nuevo optimizador.
        modelo = compilar(modelo, lr_2)
        params_ft = contar_parametros(modelo)
        print(
            f"\n[4/6] Etapa 2 - fine-tuning ({epocas_2} epocas, LR={lr_2:g})\n"
            f"  Capas descongeladas: {capas_descongeladas} de {len(base.layers)} "
            f"(BatchNormalization se mantiene congelada a proposito)\n"
            f"  Parametros entrenables: {params_ft['entrenables']:,}"
        )

        callbacks_2, medidor_2 = construir_callbacks(
            config, f"{nombre}_finetuned", epocas_1 + epocas_2
        )
        historia_2 = modelo.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=epocas_1 + epocas_2,
            initial_epoch=epocas_1,
            steps_per_epoch=pasos_train,
            validation_steps=pasos_val,
            callbacks=callbacks_2,
            class_weight=pesos,
            verbose=1,
        )
        historial_total = fusionar_historiales(historia_1.history, historia_2.history)

        ruta_final = guardar_modelo_final(modelo, config, f"{nombre}_finetuned.keras")
        print(f"  Modelo etapa 2 -> {ruta_final}")

        y_true_v, y_prob_v = met.predecir(modelo, datasets["val"].take(pasos_val))
        metricas_ft = met.calcular_metricas(y_true_v, y_prob_v, umbral)
        print(
            f"  Validacion (fine-tuned): accuracy={metricas_ft['accuracy']:.4f}  "
            f"recall={metricas_ft['recall']:.4f}  f1={metricas_ft['f1']:.4f}"
        )
        delta = metricas_ft["f1"] - metricas_congelado["f1"]
        print(f"  Aporte del fine-tuning en F1: {delta:+.4f}")
    else:
        print("\n[4/6] Fine-tuning desactivado (--sin-fine-tuning).")

    print("\n[5/6] Guardando artefactos")
    ruta_hist = met.guardar_historial(historial_total, f"reports/metricas/historial_{nombre}.csv")
    print(f"  Historial -> {ruta_hist}")

    titulo = f"Curvas de entrenamiento - MobileNetV2 (etapa 1 hasta epoca {epocas_1})"
    ruta_fig = met.figura_curvas_entrenamiento(
        historial_total, titulo, f"reports/figuras/curvas_{nombre}.png"
    )
    print(f"  Figura    -> {ruta_fig}")

    tiempos_1 = medidor_1.resumen()
    tiempos_2 = medidor_2.resumen() if medidor_2 else None
    resumen = resumen_entrenamiento(
        nombre=nombre,
        historial=historial_total,
        tiempos=tiempos_1,
        info_dispositivo=info_dispositivo,
        parametros=contar_parametros(modelo),
        extra={
            "arquitectura_base": obtener(config, "transfer.arquitectura_base", "MobileNetV2"),
            "alpha": obtener(config, "transfer.alpha", 0.75),
            "nota_arquitectura": (
                "EfficientNet-Lite no esta disponible en keras.applications; se uso "
                "MobileNetV2, estandar de facto para inferencia movil."
            ),
            "subset": args.subset,
            "batch_size": batch,
            "lr_etapa1": lr_1,
            "lr_etapa2": lr_2 if hacer_ft else None,
            "epocas_etapa1": epocas_1,
            "epocas_etapa2": epocas_2 if hacer_ft else 0,
            "capas_descongeladas": capas_descongeladas,
            "tiempos_etapa1": tiempos_1,
            "tiempos_etapa2": tiempos_2,
            "metricas_val_congelado": metricas_congelado,
            "metricas_val_finetuned": metricas_ft,
            "pesos_clase": pesos,
            "n_train": len(inventarios["train"]),
            "n_val": len(inventarios["val"]),
            "n_test": len(inventarios["test"]),
            "modelo_final": str(ruta_final.name),
        },
    )
    ruta_resumen = met.guardar_json(resumen, f"reports/metricas/entrenamiento_{nombre}.json")
    print(f"  Resumen   -> {ruta_resumen}")

    print("\n[6/6] Costo computacional")
    total_min = tiempos_1["segundos_totales"] / 60
    if tiempos_2:
        total_min += tiempos_2["segundos_totales"] / 60
    print(f"  Etapa 1: {tiempos_1['segundos_por_epoca_media']:.1f} s/epoca")
    if tiempos_2:
        print(f"  Etapa 2: {tiempos_2['segundos_por_epoca_media']:.1f} s/epoca")
    print(f"  Total   : {total_min:.1f} min en {info_dispositivo['dispositivo']}\n")

    print("Listo. Siguiente paso:")
    print("  python scripts/export_tflite.py --config config.yaml")
    print("  python scripts/evaluate.py --config config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
