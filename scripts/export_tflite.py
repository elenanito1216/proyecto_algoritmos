"""Exporta un modelo ``.keras`` a TensorFlow Lite, en float32 y en int8.

Por que dos artefactos
----------------------
- **float32**: conversion directa. Sirve de control: cualquier diferencia de
  desempeno frente al ``.keras`` original se debe al conversor, no a la
  cuantizacion.
- **int8**: cuantizacion entera completa con dataset representativo. Reduce el
  tamano en disco ~4x y acelera la inferencia en CPU movil, a costa de una
  perdida de precision numerica que hay que medir, no suponer. Esa medicion es
  la que hace ``scripts/evaluate.py``.

Nota sobre Keras 3
------------------
Desde TensorFlow 2.16, Keras 3 es el backend por defecto y
``TFLiteConverter.from_keras_model`` no siempre acepta sus modelos. La ruta
robusta es exportar primero a SavedModel con ``model.export()`` y convertir
desde ahi. El script intenta esa ruta primero y solo cae a la directa si falla.

Uso tipico:
    python scripts/export_tflite.py --config config.yaml
    python scripts/export_tflite.py --config config.yaml --modelo models/baseline_cnn.keras
    python scripts/export_tflite.py --config config.yaml --io-int8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.loader import cargar_particiones  # noqa: E402
from src.eval import metricas as met  # noqa: E402
from src.eval.complejidad import tamano_en_disco_mb  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import asegurar_directorio, resolver  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Exporta un modelo Keras a TFLite (float32 e int8 cuantizado).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al config YAML.")
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        help="Ruta al .keras a exportar. Por defecto, app.archivo_keras del YAML.",
    )
    parser.add_argument(
        "--muestras-calibracion",
        type=int,
        default=200,
        help=(
            "Imagenes reales usadas para calibrar los rangos de cuantizacion. "
            "Con menos de ~100 la calibracion es ruidosa; mas de ~500 no aporta."
        ),
    )
    parser.add_argument(
        "--io-int8",
        action="store_true",
        help=(
            "Cuantiza tambien la entrada y la salida a int8 (full integer). "
            "Necesario para aceleradores que no aceptan tensores float; obliga a "
            "la aplicacion a cuantizar la imagen antes de inferir."
        ),
    )
    parser.add_argument("--solo-float32", action="store_true", help="Omite la variante cuantizada.")
    return parser


def _generador_representativo(dataset: Any, n_muestras: int) -> Any:
    """Construye el generador de calibracion que exige la cuantizacion entera.

    El conversor necesita ver datos **reales** para estimar el rango dinamico de
    cada tensor intermedio. Calibrar con ruido aleatorio produce escalas
    absurdas y destruye la exactitud del modelo cuantizado: es el error mas
    comun en este paso.

    Args:
        dataset: ``tf.data.Dataset`` de entrenamiento (imagenes ya normalizadas).
        n_muestras: Cuantas imagenes individuales entregar.

    Returns:
        Callable sin argumentos que devuelve un iterador de listas ``[imagen]``,
        que es la firma que espera ``representative_dataset``.
    """

    def generador() -> Iterator[list[np.ndarray]]:
        entregadas = 0
        for lote_x, _ in dataset:
            for imagen in lote_x:
                if entregadas >= n_muestras:
                    return
                yield [np.expand_dims(np.asarray(imagen, dtype=np.float32), axis=0)]
                entregadas += 1

    return generador


def _crear_conversor(modelo: Any, directorio_temporal: Path) -> Any:
    """Crea un ``TFLiteConverter`` a partir de un modelo Keras 3.

    Args:
        modelo: Modelo de Keras cargado.
        directorio_temporal: Carpeta donde exportar el SavedModel intermedio.

    Returns:
        Instancia de ``tf.lite.TFLiteConverter``.

    Raises:
        RuntimeError: Si ninguna de las dos rutas de conversion funciona.
    """
    import tensorflow as tf

    destino_sm = directorio_temporal / "saved_model"
    try:
        modelo.export(str(destino_sm))
        return tf.lite.TFLiteConverter.from_saved_model(str(destino_sm))
    except Exception as error_sm:  # noqa: BLE001
        print(f"  Aviso: export() a SavedModel fallo ({error_sm}). Probando from_keras_model.")
        try:
            return tf.lite.TFLiteConverter.from_keras_model(modelo)
        except Exception as error_keras:  # noqa: BLE001
            raise RuntimeError(
                "No se pudo crear el conversor TFLite ni desde SavedModel ni desde el "
                f"modelo Keras. SavedModel: {error_sm}. Keras: {error_keras}."
            ) from error_keras


def main() -> int:
    """Punto de entrada de la exportacion a TFLite.

    Returns:
        Codigo de salida del proceso: 0 si termino correctamente.
    """
    import tensorflow as tf

    args = construir_parser().parse_args()
    config = cargar_config(args.config)
    fijar_semillas(int(obtener(config, "proyecto.semilla", 42)))

    ruta_keras = resolver(args.modelo or obtener(config, "app.archivo_keras"))
    if not ruta_keras.is_file():
        print(
            f"ERROR: no existe el modelo {ruta_keras}.\n"
            "Entrena primero con: python scripts/train_transfer.py --config config.yaml"
        )
        return 1

    print(f"\n[1/4] Cargando {ruta_keras.name}")
    modelo = tf.keras.models.load_model(ruta_keras)
    directorio_modelos = asegurar_directorio(obtener(config, "rutas.modelos", "models"))
    base_nombre = ruta_keras.stem.replace("_finetuned", "").replace("_congelado", "")

    resultados: dict[str, Any] = {
        "modelo_origen": str(ruta_keras.name),
        "tamano_keras_mb": tamano_en_disco_mb(ruta_keras),
        "artefactos": {},
    }

    directorio_temporal = Path(tempfile.mkdtemp(prefix="tflite_export_"))
    try:
        print("\n[2/4] Exportando variante float32")
        conversor = _crear_conversor(modelo, directorio_temporal)
        tflite_f32 = conversor.convert()
        ruta_f32 = directorio_modelos / f"{base_nombre}_float32.tflite"
        ruta_f32.write_bytes(tflite_f32)
        tam_f32 = tamano_en_disco_mb(ruta_f32)
        print(f"  {ruta_f32.name}: {tam_f32:.3f} MB")
        resultados["artefactos"]["float32"] = {
            "archivo": str(ruta_f32.name),
            "tamano_mb": tam_f32,
        }

        if not args.solo_float32:
            print("\n[3/4] Exportando variante int8 (cuantizacion entera completa)")
            print(f"  Calibrando con {args.muestras_calibracion} imagenes reales del train...")
            datasets, _ = cargar_particiones(config, batch_size=1, verboso=False)

            conversor_int8 = _crear_conversor(modelo, directorio_temporal)
            conversor_int8.optimizations = [tf.lite.Optimize.DEFAULT]
            conversor_int8.representative_dataset = _generador_representativo(
                datasets["train"], args.muestras_calibracion
            )
            conversor_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if args.io_int8:
                conversor_int8.inference_input_type = tf.int8
                conversor_int8.inference_output_type = tf.int8

            tflite_int8 = conversor_int8.convert()
            ruta_int8 = directorio_modelos / f"{base_nombre}_int8.tflite"
            ruta_int8.write_bytes(tflite_int8)
            tam_int8 = tamano_en_disco_mb(ruta_int8)
            print(f"  {ruta_int8.name}: {tam_int8:.3f} MB")
            resultados["artefactos"]["int8"] = {
                "archivo": str(ruta_int8.name),
                "tamano_mb": tam_int8,
                "io_int8": bool(args.io_int8),
                "muestras_calibracion": args.muestras_calibracion,
            }
        else:
            print("\n[3/4] Variante int8 omitida (--solo-float32).")
    finally:
        shutil.rmtree(directorio_temporal, ignore_errors=True)

    print("\n[4/4] Resumen de compresion")
    tam_keras = resultados["tamano_keras_mb"]
    print(f"  .keras          : {tam_keras:8.3f} MB   (referencia)")
    for clave, datos in resultados["artefactos"].items():
        factor = tam_keras / datos["tamano_mb"] if datos["tamano_mb"] else float("nan")
        print(f"  .tflite {clave:<8}: {datos['tamano_mb']:8.3f} MB   ({factor:.1f}x mas pequeno)")

    ruta_json = met.guardar_json(resultados, "reports/metricas/exportacion_tflite.json")
    print(f"\n  Resumen -> {ruta_json}")
    print("\nListo. Siguiente paso: python scripts/evaluate.py --config config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
