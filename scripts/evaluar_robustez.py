"""Mide si TTA, ensemble y recalibracion del umbral cierran la brecha de dominio.

El problema que ataca
---------------------
El modelo alcanza F1 0.9423 sobre el conjunto de prueba depurado y 0.6667 sobre
las fotografias propias del equipo, con un recall de 0.50: se pierde la mitad de
las grietas reales (reports/analisis.md, §3.4). Ninguna de las tres tecnicas de
este script requiere reentrenar; todas operan sobre los modelos ya guardados.

Las tres tecnicas
-----------------
1. **TTA** (*test-time augmentation*): se predice sobre varias vistas
   transformadas de la misma imagen y se agregan las probabilidades. Las
   transformaciones son del grupo diedrico D4 (volteos y giros de 90 grados),
   que preservan la etiqueta: una grieta girada sigue siendo una grieta. Se
   evaluan dos agregaciones, porque tienen sentidos opuestos en este dominio:
   la **media** reduce la varianza, y el **maximo** dispara si *cualquier* vista
   ve grieta, lo que privilegia el recall.
2. **Ensemble**: promedio de la CNN de linea base y de MobileNetV2. Tiene
   sentido aqui por un resultado concreto: el modelo que gana en el conjunto
   publico es el que peor generaliza a fotos propias (0.667 frente a 0.750 de la
   linea base). Cuando dos modelos fallan en casos distintos, promediarlos suele
   batir a ambos fuera de distribucion.
3. **Recalibracion del umbral**: se elige el umbral **sobre el conjunto de
   prueba** y se aplica a las fotos propias. Elegirlo sobre las fotos propias y
   despues medir en ellas seria circular; ese valor se reporta aparte y
   etiquetado como cota superior, no como resultado.

Uso tipico:
    python scripts/evaluar_robustez.py --config config.yaml
    python scripts/evaluar_robustez.py --config config.yaml --sin-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.loader import cargar_particiones, cargar_propias  # noqa: E402
from src.eval import metricas as met  # noqa: E402
from src.utils.config import cargar_config, forma_entrada, obtener  # noqa: E402
from src.utils.dispositivo import detectar_dispositivo  # noqa: E402
from src.utils.rutas import resolver  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Evalua TTA, ensemble y recalibracion del umbral.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al config YAML.")
    parser.add_argument(
        "--sin-test",
        action="store_true",
        help="Evalua solo sobre las fotos propias. Mucho mas rapido para iterar.",
    )
    parser.add_argument(
        "--recall-objetivo",
        type=float,
        default=0.95,
        help="Recall que debe garantizar el umbral calibrado sobre el conjunto de prueba.",
    )
    return parser


def generar_vistas(lote: np.ndarray, n_vistas: int) -> list[np.ndarray]:
    """Genera las vistas del grupo diedrico D4 para test-time augmentation.

    Las ocho transformaciones (identidad, tres giros de 90 grados y sus volteos)
    preservan la etiqueta en este dominio: una grieta girada o reflejada sigue
    siendo una grieta. Es la misma justificacion por la que el aumento de
    entrenamiento incluye volteo vertical, poco habitual en imagen natural.

    Args:
        lote: Tensor ``(N, H, W, C)``. H y W deben ser iguales para los giros.
        n_vistas: 1 (sin TTA), 4 (identidad y volteos) u 8 (D4 completo).

    Returns:
        Lista de ``n_vistas`` tensores del mismo tamano que ``lote``.
    """
    vistas = [lote]
    if n_vistas >= 4:
        vistas.append(lote[:, :, ::-1, :])  # volteo horizontal
        vistas.append(lote[:, ::-1, :, :])  # volteo vertical
        vistas.append(lote[:, ::-1, ::-1, :])  # giro de 180 grados
    if n_vistas >= 8:
        girado = np.rot90(lote, k=1, axes=(1, 2))
        vistas.append(girado)
        vistas.append(girado[:, :, ::-1, :])
        vistas.append(girado[:, ::-1, :, :])
        vistas.append(girado[:, ::-1, ::-1, :])
    return [np.ascontiguousarray(v) for v in vistas[:n_vistas]]


def predecir_con_vistas(
    modelos: dict[str, Any], dataset: Any, n_vistas: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recorre un dataset prediciendo con cada modelo sobre cada vista.

    Args:
        modelos: Diccionario ``{nombre: modelo_keras}``.
        dataset: ``tf.data.Dataset`` que produce ``(imagenes, etiquetas)``.
        n_vistas: Numero de vistas TTA a generar.

    Returns:
        Tupla ``(y_true, probabilidades)``. ``probabilidades[nombre]`` tiene
        forma ``(n_vistas, N)``.
    """
    etiquetas: list[np.ndarray] = []
    acumulado: dict[str, list[list[np.ndarray]]] = {
        n: [[] for _ in range(n_vistas)] for n in modelos
    }

    for lote_x, lote_y in dataset:
        etiquetas.append(np.asarray(lote_y).reshape(-1))
        vistas = generar_vistas(np.asarray(lote_x, dtype=np.float32), n_vistas)
        for nombre, modelo in modelos.items():
            for indice, vista in enumerate(vistas):
                salida = modelo(vista, training=False)
                acumulado[nombre][indice].append(np.asarray(salida).reshape(-1))

    y_true = np.concatenate(etiquetas).astype(int)
    probabilidades = {
        nombre: np.stack([np.concatenate(vista) for vista in vistas])
        for nombre, vistas in acumulado.items()
    }
    return y_true, probabilidades


def construir_variantes(
    probabilidades: dict[str, np.ndarray], etiqueta_base: str, etiqueta_transfer: str
) -> dict[str, np.ndarray]:
    """Combina las predicciones por vista en las variantes a comparar.

    Args:
        probabilidades: ``{nombre: (n_vistas, N)}``.
        etiqueta_base: Clave de la CNN de linea base.
        etiqueta_transfer: Clave de MobileNetV2.

    Returns:
        Diccionario ``{nombre_variante: probabilidades (N,)}``.
    """
    p_base = probabilidades[etiqueta_base]
    p_transfer = probabilidades[etiqueta_transfer]
    n_vistas = p_transfer.shape[0]

    variantes: dict[str, np.ndarray] = {
        "Linea base": p_base[0],
        "MobileNetV2": p_transfer[0],
        "Ensemble (media)": (p_base[0] + p_transfer[0]) / 2.0,
    }

    for k in (4, 8):
        if n_vistas >= k:
            variantes[f"MobileNetV2 + TTA{k} media"] = p_transfer[:k].mean(axis=0)
            variantes[f"MobileNetV2 + TTA{k} maximo"] = p_transfer[:k].max(axis=0)

    if n_vistas >= 8:
        variantes["Ensemble + TTA8 media"] = (
            p_base[:8].mean(axis=0) + p_transfer[:8].mean(axis=0)
        ) / 2.0
        variantes["Ensemble + TTA8 maximo"] = np.maximum(
            p_base[:8].max(axis=0), p_transfer[:8].max(axis=0)
        )

    return variantes


def _cargar_mascara_limpia(n_test: int) -> np.ndarray | None:
    """Construye la mascara del conjunto de prueba sin duplicados de entrenamiento.

    Args:
        n_test: Numero de imagenes del conjunto de prueba.

    Returns:
        Mascara booleana, o ``None`` si no existe el artefacto de fuga de datos.
    """
    ruta = resolver("reports/metricas/fuga_datos.json")
    if not ruta.is_file():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    indices = datos.get("test_contaminado", {}).get("indices_en_inventario")
    if not indices:
        return None
    mascara = np.ones(n_test, dtype=bool)
    mascara[np.asarray(indices, dtype=int)] = False
    return mascara


def main() -> int:
    """Punto de entrada de la evaluacion de robustez.

    Returns:
        Codigo de salida: 0 si termino correctamente.
    """
    import tensorflow as tf

    args = construir_parser().parse_args()
    config = cargar_config(args.config)
    fijar_semillas(int(obtener(config, "proyecto.semilla", 42)))
    info = detectar_dispositivo(verboso=False)

    umbral_defecto = float(obtener(config, "evaluacion.umbral", 0.5))
    etiqueta_base, etiqueta_transfer = "Linea base", "MobileNetV2"

    print("=" * 78)
    print("  ROBUSTEZ: TEST-TIME AUGMENTATION, ENSEMBLE Y RECALIBRACION DEL UMBRAL")
    print("=" * 78)

    print("\n[1/5] Cargando modelos")
    rutas = {
        etiqueta_base: resolver(obtener(config, "app.archivo_baseline")),
        etiqueta_transfer: resolver(obtener(config, "app.archivo_keras")),
    }
    for nombre, ruta in rutas.items():
        if not ruta.is_file():
            print(f"ERROR: falta {ruta}. Entrena antes los dos modelos.")
            return 1
        print(f"  {nombre:<14} {ruta.name}")
    modelos = {n: tf.keras.models.load_model(r) for n, r in rutas.items()}

    resultados: dict[str, Any] = {
        "dispositivo": info,
        "umbral_por_defecto": umbral_defecto,
        "recall_objetivo": args.recall_objetivo,
        "conjuntos": {},
    }

    print("\n[2/5] Cargando conjuntos")
    datasets, inventarios = cargar_particiones(config, batch_size=32, verboso=False)
    ds_propias, inv_propias = cargar_propias(config, batch_size=32)
    if ds_propias is None:
        print("ERROR: no hay fotos propias en data/propias/.")
        return 1
    print(f"  test    : {len(inventarios['test'])} imagenes")
    print(f"  propias : {len(inv_propias)} imagenes")

    umbrales_calibrados: dict[str, float] = {}

    if not args.sin_test:
        print("\n[3/5] Prediciendo sobre el conjunto de prueba (8 vistas x 2 modelos)")
        inicio = time.perf_counter()
        y_test, probs_test = predecir_con_vistas(modelos, datasets["test"], n_vistas=8)
        print(f"  {len(y_test)} imagenes en {time.perf_counter() - inicio:.0f} s")

        mascara = _cargar_mascara_limpia(len(y_test))
        if mascara is None:
            print("  AVISO: sin artefacto de fuga de datos; se usa el test completo.")
            mascara = np.ones(len(y_test), dtype=bool)
        else:
            print(f"  Conjunto depurado: {int(mascara.sum())} imagenes limpias")

        variantes_test = construir_variantes(probs_test, etiqueta_base, etiqueta_transfer)
        bloque_test: dict[str, Any] = {}

        print(
            f"\n  {'variante':<30}{'F1':>8}{'recall':>8}{'prec':>8}{'umbral*':>9}{'F1*':>8}{'rec*':>8}"
        )
        print("  " + "-" * 79)
        for nombre, prob in variantes_test.items():
            m = met.calcular_metricas(y_test[mascara], prob[mascara], umbral_defecto)
            calibrado = met.buscar_umbral_para_recall(
                y_test[mascara], prob[mascara], args.recall_objetivo
            )
            umbrales_calibrados[nombre] = float(calibrado["umbral"])
            m_cal = met.calcular_metricas(y_test[mascara], prob[mascara], calibrado["umbral"])
            bloque_test[nombre] = {
                "umbral_defecto": m,
                "umbral_calibrado": m_cal,
                "umbral_elegido": calibrado["umbral"],
            }
            print(
                f"  {nombre:<30}{m['f1']:>8.4f}{m['recall']:>8.4f}{m['precision']:>8.4f}"
                f"{calibrado['umbral']:>9.3f}{m_cal['f1']:>8.4f}{m_cal['recall']:>8.4f}"
            )
        resultados["conjuntos"]["test_depurado"] = bloque_test
        print(
            "\n  (*) umbral calibrado para garantizar recall >= "
            f"{args.recall_objetivo:.0%} sobre este conjunto"
        )

    print("\n[4/5] Prediciendo sobre las fotos propias")
    y_propias, probs_propias = predecir_con_vistas(modelos, ds_propias, n_vistas=8)
    variantes_propias = construir_variantes(probs_propias, etiqueta_base, etiqueta_transfer)
    bloque_propias: dict[str, Any] = {}

    print(
        f"\n  {'variante':<30}{'F1':>8}{'recall':>8}{'prec':>8}{'FN':>5}"
        f"{'|':>3}{'F1 cal':>8}{'rec cal':>9}{'FN':>5}"
    )
    print("  " + "-" * 79)
    for nombre, prob in variantes_propias.items():
        m = met.calcular_metricas(y_propias, prob, umbral_defecto)
        fila: dict[str, Any] = {"umbral_defecto": m}
        texto_cal = ""

        if nombre in umbrales_calibrados:
            u = umbrales_calibrados[nombre]
            m_cal = met.calcular_metricas(y_propias, prob, u)
            fila["umbral_calibrado_en_test"] = {"umbral": u, **m_cal}
            texto_cal = (
                f"{'|':>3}{m_cal['f1']:>8.4f}{m_cal['recall']:>9.4f}"
                f"{m_cal['matriz_confusion']['fn']:>5}"
            )

        # Cota superior: el mejor umbral posible SOBRE ESTAS MISMAS fotos. No es
        # un resultado utilizable (seria circular), sino el techo que marca
        # cuanto margen deja la recalibracion.
        rejilla = np.unique(np.concatenate([prob, [0.0, 1.0]]))
        mejor = max(rejilla, key=lambda u: met.calcular_metricas(y_propias, prob, u)["f1"])
        fila["oraculo_en_propias"] = {
            "umbral": float(mejor),
            **met.calcular_metricas(y_propias, prob, mejor),
        }

        bloque_propias[nombre] = fila
        print(
            f"  {nombre:<30}{m['f1']:>8.4f}{m['recall']:>8.4f}{m['precision']:>8.4f}"
            f"{m['matriz_confusion']['fn']:>5}{texto_cal}"
        )

    resultados["conjuntos"]["propias"] = bloque_propias

    print("\n[5/5] Costo de inferencia por variante")
    alto, ancho, canales = forma_entrada(config)
    muestra = np.random.default_rng(0).random((1, alto, ancho, canales)).astype(np.float32)
    costos: dict[str, dict[str, float]] = {}

    for nombre, n_vistas, usados in (
        ("MobileNetV2", 1, [etiqueta_transfer]),
        ("MobileNetV2 + TTA4", 4, [etiqueta_transfer]),
        ("MobileNetV2 + TTA8", 8, [etiqueta_transfer]),
        ("Ensemble", 1, [etiqueta_base, etiqueta_transfer]),
        ("Ensemble + TTA8", 8, [etiqueta_base, etiqueta_transfer]),
    ):
        vistas = generar_vistas(muestra, n_vistas)
        for _ in range(5):  # calentamiento
            for m in usados:
                for v in vistas:
                    modelos[m](v, training=False)
        tiempos = []
        for _ in range(20):
            t0 = time.perf_counter()
            for m in usados:
                for v in vistas:
                    modelos[m](v, training=False)
            tiempos.append((time.perf_counter() - t0) * 1000.0)
        costos[nombre] = {
            "latencia_media_ms": round(float(np.mean(tiempos)), 2),
            "latencia_std_ms": round(float(np.std(tiempos, ddof=1)), 2),
            "pasadas": n_vistas * len(usados),
        }
        print(
            f"  {nombre:<22} {costos[nombre]['latencia_media_ms']:>8.2f} ms "
            f"(+-{costos[nombre]['latencia_std_ms']:.2f})  "
            f"{costos[nombre]['pasadas']} pasada(s)"
        )

    resultados["costo"] = costos
    ruta_json = met.guardar_json(resultados, "reports/metricas/robustez.json")
    print(f"\n  Informe -> {ruta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
