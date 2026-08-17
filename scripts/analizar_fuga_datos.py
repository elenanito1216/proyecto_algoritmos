"""Detecta fuga de datos entre particiones mediante hashing perceptual.

El problema
-----------
Los datasets publicos de grietas se construyen recortando **parches** de un
numero reducido de fotografias madre. Si el reparto train/val/test se hace al
azar, dos parches solapados de la misma pared pueden acabar uno en entrenamiento
y otro en prueba. El modelo reconoce entonces la textura concreta de esa pared,
no el concepto de grieta, y la exactitud en prueba sube a valores irreales.

`src/data/loader.py` ofrece `datos.split.agrupar_por_origen` para repartir por
superficie de origen, pero **depende de que el nombre del archivo codifique ese
origen**. Cuando los archivos se llaman ``00001.jpg``, ``00002.jpg``... no hay
nada que agrupar y esa salvaguarda queda inerte.

La solucion
-----------
Medir la fuga directamente sobre los pixeles. Se calcula un **pHash** (hash
perceptual basado en la DCT) de cada imagen y se buscan pares casi identicos que
crucen la frontera entre particiones. Dos imagenes con distancia de Hamming
pequena son visualmente la misma escena aunque difieran en compresion, recorte
menor o iluminacion.

Sobre el pHash
--------------
1. Escala de grises, redimension a 32x32.
2. Transformada discreta del coseno (DCT-II).
3. Se conserva el bloque 8x8 de bajas frecuencias, descartando el coeficiente DC
   (que solo codifica el brillo medio y no la estructura).
4. Cada uno de los 63 coeficientes restantes se compara con su mediana: mayor
   que la mediana es un 1, menor un 0.
5. El resultado son 64 bits robustos frente a cambios de escala, brillo y
   compresion, pero sensibles a la estructura de la imagen.

La distancia entre dos hashes es el numero de bits distintos (Hamming), de 0 a
64. Los umbrales habituales en la literatura: 0 = perceptualmente identicas,
<= 5 = casi duplicadas, <= 10 = muy similares.

Uso tipico:
    python scripts/analizar_fuga_datos.py --config config.yaml
    python scripts/analizar_fuga_datos.py --config config.yaml --umbral 8
    python scripts/analizar_fuga_datos.py --config config.yaml --sin-reevaluar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import construir_inventario, dividir  # noqa: E402
from src.eval import metricas as met  # noqa: E402
from src.utils.config import cargar_config, obtener  # noqa: E402
from src.utils.rutas import resolver  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402

# Tabla de popcount para bytes: cuantos bits a 1 tiene cada valor 0-255.
# Permite calcular distancias de Hamming vectorizadas sobre arreglos enteros.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Detecta fuga de datos entre particiones con hashing perceptual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al config YAML.")
    parser.add_argument(
        "--umbral",
        type=int,
        default=5,
        help=(
            "Distancia de Hamming maxima (de 64 bits) para considerar dos imagenes "
            "casi duplicadas. 0 = hash identico; 5 es el valor habitual."
        ),
    )
    parser.add_argument(
        "--umbral-correlacion",
        type=float,
        default=0.95,
        help=(
            "Correlacion de Pixeles minima para CONFIRMAR que un candidato del pHash "
            "es realmente un duplicado. El pHash solo propone; esto verifica."
        ),
    )
    parser.add_argument(
        "--sin-reevaluar",
        action="store_true",
        help=(
            "Omite el recalculo de metricas sobre el conjunto de prueba depurado. "
            "Util si aun no se ha ejecutado scripts/evaluate.py."
        ),
    )
    return parser


def verificar_par(ruta_a: Path, ruta_b: Path, umbral_correlacion: float) -> dict[str, Any]:
    """Confirma a nivel de pixel si dos imagenes son realmente duplicadas.

    El pHash solo **propone** candidatos: con 63 bits sobre texturas de hormigon,
    que son muy homogeneas en bajas frecuencias, colisiona con frecuencia. Esta
    funcion es la que decide.

    El discriminador es la **correlacion de Pearson**, no el error absoluto medio.
    Razon: en superficies uniformes, dos parches sin ninguna relacion tienen
    valores de gris parecidos y por tanto un MAE bajo. Medido sobre este dataset,
    el MAE no separa (duplicados reales dan 1.7-2.3, igual que algunos pares no
    relacionados), mientras que la correlacion separa sin ambiguedad: los
    duplicados quedan por encima de 0.995 y las imagenes distintas por debajo de
    0.19. Un MAE de ~2 con correlacion 0.996 es la firma de la misma imagen
    recodificada en JPEG con otra calidad.

    Args:
        ruta_a: Primera imagen.
        ruta_b: Segunda imagen.
        umbral_correlacion: Correlacion minima para declarar duplicado.

    Returns:
        Diccionario con ``duplicado`` (bool), ``correlacion`` y ``mae``.
    """
    a = cv2.imread(str(ruta_a), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(ruta_b), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return {"duplicado": False, "correlacion": float("nan"), "mae": float("nan")}

    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    va = a.ravel().astype(np.float64)
    vb = b.ravel().astype(np.float64)
    mae = float(np.abs(va - vb).mean())

    # Una imagen completamente plana no tiene varianza y la correlacion no esta
    # definida; en ese caso se decide por diferencia absoluta.
    if va.std() < 1e-9 or vb.std() < 1e-9:
        correlacion = 1.0 if mae < 1.0 else 0.0
    else:
        correlacion = float(np.corrcoef(va, vb)[0, 1])

    return {
        "duplicado": bool(correlacion >= umbral_correlacion),
        "correlacion": round(correlacion, 6),
        "mae": round(mae, 4),
    }


def calcular_phash(ruta: Path, lado: int = 32) -> np.uint64 | None:
    """Calcula el hash perceptual de 64 bits de una imagen.

    Args:
        ruta: Ruta al archivo de imagen.
        lado: Lado al que se redimensiona antes de la DCT. Debe ser >= 8.

    Returns:
        Hash de 64 bits como ``np.uint64``, o ``None`` si la imagen no se pudo
        leer (archivo corrupto o formato no soportado).
    """
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return None

    reducida = cv2.resize(imagen, (lado, lado), interpolation=cv2.INTER_AREA)
    transformada = cv2.dct(np.float32(reducida))

    # Bloque de bajas frecuencias, sin el coeficiente DC: este solo codifica el
    # brillo medio, y conservarlo haria el hash sensible a la exposicion.
    bloque = transformada[:8, :8].flatten()[1:]
    bits = bloque > np.median(bloque)

    return np.uint64(int("".join("1" if b else "0" for b in bits) + "0", 2))


def hashear_inventario(rutas: list[Path], etiqueta: str) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el pHash de una lista de imagenes mostrando progreso.

    Args:
        rutas: Rutas de las imagenes.
        etiqueta: Nombre de la particion, para el mensaje de progreso.

    Returns:
        Tupla ``(hashes, indices_validos)``. ``hashes`` contiene solo las
        imagenes legibles; ``indices_validos`` sus posiciones en la lista
        original, para poder mapear resultados de vuelta.
    """
    hashes: list[np.uint64] = []
    validos: list[int] = []
    inicio = time.perf_counter()

    for posicion, ruta in enumerate(rutas):
        h = calcular_phash(ruta)
        if h is not None:
            hashes.append(h)
            validos.append(posicion)

        if posicion % 5000 == 0 and posicion:
            transcurrido = time.perf_counter() - inicio
            ritmo = posicion / transcurrido
            restante = (len(rutas) - posicion) / max(ritmo, 1e-9)
            print(
                f"    {etiqueta}: {posicion:>6}/{len(rutas)} "
                f"({ritmo:.0f} img/s, faltan {restante:.0f} s)",
                flush=True,
            )

    ilegibles = len(rutas) - len(hashes)
    if ilegibles:
        print(f"    {etiqueta}: {ilegibles} imagen(es) ilegible(s), excluida(s)")

    return np.asarray(hashes, dtype=np.uint64), np.asarray(validos, dtype=np.int64)


def distancia_minima_cruzada(
    consulta: np.ndarray, referencia: np.ndarray, bloque: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Para cada hash de ``consulta``, halla su vecino mas cercano en ``referencia``.

    La comparacion es exhaustiva pero vectorizada: se procesa por bloques para
    acotar la memoria. Con 8.700 consultas y 40.700 referencias son ~355 millones
    de pares, que NumPy resuelve en decenas de segundos.

    Args:
        consulta: Hashes uint64 cuyo vecino se busca, forma ``(N,)``.
        referencia: Hashes uint64 contra los que se compara, forma ``(M,)``.
        bloque: Cuantas consultas se procesan a la vez.

    Returns:
        Tupla ``(distancias, indices)``: para cada consulta, la distancia de
        Hamming minima encontrada y la posicion del vecino en ``referencia``.
    """
    n = len(consulta)
    distancias = np.empty(n, dtype=np.uint8)
    indices = np.empty(n, dtype=np.int64)

    for inicio in range(0, n, bloque):
        fin = min(inicio + bloque, n)
        # XOR de cada consulta contra toda la referencia: los bits a 1 del
        # resultado son exactamente los bits en que ambos hashes difieren.
        xor = consulta[inicio:fin, None] ^ referencia[None, :]
        # popcount vectorizado: se ve el uint64 como 8 bytes y se suma la tabla.
        bytes_xor = xor.view(np.uint8).reshape(fin - inicio, len(referencia), 8)
        hamming = _POPCOUNT[bytes_xor].sum(axis=2)

        indices[inicio:fin] = hamming.argmin(axis=1)
        distancias[inicio:fin] = hamming.min(axis=1)

    return distancias, indices


def main() -> int:
    """Punto de entrada del analisis de fuga de datos.

    Returns:
        Codigo de salida: 0 si el analisis se completo.
    """
    args = construir_parser().parse_args()
    config = cargar_config(args.config)
    fijar_semillas(int(obtener(config, "proyecto.semilla", 42)))

    print("=" * 74)
    print("  ANALISIS DE FUGA DE DATOS ENTRE PARTICIONES (hashing perceptual)")
    print("=" * 74)

    print("\n[1/5] Reconstruyendo la particion exacta del entrenamiento")
    inventario = construir_inventario(obtener(config, "rutas.datos_raw"), config)
    particiones = dividir(inventario, config, verboso=False)
    for nombre, parte in particiones.items():
        print(f"  {nombre:<6}: {len(parte):>6} imagenes")

    print("\n[2/5] Calculando pHash de cada imagen")
    hashes: dict[str, np.ndarray] = {}
    validos: dict[str, np.ndarray] = {}
    for nombre in ("train", "val", "test"):
        hashes[nombre], validos[nombre] = hashear_inventario(particiones[nombre].rutas, nombre)
        print(f"  {nombre:<6}: {len(hashes[nombre]):>6} hashes")

    print(f"\n[3/5] Buscando vecinos mas cercanos (umbral = {args.umbral}/64 bits)")
    resultados: dict[str, Any] = {
        "umbral_hamming": args.umbral,
        "n_train": int(len(hashes["train"])),
        "n_val": int(len(hashes["val"])),
        "n_test": int(len(hashes["test"])),
        "cruces": {},
    }

    comparaciones = [("test", "train"), ("val", "train"), ("test", "val")]
    distancias_test_train: np.ndarray | None = None
    indices_test_train: np.ndarray | None = None

    for consulta, referencia in comparaciones:
        inicio = time.perf_counter()
        dist, idx = distancia_minima_cruzada(hashes[consulta], hashes[referencia])
        duracion = time.perf_counter() - inicio

        identicas = int((dist == 0).sum())
        casi = int((dist <= args.umbral).sum())
        similares = int((dist <= 10).sum())
        total = len(dist)

        print(
            f"  {consulta:>5} vs {referencia:<6} ({duracion:5.1f} s): "
            f"identicas {identicas:>5} ({100 * identicas / total:5.2f}%)  |  "
            f"<= {args.umbral}: {casi:>5} ({100 * casi / total:5.2f}%)  |  "
            f"<= 10: {similares:>5} ({100 * similares / total:5.2f}%)"
        )

        resultados["cruces"][f"{consulta}_vs_{referencia}"] = {
            "n_consulta": total,
            "identicas": identicas,
            "casi_duplicadas": casi,
            "similares_hasta_10": similares,
            "pct_identicas": round(100 * identicas / total, 4),
            "pct_casi_duplicadas": round(100 * casi / total, 4),
            "distancia_mediana": int(np.median(dist)),
            "distancia_minima": int(dist.min()),
            "histograma_0_a_15": [int((dist == d).sum()) for d in range(16)],
        }

        if consulta == "test" and referencia == "train":
            distancias_test_train, indices_test_train = dist, idx

    assert distancias_test_train is not None and indices_test_train is not None

    print("\n[4/5] Verificando pixel a pixel los candidatos del pHash")
    candidatos = np.flatnonzero(distancias_test_train <= args.umbral)
    print(
        f"  {len(candidatos)} candidatos con distancia <= {args.umbral}; "
        f"confirmando con correlacion >= {args.umbral_correlacion}..."
    )

    contaminadas = np.zeros(len(distancias_test_train), dtype=bool)
    confirmados: list[dict[str, Any]] = []
    inicio = time.perf_counter()

    for pos in candidatos:
        i_test = int(validos["test"][pos])
        i_train = int(validos["train"][indices_test_train[pos]])
        ruta_test = particiones["test"].rutas[i_test]
        ruta_train = particiones["train"].rutas[i_train]

        veredicto = verificar_par(ruta_test, ruta_train, args.umbral_correlacion)
        if veredicto["duplicado"]:
            contaminadas[pos] = True
            confirmados.append(
                {
                    "distancia_hash": int(distancias_test_train[pos]),
                    "test": ruta_test.name,
                    "train": ruta_train.name,
                    "clase_test": particiones["test"].clases[
                        int(particiones["test"].etiquetas[i_test])
                    ],
                    "clase_train": particiones["train"].clases[
                        int(particiones["train"].etiquetas[i_train])
                    ],
                    **veredicto,
                }
            )

    n_confirmados = int(contaminadas.sum())
    tasa = n_confirmados / max(len(candidatos), 1)
    print(
        f"  Confirmados {n_confirmados} de {len(candidatos)} candidatos "
        f"({100 * tasa:.1f}%) en {time.perf_counter() - inicio:.1f} s.\n"
        f"  El pHash propuso {len(candidatos) - n_confirmados} falsos positivos: "
        "es su limitacion sobre texturas homogeneas."
    )

    discordantes = [c for c in confirmados if c["clase_test"] != c["clase_train"]]
    if discordantes:
        print(
            f"  ATENCION: {len(discordantes)} duplicado(s) confirmado(s) tienen "
            "ETIQUETAS DISTINTAS entre train y test. Eso es ruido de etiquetado."
        )

    print("\n  Ejemplos confirmados:")
    for c in sorted(confirmados, key=lambda x: -x["correlacion"])[:8]:
        print(
            f"    r={c['correlacion']:.4f} mae={c['mae']:>6.2f}  "
            f"test/{c['test']} <-> train/{c['train']}  ({c['clase_test']})"
        )

    resultados["verificacion"] = {
        "umbral_correlacion": args.umbral_correlacion,
        "candidatos_phash": int(len(candidatos)),
        "duplicados_confirmados": n_confirmados,
        "falsos_positivos_phash": int(len(candidatos) - n_confirmados),
        "precision_del_phash": round(tasa, 4),
        "duplicados_con_etiqueta_discordante": len(discordantes),
    }
    resultados["duplicados_confirmados"] = confirmados[:200]
    resultados["test_contaminado"] = {
        "n_contaminadas": n_confirmados,
        "n_limpias": int((~contaminadas).sum()),
        "pct_contaminado": round(100 * float(contaminadas.mean()), 4),
    }

    print("\n[5/5] Reevaluando sobre el conjunto de prueba depurado")
    if args.sin_reevaluar:
        print("  (omitido por --sin-reevaluar)")
    else:
        resultados["reevaluacion"] = _reevaluar_sin_contaminadas(
            config, particiones["test"], validos["test"], contaminadas, args.umbral
        )

    ruta_json = met.guardar_json(resultados, "reports/metricas/fuga_datos.json")
    print(f"\n  Informe -> {ruta_json}")

    _imprimir_veredicto(resultados, args.umbral)
    return 0


def _reevaluar_sin_contaminadas(
    config: dict[str, Any],
    inventario_test: Any,
    validos_test: np.ndarray,
    contaminadas: np.ndarray,
    umbral: int,
) -> dict[str, Any] | None:
    """Recalcula las metricas excluyendo las imagenes de prueba contaminadas.

    Reutiliza las probabilidades ya almacenadas por ``scripts/evaluate.py``, de
    modo que no hace falta volver a pasar ningun modelo: el orden del conjunto
    de prueba es determinista (no se baraja) y coincide con el del inventario.

    Args:
        config: Configuracion del proyecto.
        inventario_test: Inventario de la particion de prueba.
        validos_test: Indices de las imagenes que se pudieron hashear.
        contaminadas: Mascara booleana sobre ``validos_test``.
        umbral: Umbral de Hamming aplicado, para el informe.

    Returns:
        Diccionario con las metricas antes y despues por modelo, o ``None`` si
        no se encontro el artefacto de evaluacion o el orden no cuadra.
    """
    ruta_eval = resolver("reports/metricas/evaluacion.json")
    if not ruta_eval.is_file():
        print("  No existe reports/metricas/evaluacion.json; ejecuta antes evaluate.py")
        return None

    datos = json.loads(ruta_eval.read_text(encoding="utf-8"))
    umbral_decision = float(datos.get("umbral", 0.5))
    salida: dict[str, Any] = {"umbral_hamming": umbral, "modelos": {}}

    for nombre, bloque in datos.get("modelos", {}).items():
        probs = bloque.get("test", {}).get("probabilidades")
        if not probs:
            continue

        y_true = np.asarray(probs["y_true"], dtype=int)
        y_prob = np.asarray(probs["y_prob"], dtype=float)

        if len(y_true) != len(inventario_test):
            print(
                f"  {nombre}: el artefacto tiene {len(y_true)} predicciones y el "
                f"inventario {len(inventario_test)}. Se omite."
            )
            continue

        # Verificacion de alineamiento: las etiquetas almacenadas deben coincidir
        # con las del inventario reconstruido. Si no, el orden no es el mismo y
        # cualquier filtrado por indice seria incorrecto.
        if not np.array_equal(y_true, inventario_test.etiquetas.astype(int)):
            print(f"  {nombre}: las etiquetas no coinciden con el inventario. Se omite.")
            continue

        mascara_limpia = np.ones(len(y_true), dtype=bool)
        mascara_limpia[validos_test[contaminadas]] = False

        antes = met.calcular_metricas(y_true, y_prob, umbral_decision)
        despues = met.calcular_metricas(
            y_true[mascara_limpia], y_prob[mascara_limpia], umbral_decision
        )

        salida["modelos"][nombre] = {
            "n_antes": int(len(y_true)),
            "n_despues": int(mascara_limpia.sum()),
            "antes": antes,
            "despues": despues,
            "delta_f1": round(despues["f1"] - antes["f1"], 6),
            "delta_accuracy": round(despues["accuracy"] - antes["accuracy"], 6),
            "delta_recall": round(despues["recall"] - antes["recall"], 6),
        }
        print(
            f"  {nombre:<26} F1 {antes['f1']:.4f} -> {despues['f1']:.4f} "
            f"({salida['modelos'][nombre]['delta_f1']:+.4f})  "
            f"sobre {int(mascara_limpia.sum())} imagenes limpias"
        )

    return salida


def _imprimir_veredicto(resultados: dict[str, Any], umbral: int) -> None:
    """Imprime la conclusion del analisis en lenguaje claro.

    Args:
        resultados: Diccionario de resultados acumulado.
        umbral: Umbral de Hamming usado.
    """
    cruce = resultados["cruces"]["test_vs_train"]
    verif = resultados["verificacion"]
    contaminado = resultados["test_contaminado"]
    pct = contaminado["pct_contaminado"]

    print("\n" + "=" * 74)
    print("  VEREDICTO")
    print("=" * 74)
    print(
        f"  El pHash propuso {verif['candidatos_phash']} candidatos "
        f"(distancia <= {umbral}/64).\n"
        f"  La verificacion pixel a pixel confirmo "
        f"{verif['duplicados_confirmados']} ({100 * verif['precision_del_phash']:.1f}% "
        f"de los candidatos).\n\n"
        f"  {contaminado['n_contaminadas']} de {cruce['n_consulta']} imagenes de prueba "
        f"({pct:.2f} %) tienen un duplicado CONFIRMADO en entrenamiento."
    )

    if pct < 1.0:
        print(
            "\n  -> FUGA DESPRECIABLE. El conjunto de prueba es esencialmente\n"
            "     independiente del de entrenamiento y las metricas reportadas\n"
            "     se sostienen."
        )
    elif pct < 10.0:
        print(
            "\n  -> FUGA MODERADA. Conviene reportar tambien las metricas sobre el\n"
            "     conjunto depurado y explicar la diferencia en el informe."
        )
    else:
        print(
            "\n  -> FUGA ALTA. Las metricas del conjunto de prueba completo estan\n"
            "     infladas. El numero honesto es el del conjunto depurado."
        )
    print("=" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
