"""Metricas de desempeno para clasificacion binaria de grietas.

Todas las funciones que calculan escriben ademas su resultado en
``reports/metricas/`` como JSON o CSV. Ese es el contrato del proyecto: la
aplicacion Streamlit **lee artefactos**, nunca recalcula ni trae numeros
escritos a mano en su codigo. Si una cifra aparece en la interfaz, existe un
archivo en ``reports/`` que la respalda.

Sobre que metrica mirar
-----------------------
La exactitud es la metrica menos informativa aqui. En un dataset con 15% de
imagenes agrietadas, un modelo que responde siempre "sin grieta" acierta el 85%
y no detecta ni una sola grieta. Lo que importa es el **recall de la clase
positiva**: la fraccion de grietas reales que el sistema encuentra. Un falso
negativo es una grieta peligrosa que nadie va a revisar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.rutas import asegurar_directorio, resolver


def predecir(modelo: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene etiquetas verdaderas y probabilidades predichas sobre un dataset.

    Se recorre el ``tf.data.Dataset`` acumulando las etiquetas en lugar de
    reutilizar el inventario, porque el pipeline puede haber descartado imagenes
    corruptas con ``ignore_errors()`` y los indices ya no coincidirian.

    Args:
        modelo: Modelo de Keras compilado o no.
        dataset: ``tf.data.Dataset`` que produce pares ``(imagenes, etiquetas)``.

    Returns:
        Tupla ``(y_true, y_prob)`` con dos vectores ``(N,)`` de tipo float.
    """
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for lote_x, lote_y in dataset:
        salida = modelo.predict(lote_x, verbose=0)
        y_prob.append(np.asarray(salida).reshape(-1))
        y_true.append(np.asarray(lote_y).reshape(-1))
    if not y_true:
        return np.empty(0), np.empty(0)
    return np.concatenate(y_true).astype(float), np.concatenate(y_prob).astype(float)


def calcular_metricas(
    y_true: np.ndarray, y_prob: np.ndarray, umbral: float = 0.5
) -> dict[str, Any]:
    """Calcula el cuadro completo de metricas para un umbral dado.

    Args:
        y_true: Etiquetas verdaderas ``(N,)`` con valores 0/1.
        y_prob: Probabilidades predichas ``(N,)`` en ``[0, 1]``.
        umbral: Umbral de decision. Bajarlo aumenta el recall de la clase
            positiva a costa de la precision.

    Returns:
        Diccionario con ``umbral``, ``n_muestras``, ``accuracy``, ``precision``,
        ``recall``, ``f1``, ``especificidad``, ``balanced_accuracy``, ``roc_auc``,
        ``pr_auc``, la matriz de confusion desglosada (``vn``, ``fp``, ``fn``,
        ``vp``) y ``tasa_falsos_negativos``.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= umbral).astype(int)
    y_verdad = y_true.astype(int)

    matriz = confusion_matrix(y_verdad, y_pred, labels=[0, 1])
    vn, fp, fn, vp = (int(v) for v in matriz.ravel())

    hay_dos_clases = len(np.unique(y_verdad)) == 2
    roc = float(roc_auc_score(y_verdad, y_prob)) if hay_dos_clases else None
    pr = float(average_precision_score(y_verdad, y_prob)) if hay_dos_clases else None

    recall = float(recall_score(y_verdad, y_pred, zero_division=0))
    especificidad = float(vn / (vn + fp)) if (vn + fp) > 0 else 0.0

    return {
        "umbral": float(umbral),
        "n_muestras": int(len(y_verdad)),
        "accuracy": float(accuracy_score(y_verdad, y_pred)),
        "precision": float(precision_score(y_verdad, y_pred, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(y_verdad, y_pred, zero_division=0)),
        "especificidad": especificidad,
        "balanced_accuracy": float((recall + especificidad) / 2.0),
        "roc_auc": roc,
        "pr_auc": pr,
        "matriz_confusion": {"vn": vn, "fp": fp, "fn": fn, "vp": vp},
        "matriz_confusion_lista": matriz.tolist(),
        "tasa_falsos_negativos": float(fn / (fn + vp)) if (fn + vp) > 0 else 0.0,
        "tasa_falsos_positivos": float(fp / (fp + vn)) if (fp + vn) > 0 else 0.0,
    }


def curva_precision_recall(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, list[float]]:
    """Calcula la curva precision-recall completa.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        y_prob: Probabilidades predichas ``(N,)``.

    Returns:
        Diccionario con las listas ``precision``, ``recall`` y ``umbrales``.
        ``umbrales`` tiene un elemento menos que las otras dos, por convencion de
        scikit-learn; se rellena con 1.0 al final para que las tres tengan la
        misma longitud y sean graficables directamente con Plotly.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, umbrales = precision_recall_curve(y_true.astype(int), y_prob)
    return {
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "umbrales": [float(v) for v in umbrales] + [1.0],
    }


def buscar_umbral_para_recall(
    y_true: np.ndarray, y_prob: np.ndarray, recall_objetivo: float = 0.95
) -> dict[str, float]:
    """Encuentra el umbral mas alto que aun garantiza el recall objetivo.

    El compromiso, explicado
    ------------------------
    Bajar el umbral hace que el modelo declare "grieta" con menos evidencia. Eso
    reduce los falsos negativos (grietas que se escapan) y aumenta los falsos
    positivos (paredes sanas marcadas como agrietadas). En seguridad estructural
    la asimetria es clara: un falso positivo cuesta una revision innecesaria; un
    falso negativo puede costar vidas. Por eso se elige el **umbral mas alto**
    que todavia cumple el recall exigido: se compra recall al menor precio
    posible en precision, no se regala precision sin necesidad.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        y_prob: Probabilidades predichas ``(N,)``.
        recall_objetivo: Recall minimo exigido para la clase positiva.

    Returns:
        Diccionario con ``umbral``, ``recall_alcanzado``, ``precision_resultante``
        y ``recall_objetivo``. Si ningun umbral alcanza el objetivo se devuelve
        ``umbral=0.0`` (el modelo declara todo positivo), que es el limite
        alcanzable.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, umbrales = precision_recall_curve(y_true.astype(int), y_prob)
    precision, recall = precision[:-1], recall[:-1]

    validos = np.flatnonzero(recall >= recall_objetivo)
    if validos.size == 0:
        return {
            "umbral": 0.0,
            "recall_alcanzado": 1.0,
            "precision_resultante": float(np.mean(y_true)),
            "recall_objetivo": float(recall_objetivo),
            "alcanzable": False,
        }

    mejor = int(validos[np.argmax(umbrales[validos])])
    return {
        "umbral": float(umbrales[mejor]),
        "recall_alcanzado": float(recall[mejor]),
        "precision_resultante": float(precision[mejor]),
        "recall_objetivo": float(recall_objetivo),
        "alcanzable": True,
    }


def analizar_falsos_negativos(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rutas: list[Path] | None = None,
    umbral: float = 0.5,
    top_n: int = 20,
) -> dict[str, Any]:
    """Analiza en detalle los falsos negativos: el error caro de este dominio.

    Args:
        y_true: Etiquetas verdaderas ``(N,)``.
        y_prob: Probabilidades predichas ``(N,)``.
        rutas: Rutas de las imagenes, en el mismo orden. Si se pasan, se listan
            los casos concretos para poder inspeccionarlos visualmente.
        umbral: Umbral de decision aplicado.
        top_n: Cuantos casos peores listar.

    Returns:
        Diccionario con ``n_falsos_negativos``, ``tasa``,
        ``probabilidad_media_fn``, ``probabilidad_mediana_fn``, la distribucion
        de probabilidades en ``histograma`` y la lista ``peores_casos``
        (los falsos negativos con probabilidad mas baja, es decir, aquellos en
        los que el modelo estuvo mas seguro de equivocarse).
    """
    y_pred = (y_prob >= umbral).astype(int)
    indices_fn = np.flatnonzero((y_true.astype(int) == 1) & (y_pred == 0))
    n_positivos = int((y_true.astype(int) == 1).sum())

    resultado: dict[str, Any] = {
        "umbral": float(umbral),
        "n_positivos_reales": n_positivos,
        "n_falsos_negativos": int(indices_fn.size),
        "tasa": float(indices_fn.size / n_positivos) if n_positivos else 0.0,
        "peores_casos": [],
    }

    if indices_fn.size == 0:
        resultado["probabilidad_media_fn"] = None
        resultado["probabilidad_mediana_fn"] = None
        return resultado

    probs_fn = y_prob[indices_fn]
    resultado["probabilidad_media_fn"] = float(np.mean(probs_fn))
    resultado["probabilidad_mediana_fn"] = float(np.median(probs_fn))
    resultado["fn_cerca_del_umbral"] = int((probs_fn >= umbral - 0.1).sum())

    orden = indices_fn[np.argsort(probs_fn)][:top_n]
    for indice in orden:
        caso: dict[str, Any] = {
            "indice": int(indice),
            "probabilidad": float(y_prob[indice]),
        }
        if rutas is not None and int(indice) < len(rutas):
            caso["archivo"] = Path(rutas[int(indice)]).name
        resultado["peores_casos"].append(caso)

    return resultado


def guardar_json(datos: dict[str, Any], ruta_relativa: str | Path) -> Path:
    """Guarda un diccionario como JSON con indentacion legible.

    Args:
        datos: Diccionario serializable.
        ruta_relativa: Ruta de destino relativa a la raiz del proyecto.

    Returns:
        Ruta absoluta del archivo escrito.
    """
    destino = resolver(ruta_relativa)
    asegurar_directorio(destino.parent)
    with destino.open("w", encoding="utf-8") as manejador:
        json.dump(datos, manejador, indent=2, ensure_ascii=False, default=_serializar)
    return destino


def _serializar(objeto: Any) -> Any:
    """Convierte tipos de NumPy a tipos nativos para ``json.dump``.

    Args:
        objeto: Objeto no serializable encontrado por el codificador JSON.

    Returns:
        Equivalente nativo de Python.

    Raises:
        TypeError: Si el objeto no es un tipo de NumPy reconocido.
    """
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        return float(objeto)
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    if isinstance(objeto, Path):
        return str(objeto)
    raise TypeError(f"Tipo no serializable: {type(objeto)}")


def guardar_historial(historial: dict[str, list[float]], ruta_relativa: str | Path) -> Path:
    """Guarda las curvas de entrenamiento como CSV.

    Args:
        historial: Diccionario ``History.history`` de Keras.
        ruta_relativa: Ruta de destino relativa a la raiz del proyecto.

    Returns:
        Ruta absoluta del CSV escrito.
    """
    destino = resolver(ruta_relativa)
    asegurar_directorio(destino.parent)
    tabla = pd.DataFrame(historial)
    tabla.insert(0, "epoca", np.arange(1, len(tabla) + 1))
    tabla.to_csv(destino, index=False, encoding="utf-8")
    return destino


def figura_matriz_confusion(
    metricas: dict[str, Any], clases: list[str], titulo: str, ruta_relativa: str | Path
) -> Path:
    """Genera y guarda la figura de la matriz de confusion.

    Args:
        metricas: Salida de :func:`calcular_metricas`.
        clases: Nombres de las clases, indexados por etiqueta.
        titulo: Titulo de la figura.
        ruta_relativa: Ruta de destino (PNG) relativa a la raiz.

    Returns:
        Ruta absoluta de la figura guardada.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matriz = np.array(metricas["matriz_confusion_lista"])
    figura, eje = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=clases,
        yticklabels=clases,
        ax=eje,
    )
    eje.set_xlabel("Prediccion")
    eje.set_ylabel("Etiqueta real")
    eje.set_title(f"{titulo}\numbral={metricas['umbral']:.2f}  F1={metricas['f1']:.3f}")
    figura.tight_layout()

    destino = resolver(ruta_relativa)
    asegurar_directorio(destino.parent)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino


def figura_curvas_entrenamiento(
    historial: dict[str, list[float]], titulo: str, ruta_relativa: str | Path
) -> Path:
    """Grafica las curvas de perdida y exactitud, entrenamiento contra validacion.

    La lectura relevante es la separacion entre ambas curvas: si la de
    validacion se estanca o empeora mientras la de entrenamiento sigue bajando,
    hay sobreajuste y el ``EarlyStopping`` deberia haber detenido la corrida.

    Args:
        historial: Diccionario ``History.history`` de Keras.
        titulo: Titulo de la figura.
        ruta_relativa: Ruta de destino (PNG) relativa a la raiz.

    Returns:
        Ruta absoluta de la figura guardada.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
    epocas = np.arange(1, len(historial.get("loss", [])) + 1)

    ejes[0].plot(epocas, historial.get("loss", []), label="entrenamiento")
    if "val_loss" in historial:
        ejes[0].plot(epocas, historial["val_loss"], label="validacion")
    ejes[0].set_xlabel("Epoca")
    ejes[0].set_ylabel("Perdida (binary crossentropy)")
    ejes[0].set_title("Perdida")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)

    clave_acc = "accuracy" if "accuracy" in historial else "binary_accuracy"
    ejes[1].plot(epocas, historial.get(clave_acc, []), label="entrenamiento")
    if f"val_{clave_acc}" in historial:
        ejes[1].plot(epocas, historial[f"val_{clave_acc}"], label="validacion")
    ejes[1].set_xlabel("Epoca")
    ejes[1].set_ylabel("Exactitud")
    ejes[1].set_title("Exactitud")
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)

    figura.suptitle(titulo)
    figura.tight_layout()

    destino = resolver(ruta_relativa)
    asegurar_directorio(destino.parent)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino


def figura_precision_recall(
    curva: dict[str, list[float]], titulo: str, ruta_relativa: str | Path
) -> Path:
    """Grafica la curva precision-recall.

    Args:
        curva: Salida de :func:`curva_precision_recall`.
        titulo: Titulo de la figura.
        ruta_relativa: Ruta de destino (PNG) relativa a la raiz.

    Returns:
        Ruta absoluta de la figura guardada.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, eje = plt.subplots(figsize=(5.5, 4.5))
    eje.plot(curva["recall"], curva["precision"], linewidth=2)
    eje.set_xlabel("Recall (clase: con grieta)")
    eje.set_ylabel("Precision")
    eje.set_title(titulo)
    eje.grid(alpha=0.3)
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1.02)
    figura.tight_layout()

    destino = resolver(ruta_relativa)
    asegurar_directorio(destino.parent)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino
