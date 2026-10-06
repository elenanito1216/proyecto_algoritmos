"""Infraestructura de entrenamiento compartida entre la linea base y el transfer.

Existe para que ``scripts/train_baseline.py`` y ``scripts/train_transfer.py`` no
dupliquen la construccion de callbacks ni la logica de reanudacion. La regla del
proyecto es que los scripts orquestan y los modulos de ``src/`` implementan.

Reanudacion tras una interrupcion
---------------------------------
Entrenar en CPU puede llevar horas y la sesion se puede cortar (cierre del
portatil, corte de luz, ``Ctrl+C``). Por eso se guarda un checkpoint **en cada
epoca** junto a un pequeno JSON de estado con la ultima epoca completada. Con
``--resume``, el script recarga el modelo y arranca ``model.fit`` con
``initial_epoch``, de modo que ni se repite trabajo ni se falsean las curvas.
``ModelCheckpoint`` por si solo no basta: guarda los pesos pero no en que epoca
iba.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.config import obtener
from src.utils.rutas import asegurar_directorio, resolver


def _ruta_checkpoint(config: dict[str, Any], nombre: str) -> Path:
    """Ruta del checkpoint reanudable de un experimento.

    Args:
        config: Configuracion del proyecto.
        nombre: Nombre del experimento, p. ej. ``"baseline_cnn"``.

    Returns:
        Ruta absoluta del archivo ``.keras`` de checkpoint.
    """
    directorio = asegurar_directorio(Path(obtener(config, "rutas.modelos", "models")) / "checkpoints")
    return directorio / f"{nombre}_ultimo.keras"


def _ruta_estado(config: dict[str, Any], nombre: str) -> Path:
    """Ruta del JSON con el estado de reanudacion de un experimento.

    Args:
        config: Configuracion del proyecto.
        nombre: Nombre del experimento.

    Returns:
        Ruta absoluta del archivo JSON de estado.
    """
    directorio = asegurar_directorio(Path(obtener(config, "rutas.modelos", "models")) / "checkpoints")
    return directorio / f"{nombre}_estado.json"


class RegistroDeProgreso:
    """Callback que persiste la ultima epoca completada tras cada epoca.

    Attributes:
        ruta: Archivo JSON donde se escribe el estado.
    """

    def __init__(self, ruta: Path, nombre: str) -> None:
        self.ruta = ruta
        self._nombre = nombre

    def crear_callback(self) -> Any:
        """Construye el callback de Keras asociado.

        Returns:
            Instancia de ``keras.callbacks.Callback``.
        """
        import tensorflow as tf

        registro = self

        class _CallbackProgreso(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                estado = {
                    "experimento": registro._nombre,
                    "ultima_epoca_completada": int(epoch) + 1,
                    "metricas": {k: float(v) for k, v in (logs or {}).items()},
                }
                registro.ruta.write_text(
                    json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        return _CallbackProgreso()


def construir_callbacks(
    config: dict[str, Any], nombre: str, total_epocas: int
) -> tuple[list[Any], Any]:
    """Construye la lista de callbacks del entrenamiento.

    Incluye ``EarlyStopping`` (con ``restore_best_weights``), ``ModelCheckpoint``
    guardando cada epoca, ``ReduceLROnPlateau``, el registro de progreso para
    poder reanudar y el medidor de tiempos por epoca.

    Args:
        config: Configuracion del proyecto.
        nombre: Nombre del experimento; determina los nombres de archivo.
        total_epocas: Epocas planificadas, usado para estimar el tiempo restante.

    Returns:
        Tupla ``(callbacks, medidor)``. El medidor se devuelve aparte porque el
        script necesita consultar sus tiempos al terminar.
    """
    import tensorflow as tf

    from src.utils.dispositivo import MedidorDeEpocas

    cfg = obtener(config, "entrenamiento.callbacks", {})
    callbacks: list[Any] = []

    es = cfg.get("early_stopping", {})
    if es.get("activo", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es.get("monitor", "val_loss"),
                patience=int(es.get("paciencia", 5)),
                restore_best_weights=bool(es.get("restaurar_mejores_pesos", True)),
                verbose=1,
            )
        )

    chk = cfg.get("checkpoint", {})
    if chk.get("activo", True):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(_ruta_checkpoint(config, nombre)),
                monitor=chk.get("monitor", "val_loss"),
                save_best_only=bool(chk.get("guardar_solo_mejor", False)),
                save_weights_only=False,
                save_freq="epoch",
                verbose=0,
            )
        )

    rlr = cfg.get("reduce_lr", {})
    if rlr.get("activo", True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=rlr.get("monitor", "val_loss"),
                factor=float(rlr.get("factor", 0.5)),
                patience=int(rlr.get("paciencia", 2)),
                min_lr=float(rlr.get("lr_minimo", 1e-7)),
                verbose=1,
            )
        )

    callbacks.append(RegistroDeProgreso(_ruta_estado(config, nombre), nombre).crear_callback())

    medidor = MedidorDeEpocas()
    callbacks.append(medidor.crear_callback(total_epocas))

    return callbacks, medidor


def intentar_reanudar(config: dict[str, Any], nombre: str) -> tuple[Any | None, int]:
    """Carga el ultimo checkpoint de un experimento, si existe.

    Args:
        config: Configuracion del proyecto.
        nombre: Nombre del experimento.

    Returns:
        Tupla ``(modelo, epoca_inicial)``. Si no hay checkpoint devuelve
        ``(None, 0)``.
    """
    import tensorflow as tf

    ruta_modelo = _ruta_checkpoint(config, nombre)
    ruta_estado = _ruta_estado(config, nombre)

    if not ruta_modelo.is_file():
        print(f"  No hay checkpoint previo en {ruta_modelo.name}; se entrena desde cero.")
        return None, 0

    epoca_inicial = 0
    if ruta_estado.is_file():
        try:
            estado = json.loads(ruta_estado.read_text(encoding="utf-8"))
            epoca_inicial = int(estado.get("ultima_epoca_completada", 0))
        except (json.JSONDecodeError, ValueError):
            print("  El archivo de estado esta corrupto; se reanuda desde la epoca 0.")

    modelo = tf.keras.models.load_model(ruta_modelo)
    print(f"  Reanudando desde {ruta_modelo.name}, epoca inicial = {epoca_inicial}.")
    return modelo, epoca_inicial


def guardar_modelo_final(modelo: Any, config: dict[str, Any], nombre_archivo: str) -> Path:
    """Guarda el modelo entrenado en formato ``.keras``.

    Args:
        modelo: Modelo de Keras entrenado.
        config: Configuracion del proyecto.
        nombre_archivo: Nombre del archivo con extension, p. ej.
            ``"mobilenetv2_finetuned.keras"``.

    Returns:
        Ruta absoluta del archivo guardado.
    """
    directorio = asegurar_directorio(obtener(config, "rutas.modelos", "models"))
    destino = directorio / nombre_archivo
    modelo.save(destino)
    return destino


def fusionar_historiales(*historiales: dict[str, list[float]]) -> dict[str, list[float]]:
    """Concatena varios ``History.history`` en uno solo.

    Necesario en el transfer learning, donde el entrenamiento tiene dos etapas
    (cabeza congelada y fine-tuning) y las curvas del informe deben mostrarlas
    de forma continua.

    Args:
        *historiales: Diccionarios ``History.history`` en orden cronologico.

    Returns:
        Diccionario con las series concatenadas. Las claves ausentes en alguna
        etapa se rellenan con ``float("nan")`` para que todas las series
        conserven la misma longitud y sean graficables.
    """
    claves: list[str] = []
    for historial in historiales:
        for clave in historial:
            if clave not in claves:
                claves.append(clave)

    fusionado: dict[str, list[float]] = {clave: [] for clave in claves}
    for historial in historiales:
        longitud = max((len(v) for v in historial.values()), default=0)
        for clave in claves:
            serie = historial.get(clave, [])
            fusionado[clave].extend(
                [float(v) for v in serie] + [float("nan")] * (longitud - len(serie))
            )
    return fusionado


def resumen_entrenamiento(
    nombre: str,
    historial: dict[str, list[float]],
    tiempos: dict[str, float],
    info_dispositivo: dict[str, Any],
    parametros: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Empaqueta el resumen de una corrida para guardarlo como JSON.

    Args:
        nombre: Nombre del experimento.
        historial: Curvas de entrenamiento.
        tiempos: Salida de ``MedidorDeEpocas.resumen()``.
        info_dispositivo: Salida de ``detectar_dispositivo()``.
        parametros: Conteo de parametros del modelo.
        extra: Campos adicionales especificos del experimento.

    Returns:
        Diccionario listo para ``guardar_json``.
    """
    epocas = max((len(v) for v in historial.values()), default=0)
    mejor_val_loss = None
    if historial.get("val_loss"):
        finitos = [v for v in historial["val_loss"] if v == v]
        mejor_val_loss = min(finitos) if finitos else None

    resumen: dict[str, Any] = {
        "experimento": nombre,
        "epocas_ejecutadas": epocas,
        "mejor_val_loss": mejor_val_loss,
        "parametros": parametros,
        "tiempos": tiempos,
        "dispositivo": info_dispositivo,
    }
    if extra:
        resumen.update(extra)
    return resumen


def resolver_artefacto(config: dict[str, Any], clave: str) -> Path:
    """Resuelve la ruta de un artefacto declarado en la seccion ``app``.

    Args:
        config: Configuracion del proyecto.
        clave: Clave dentro de ``app``, p. ej. ``"archivo_keras"``.

    Returns:
        Ruta absoluta del artefacto.
    """
    return resolver(obtener(config, f"app.{clave}"))
