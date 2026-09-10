"""Carga, particion y aumento del conjunto de datos de grietas.

Responsabilidades del modulo:

1. **Descubrir** la estructura del dataset sin que el usuario tenga que
   declararla: subcarpetas por clase (``Positive/``, ``Negative/``) o un CSV de
   etiquetas. Ambas convenciones aparecen en los datasets publicos de referencia
   (Concrete Crack Images usa subcarpetas; SDNET2018 mezcla ambas).
2. **Particionar** de forma estratificada 70/15/15 con semilla fija, con la
   opcion de agrupar por superficie de origen para evitar fuga de datos.
3. **Construir** pipelines ``tf.data`` con ``cache()`` y ``prefetch()``, y
   aplicar aumento **solo** al conjunto de entrenamiento.

Representacion del dato: cada imagen se decodifica a un tensor
``(alto, ancho, 3)`` de tipo ``float32``. Un lote es entonces un tensor
``(N, H, W, C)`` = ``(batch_size, 160, 160, 3)`` con valores normalizados al
rango ``[0, 1]`` (division por 255). La normalizacion especifica que exige cada
arquitectura (MobileNetV2 espera ``[-1, 1]``) se aplica **dentro** del modelo,
como primera capa, para que la app solo tenga que preprocesar una vez.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import forma_entrada, obtener
from src.utils.rutas import resolver


@dataclass
class Inventario:
    """Catalogo de imagenes descubiertas en un directorio.

    Attributes:
        rutas: Rutas absolutas de cada imagen.
        etiquetas: Vector de enteros ``(N,)`` con 0 = sin grieta, 1 = con grieta.
        clases: Nombres de clase indexados por su etiqueta entera.
        grupos: Identificador de superficie de origen por imagen, o ``None`` si
            no se solicito agrupacion. Se usa para evitar fuga de datos.
        estructura: ``"subcarpetas"`` o ``"csv"``; util para el informe.
    """

    rutas: list[Path]
    etiquetas: np.ndarray
    clases: list[str]
    grupos: np.ndarray | None = None
    estructura: str = "desconocida"
    metadatos: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rutas)

    def conteo_por_clase(self) -> dict[str, int]:
        """Cuenta cuantas imagenes hay de cada clase.

        Returns:
            Diccionario ``{nombre_clase: n_imagenes}``.
        """
        return {
            nombre: int((self.etiquetas == indice).sum())
            for indice, nombre in enumerate(self.clases)
        }

    def a_dataframe(self) -> pd.DataFrame:
        """Convierte el inventario en un ``DataFrame`` para el EDA.

        Returns:
            DataFrame con columnas ``ruta``, ``archivo``, ``etiqueta``,
            ``clase`` y (si existe) ``grupo``.
        """
        datos: dict[str, Any] = {
            "ruta": [str(p) for p in self.rutas],
            "archivo": [p.name for p in self.rutas],
            "etiqueta": self.etiquetas,
            "clase": [self.clases[int(e)] for e in self.etiquetas],
        }
        if self.grupos is not None:
            datos["grupo"] = self.grupos
        return pd.DataFrame(datos)


# --------------------------------------------------------------------------- #
# Descubrimiento de la estructura
# --------------------------------------------------------------------------- #


def _normalizar(texto: str) -> str:
    """Pasa un texto a minusculas sin acentos, para comparar nombres de carpeta.

    Args:
        texto: Cadena original.

    Returns:
        Cadena en minusculas, sin tildes ni dieresis.
    """
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).lower().strip()


def detectar_estructura(directorio: Path, config: dict[str, Any]) -> str:
    """Determina si el dataset esta organizado en subcarpetas o descrito por CSV.

    La deteccion es explicita y verificable en lugar de asumida: primero busca
    subcarpetas cuyos nombres coincidan con los alias de clase declarados en
    ``config.yaml``; si no las encuentra, busca el CSV de etiquetas.

    Args:
        directorio: Directorio raiz del dataset.
        config: Configuracion del proyecto.

    Returns:
        ``"subcarpetas"`` o ``"csv"``.

    Raises:
        FileNotFoundError: Si el directorio no existe.
        ValueError: Si no se reconoce ninguna de las dos estructuras.
    """
    if not directorio.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de datos: {directorio}\n"
            "Revisa 'rutas.datos_raw' en config.yaml y coloca alli el dataset."
        )

    alias = obtener(config, "datos.alias_clases", {})
    reconocidos = {_normalizar(a) for lista in alias.values() for a in lista}
    subcarpetas = [d for d in directorio.iterdir() if d.is_dir()]
    if any(_normalizar(d.name) in reconocidos for d in subcarpetas):
        return "subcarpetas"

    nombre_csv = obtener(config, "datos.csv.nombre_archivo", "labels.csv")
    if (directorio / nombre_csv).is_file():
        return "csv"

    # Un unico CSV en la carpeta tambien vale: el nombre exacto es un detalle.
    csvs = list(directorio.glob("*.csv"))
    if len(csvs) == 1:
        return "csv"

    nombres = ", ".join(sorted(d.name for d in subcarpetas)) or "(ninguna)"
    raise ValueError(
        f"No se reconocio la estructura del dataset en {directorio}.\n"
        f"  Subcarpetas encontradas: {nombres}\n"
        f"  Se esperaba una carpeta por clase (p. ej. 'Positive/' y 'Negative/')\n"
        f"  o un archivo '{nombre_csv}' con las etiquetas.\n"
        "  Ajusta 'datos.alias_clases' o 'datos.csv' en config.yaml."
    )


def _mapa_alias_a_indice(config: dict[str, Any]) -> dict[str, int]:
    """Construye el mapa ``alias normalizado -> etiqueta entera``.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Diccionario que traduce cualquier alias aceptado a su indice de clase.
    """
    clases: list[str] = obtener(config, "datos.clases", ["Negative", "Positive"])
    alias: dict[str, list[str]] = obtener(config, "datos.alias_clases", {})
    mapa: dict[str, int] = {}
    for indice, canonico in enumerate(clases):
        mapa[_normalizar(canonico)] = indice
        for variante in alias.get(canonico, []):
            mapa[_normalizar(str(variante))] = indice
    return mapa


def _extraer_grupo(nombre: str, patron: str) -> str:
    """Extrae el identificador de superficie de origen del nombre de archivo.

    Args:
        nombre: Nombre del archivo sin extension.
        patron: Expresion regular cuyo grupo 1 es el identificador.

    Returns:
        El identificador capturado, o el nombre completo si la regex no casa
        (en ese caso la imagen forma su propio grupo unitario y el split se
        comporta como uno aleatorio para ella).
    """
    coincidencia = re.match(patron, nombre)
    return coincidencia.group(1) if coincidencia else nombre


def construir_inventario(
    directorio: str | Path,
    config: dict[str, Any],
    forzar_estructura: str | None = None,
) -> Inventario:
    """Recorre un directorio y cataloga sus imagenes con sus etiquetas.

    Args:
        directorio: Ruta del dataset, relativa a la raiz del repo o absoluta.
        config: Configuracion del proyecto.
        forzar_estructura: ``"subcarpetas"`` o ``"csv"`` para saltarse la
            deteccion automatica. ``None`` para detectar.

    Returns:
        Inventario con rutas, etiquetas y (si procede) grupos de origen.

    Raises:
        ValueError: Si no se encuentra ninguna imagen valida.
    """
    raiz = resolver(directorio)
    estructura = forzar_estructura or detectar_estructura(raiz, config)
    clases: list[str] = obtener(config, "datos.clases", ["Negative", "Positive"])
    extensiones = {e.lower() for e in obtener(config, "datos.extensiones", [".jpg"])}
    mapa = _mapa_alias_a_indice(config)

    rutas: list[Path] = []
    etiquetas: list[int] = []

    if estructura == "subcarpetas":
        for subdir in sorted(p for p in raiz.iterdir() if p.is_dir()):
            indice = mapa.get(_normalizar(subdir.name))
            if indice is None:
                # Carpeta ajena al esquema de clases (p. ej. '.ipynb_checkpoints').
                continue
            for archivo in sorted(subdir.rglob("*")):
                if archivo.is_file() and archivo.suffix.lower() in extensiones:
                    rutas.append(archivo)
                    etiquetas.append(indice)
    else:
        nombre_csv = obtener(config, "datos.csv.nombre_archivo", "labels.csv")
        ruta_csv = raiz / nombre_csv
        if not ruta_csv.is_file():
            ruta_csv = next(iter(raiz.glob("*.csv")))
        tabla = pd.read_csv(ruta_csv)

        col_img = obtener(config, "datos.csv.columna_imagen", "filename")
        col_lbl = obtener(config, "datos.csv.columna_etiqueta", "label")
        faltantes = [c for c in (col_img, col_lbl) if c not in tabla.columns]
        if faltantes:
            raise ValueError(
                f"El CSV {ruta_csv} no tiene la(s) columna(s) {faltantes}. "
                f"Columnas disponibles: {list(tabla.columns)}. "
                "Ajusta 'datos.csv' en config.yaml."
            )

        for _, fila in tabla.iterrows():
            candidata = raiz / str(fila[col_img])
            if not candidata.is_file():
                # El CSV puede traer solo el nombre; se busca recursivamente.
                encontradas = list(raiz.rglob(Path(str(fila[col_img])).name))
                if not encontradas:
                    continue
                candidata = encontradas[0]
            indice = mapa.get(_normalizar(str(fila[col_lbl])))
            if indice is None:
                continue
            rutas.append(candidata)
            etiquetas.append(indice)

    if not rutas:
        raise ValueError(
            f"No se encontro ninguna imagen valida en {raiz} "
            f"(estructura detectada: {estructura}, extensiones: {sorted(extensiones)})."
        )

    grupos = None
    if obtener(config, "datos.split.agrupar_por_origen", False):
        patron = obtener(config, "datos.split.regex_origen", r"^(.*)$")
        grupos = np.array([_extraer_grupo(p.stem, patron) for p in rutas])

    return Inventario(
        rutas=rutas,
        etiquetas=np.asarray(etiquetas, dtype=np.int32),
        clases=clases,
        grupos=grupos,
        estructura=estructura,
        metadatos={"directorio": str(raiz)},
    )


# --------------------------------------------------------------------------- #
# Particion
# --------------------------------------------------------------------------- #


def submuestrear(inventario: Inventario, fraccion: float, semilla: int) -> Inventario:
    """Reduce el inventario a una fraccion estratificada del total.

    Sirve para validar el pipeline completo en minutos antes de lanzar la
    corrida larga en CPU: se conserva la proporcion de clases para que las
    metricas del ensayo sean interpretables.

    Args:
        inventario: Inventario completo.
        fraccion: Fraccion en ``(0, 1]``. Si es >= 1 se devuelve el original.
        semilla: Semilla del muestreo.

    Returns:
        Inventario reducido.
    """
    if fraccion >= 1.0:
        return inventario

    generador = np.random.default_rng(semilla)
    seleccion: list[int] = []
    for indice in range(len(inventario.clases)):
        posiciones = np.flatnonzero(inventario.etiquetas == indice)
        n = max(1, int(round(len(posiciones) * fraccion)))
        seleccion.extend(generador.choice(posiciones, size=n, replace=False).tolist())

    seleccion.sort()
    return Inventario(
        rutas=[inventario.rutas[i] for i in seleccion],
        etiquetas=inventario.etiquetas[seleccion],
        clases=inventario.clases,
        grupos=None if inventario.grupos is None else inventario.grupos[seleccion],
        estructura=inventario.estructura,
        metadatos={**inventario.metadatos, "submuestreo": fraccion},
    )


def _subinventario(inventario: Inventario, indices: np.ndarray) -> Inventario:
    """Crea un inventario con el subconjunto de indices dado.

    Args:
        inventario: Inventario origen.
        indices: Posiciones a conservar.

    Returns:
        Nuevo inventario.
    """
    return Inventario(
        rutas=[inventario.rutas[i] for i in indices],
        etiquetas=inventario.etiquetas[indices],
        clases=inventario.clases,
        grupos=None if inventario.grupos is None else inventario.grupos[indices],
        estructura=inventario.estructura,
        metadatos=inventario.metadatos,
    )


def dividir(
    inventario: Inventario, config: dict[str, Any], verboso: bool = True
) -> dict[str, Inventario]:
    """Divide el inventario en train/val/test de forma estratificada.

    Si ``datos.split.agrupar_por_origen`` esta activo, la particion respeta los
    grupos: todos los parches de una misma superficie caen en la misma
    particion. Esto sacrifica algo de estratificacion exacta a cambio de
    eliminar la **fuga de datos**, que es el problema real cuando el dataset son
    recortes de un numero pequeno de fotografias madre.

    Args:
        inventario: Inventario completo.
        config: Configuracion del proyecto.
        verboso: Imprime el reparto resultante y advertencias.

    Returns:
        Diccionario con las claves ``"train"``, ``"val"`` y ``"test"``.

    Raises:
        ValueError: Si las fracciones del split no suman 1.0.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    fr_train = float(obtener(config, "datos.split.train", 0.70))
    fr_val = float(obtener(config, "datos.split.val", 0.15))
    fr_test = float(obtener(config, "datos.split.test", 0.15))
    semilla = int(obtener(config, "proyecto.semilla", 42))

    if abs(fr_train + fr_val + fr_test - 1.0) > 1e-6:
        raise ValueError(
            f"Las fracciones del split deben sumar 1.0 "
            f"(train={fr_train}, val={fr_val}, test={fr_test})."
        )

    indices = np.arange(len(inventario))

    if inventario.grupos is not None:
        # Particion por grupos: primero se aparta (val + test), luego se parte
        # ese resto en dos respetando de nuevo los grupos.
        gss1 = GroupShuffleSplit(n_splits=1, test_size=fr_val + fr_test, random_state=semilla)
        idx_train, idx_resto = next(gss1.split(indices, inventario.etiquetas, inventario.grupos))

        proporcion_test = fr_test / (fr_val + fr_test)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=proporcion_test, random_state=semilla)
        rel_val, rel_test = next(
            gss2.split(idx_resto, inventario.etiquetas[idx_resto], inventario.grupos[idx_resto])
        )
        idx_val, idx_test = idx_resto[rel_val], idx_resto[rel_test]
    else:
        idx_train, idx_resto = train_test_split(
            indices,
            test_size=fr_val + fr_test,
            random_state=semilla,
            stratify=inventario.etiquetas,
        )
        idx_val, idx_test = train_test_split(
            idx_resto,
            test_size=fr_test / (fr_val + fr_test),
            random_state=semilla,
            stratify=inventario.etiquetas[idx_resto],
        )

    # Se mezcla el ORDEN de cada particion con una permutacion sembrada, en vez
    # de ordenar los indices.
    #
    # Por que esto es critico y no cosmetico: construir_inventario() recorre las
    # subcarpetas en orden alfabetico, asi que el inventario queda como un bloque
    # de 'Negative' seguido de un bloque de 'Positive'. Ordenar los indices con
    # np.sort() reconstruia ese orden por clase dentro de cada particion. Como el
    # buffer de tf.data (preproceso.barajar_buffer) solo mezcla dentro de una
    # ventana deslizante, con 40.000 imagenes y un buffer de 2.000 el modelo
    # recibia ~750 lotes seguidos de una sola clase y luego ~520 de la otra:
    # jamas veia un lote mezclado, y colapsaba a predecir siempre la ultima clase
    # que habia visto (val_auc = 0.5).
    #
    # Barajar aqui, sobre la lista de rutas, cuesta microsegundos y hace que el
    # buffer de tf.data solo tenga que aportar variacion entre epocas, que es
    # para lo que sirve.
    generador_orden = np.random.default_rng(semilla)

    def _mezclar(indices: np.ndarray) -> np.ndarray:
        """Permuta los indices de una particion de forma determinista.

        Args:
            indices: Indices de la particion.

        Returns:
            Los mismos indices en orden pseudoaleatorio reproducible.
        """
        return np.asarray(indices)[generador_orden.permutation(len(indices))]

    particiones = {
        "train": _subinventario(inventario, _mezclar(idx_train)),
        "val": _subinventario(inventario, _mezclar(idx_val)),
        "test": _subinventario(inventario, _mezclar(idx_test)),
    }

    if verboso:
        print(f"  Estructura detectada : {inventario.estructura}")
        print(f"  Imagenes totales     : {len(inventario)}")
        for nombre, parte in particiones.items():
            conteo = parte.conteo_por_clase()
            porcentaje = 100.0 * len(parte) / max(len(inventario), 1)
            detalle = "  ".join(f"{c}={n}" for c, n in conteo.items())
            print(f"  {nombre:<6}: {len(parte):>6} ({porcentaje:4.1f}%)   {detalle}")

        if inventario.grupos is None:
            print(
                "  AVISO: split aleatorio. Si el dataset son recortes de una misma\n"
                "         superficie, hay FUGA DE DATOS y las metricas quedaran\n"
                "         infladas. Activa 'datos.split.agrupar_por_origen: true'\n"
                "         en config.yaml y ajusta 'regex_origen'."
            )
        else:
            n_grupos = len(np.unique(inventario.grupos))
            print(f"  Split agrupado por origen: {n_grupos} superficies distintas.")

    return particiones


def calcular_pesos_clase(inventario: Inventario) -> dict[int, float]:
    """Calcula pesos de clase inversamente proporcionales a su frecuencia.

    Compensa el desbalance para que el modelo no aprenda a ignorar la clase
    minoritaria. En este dominio importa especialmente: la clase minoritaria
    suele ser "con grieta", que es justo la que no se puede fallar.

    Args:
        inventario: Inventario del conjunto de entrenamiento.

    Returns:
        Diccionario ``{indice_clase: peso}`` listo para ``model.fit(class_weight=...)``.
    """
    from sklearn.utils.class_weight import compute_class_weight

    presentes = np.unique(inventario.etiquetas)
    pesos = compute_class_weight("balanced", classes=presentes, y=inventario.etiquetas)
    return {int(c): float(w) for c, w in zip(presentes, pesos)}


# --------------------------------------------------------------------------- #
# Pipeline tf.data
# --------------------------------------------------------------------------- #


def construir_capa_degradacion(factor_maximo: float, probabilidad: float) -> Any:
    """Crea una capa que simula la perdida de detalle del redimensionado.

    El problema que ataca
    ---------------------
    §4.6 del informe demuestra que la brecha entre el conjunto de prueba y las
    fotografias propias es en buena parte un desajuste de **escala**: los parches
    de entrenamiento se reducen 1.4 veces al entrar al modelo y una foto de
    telefono se reduce 7.5 veces. Una fisura de 3 px queda en 0.4 px y desaparece.

    El analisis por mosaicos corrige eso en inferencia, a cambio de multiplicar
    por 2.4 el coste. Esta capa ataca la misma causa desde el otro lado: si
    durante el entrenamiento el modelo ve grietas ya degradadas por una reduccion
    fuerte, aprende a reconocerlas asi, y quiza deje de hacer falta trocear.

    Como funciona
    -------------
    Reduce la imagen por un factor aleatorio y la devuelve a su tamano original.
    El viaje de ida y vuelta es lo que importa: el tensor conserva su forma, pero
    ha perdido irreversiblemente el detalle fino, exactamente igual que una
    fotografia grande reducida a 160 px.

    Limitacion conocida: el factor se sortea **una vez por lote**, no por imagen.
    Hacerlo por imagen exigiria ``tf.map_fn`` con formas dinamicas, que en CPU
    cuesta mas que el propio entrenamiento. La consecuencia es que las muestras
    de un mismo lote comparten degradacion; con lotes barajados y muchas epocas,
    cada imagen acaba viendo un abanico amplio de factores igualmente.

    Args:
        factor_maximo: Reduccion maxima a simular. Con 4.0 una imagen de 160 px
            puede llegar a verse como si midiera 40 px.
        probabilidad: Fraccion de lotes que se degradan. El resto pasa intacto,
            para que el modelo no olvide como es una grieta bien resuelta.

    Returns:
        Una capa de Keras aplicable dentro del ``Sequential`` de aumento.
    """
    import tensorflow as tf

    class DegradacionEscala(tf.keras.layers.Layer):
        """Reduce y restaura la imagen para simular perdida de resolucion."""

        def __init__(self, factor_maximo: float, probabilidad: float, **kwargs: Any) -> None:
            """Guarda los parametros de la degradacion.

            Args:
                factor_maximo: Reduccion maxima a simular.
                probabilidad: Fraccion de lotes que se degradan.
                **kwargs: Argumentos de ``tf.keras.layers.Layer``.
            """
            super().__init__(**kwargs)
            self.factor_maximo = float(factor_maximo)
            self.probabilidad = float(probabilidad)

        def call(self, entradas: Any, training: bool | None = None) -> Any:
            """Aplica la degradacion solo durante el entrenamiento.

            Args:
                entradas: Lote ``(N, H, W, C)``.
                training: Bandera de fase de Keras.

            Returns:
                El lote, degradado o intacto, con la misma forma.
            """
            if not training or self.factor_maximo <= 1.0 or self.probabilidad <= 0.0:
                return entradas

            forma = tf.shape(entradas)
            alto, ancho = forma[1], forma[2]

            def degradar() -> Any:
                factor = tf.random.uniform([], 1.0, self.factor_maximo)
                reducido = tf.maximum(
                    tf.cast(tf.cast(tf.stack([alto, ancho]), tf.float32) / factor, tf.int32), 8
                )
                pequena = tf.image.resize(entradas, reducido, method="bilinear")
                return tf.image.resize(pequena, tf.stack([alto, ancho]), method="bilinear")

            return tf.cond(
                tf.random.uniform([]) < self.probabilidad, degradar, lambda: entradas
            )

        def get_config(self) -> dict[str, Any]:
            """Permite serializar la capa junto con el modelo.

            Returns:
                Configuracion reconstruible de la capa.
            """
            return {
                **super().get_config(),
                "factor_maximo": self.factor_maximo,
                "probabilidad": self.probabilidad,
            }

    return DegradacionEscala(factor_maximo, probabilidad, name="degradacion_escala")


def construir_capa_aumento(config: dict[str, Any]) -> Any:
    """Construye la capa de aumento de datos como un ``Sequential`` de Keras.

    Se implementa con capas de Keras (y no con ``tf.image`` sueltas) para que el
    aumento sea parte del grafo, se ejecute en el dispositivo y quede desactivado
    automaticamente en inferencia (``training=False``).

    El volteo vertical esta permitido, a diferencia de lo habitual en imagen
    natural: una grieta en hormigon no tiene una orientacion "de pie" canonica,
    asi que invertirla produce una muestra fisicamente plausible.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Modelo ``Sequential`` con las transformaciones activas segun el YAML.
    """
    import tensorflow as tf

    aum = obtener(config, "preproceso.aumento", {})
    semilla = int(obtener(config, "proyecto.semilla", 42))
    capas: list[Any] = []

    modo_volteo = None
    if aum.get("volteo_horizontal", False) and aum.get("volteo_vertical", False):
        modo_volteo = "horizontal_and_vertical"
    elif aum.get("volteo_horizontal", False):
        modo_volteo = "horizontal"
    elif aum.get("volteo_vertical", False):
        modo_volteo = "vertical"
    if modo_volteo:
        capas.append(tf.keras.layers.RandomFlip(modo_volteo, seed=semilla))

    if aum.get("rotacion", 0):
        capas.append(tf.keras.layers.RandomRotation(float(aum["rotacion"]), seed=semilla))
    if aum.get("zoom", 0):
        capas.append(tf.keras.layers.RandomZoom(float(aum["zoom"]), seed=semilla))
    if aum.get("traslacion", 0):
        t = float(aum["traslacion"])
        capas.append(tf.keras.layers.RandomTranslation(t, t, seed=semilla))
    # La degradacion va ANTES que contraste y brillo, y el orden importa: simula
    # una camara que capturo la escena a menor resolucion, y los ajustes
    # fotometricos ocurren sobre lo que la camara entrego, no antes.
    degradacion = aum.get("degradacion_escala", {}) or {}
    if degradacion.get("activo", False):
        capas.append(
            construir_capa_degradacion(
                float(degradacion.get("factor_maximo", 4.0)),
                float(degradacion.get("probabilidad", 0.5)),
            )
        )

    if aum.get("contraste", 0):
        capas.append(tf.keras.layers.RandomContrast(float(aum["contraste"]), seed=semilla))
    if aum.get("brillo", 0):
        # value_range=(0, 1) porque el pipeline entrega tensores ya normalizados.
        capas.append(
            tf.keras.layers.RandomBrightness(
                float(aum["brillo"]), value_range=(0.0, 1.0), seed=semilla
            )
        )

    return tf.keras.Sequential(capas, name="aumento_datos")


def _decodificar(ruta: Any, etiqueta: Any, alto: int, ancho: int, canales: int) -> tuple[Any, Any]:
    """Lee un archivo de imagen y lo convierte en tensor normalizado.

    Args:
        ruta: Tensor escalar de tipo string con la ruta del archivo.
        etiqueta: Tensor escalar con la etiqueta entera.
        alto: Alto de destino en pixeles.
        ancho: Ancho de destino en pixeles.
        canales: Numero de canales (3 = RGB).

    Returns:
        Par ``(imagen, etiqueta)`` donde imagen es ``float32`` de forma
        ``(alto, ancho, canales)`` en el rango ``[0, 1]``.
    """
    import tensorflow as tf

    bytes_imagen = tf.io.read_file(ruta)
    # expand_animations=False evita que un GIF devuelva un tensor 4D y rompa el
    # resize; decode_image cubre jpg/png/bmp/gif con una sola llamada.
    imagen = tf.io.decode_image(bytes_imagen, channels=canales, expand_animations=False)
    imagen = tf.image.resize(imagen, [alto, ancho], method="bilinear")
    imagen = tf.cast(imagen, tf.float32) / 255.0
    imagen.set_shape([alto, ancho, canales])
    return imagen, tf.cast(etiqueta, tf.float32)


def pasos_por_epoca(inventario: Inventario, batch_size: int) -> int:
    """Calcula cuantos lotes componen una epoca completa.

    Hace falta porque ``ignore_errors()`` deja el dataset con cardinalidad
    desconocida y Keras no puede deducir el numero de pasos por su cuenta.

    Args:
        inventario: Particion sobre la que se itera.
        batch_size: Tamano de lote.

    Returns:
        Numero de pasos, al menos 1.
    """
    return max(1, math.ceil(len(inventario) / max(batch_size, 1)))


def crear_dataset(
    inventario: Inventario,
    config: dict[str, Any],
    entrenamiento: bool = False,
    batch_size: int | None = None,
    barajar: bool | None = None,
    repetir: bool | None = None,
) -> Any:
    """Construye un ``tf.data.Dataset`` a partir de un inventario.

    Orden del pipeline y por que ese orden:

    ``map(decodificar) -> ignore_errors -> cache -> shuffle -> batch ->
    aumento -> repeat -> prefetch``

    - ``cache()`` va **antes** del ``shuffle()``. Es el punto que se suele
      equivocar: si se cachea despues de barajar, el orden barajado queda
      congelado en la cache y a partir de la segunda epoca el modelo ve siempre
      la misma secuencia, con lo que el barajado deja de servir para nada.
      Cacheando antes se guarda el trabajo caro y determinista (decodificar y
      redimensionar) y el barajado sigue siendo nuevo en cada epoca.
    - El **aumento va despues del ``batch()``**: aplicar las capas de Keras a un
      lote entero aprovecha la vectorizacion, y como va despues de la cache, las
      transformaciones son distintas en cada epoca.
    - ``ignore_errors()`` evita que una imagen corrupta o truncada aborte un
      entrenamiento de horas. El precio es que la cardinalidad pasa a ser
      desconocida, y por eso los scripts de entrenamiento pasan
      ``steps_per_epoch`` calculado con :func:`pasos_por_epoca`.

    Args:
        inventario: Conjunto de imagenes y etiquetas.
        config: Configuracion del proyecto.
        entrenamiento: Si ``True``, aplica aumento de datos. **Solo** debe ser
            ``True`` para la particion de entrenamiento: aumentar validacion o
            prueba falsea la evaluacion.
        batch_size: Tamano de lote. Si es ``None``, se toma del YAML.
        barajar: Si se baraja el conjunto. Por defecto coincide con
            ``entrenamiento``.
        repetir: Si el dataset se repite indefinidamente. Por defecto coincide
            con ``entrenamiento``; se combina con ``steps_per_epoch``.

    Returns:
        Dataset por lotes, cacheado y con prefetch, listo para ``model.fit``.
    """
    import tensorflow as tf

    alto, ancho, canales = forma_entrada(config)
    lote = int(batch_size or obtener(config, "entrenamiento.batch_size", 32))
    barajar = entrenamiento if barajar is None else barajar
    repetir = entrenamiento if repetir is None else repetir
    semilla = int(obtener(config, "proyecto.semilla", 42))
    autotune = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in inventario.rutas], inventario.etiquetas)
    )
    ds = ds.map(
        lambda r, e: _decodificar(r, e, alto, ancho, canales),
        num_parallel_calls=autotune,
    )
    ds = ds.ignore_errors()

    if obtener(config, "preproceso.cache", True):
        ds = ds.cache()

    if barajar:
        buffer = min(int(obtener(config, "preproceso.barajar_buffer", 2000)), len(inventario))
        ds = ds.shuffle(max(buffer, 1), seed=semilla, reshuffle_each_iteration=True)

    ds = ds.batch(lote)

    if entrenamiento and obtener(config, "preproceso.aumento.activo", True):
        aumento = construir_capa_aumento(config)
        ds = ds.map(lambda x, y: (aumento(x, training=True), y), num_parallel_calls=autotune)

    if repetir:
        ds = ds.repeat()

    if obtener(config, "preproceso.prefetch", True):
        ds = ds.prefetch(autotune)

    return ds


def cargar_particiones(
    config: dict[str, Any],
    subset: float | None = None,
    batch_size: int | None = None,
    verboso: bool = True,
) -> tuple[dict[str, Any], dict[str, Inventario]]:
    """Punto de entrada unico: del directorio de datos a los tres ``tf.data``.

    Args:
        config: Configuracion del proyecto.
        subset: Fraccion del dataset a usar, en ``(0, 1]``. ``None`` = todo.
        batch_size: Sobrescribe el tamano de lote del YAML.
        verboso: Imprime el reparto y las advertencias de fuga de datos.

    Returns:
        Tupla ``(datasets, inventarios)``. ``datasets`` tiene las claves
        ``train``/``val``/``test`` con objetos ``tf.data.Dataset``;
        ``inventarios`` las mismas claves con los objetos :class:`Inventario`
        (necesarios para recuperar etiquetas verdaderas en la evaluacion).
    """
    semilla = int(obtener(config, "proyecto.semilla", 42))
    inventario = construir_inventario(obtener(config, "rutas.datos_raw"), config)

    if subset is not None and subset < 1.0:
        inventario = submuestrear(inventario, subset, semilla)
        if verboso:
            print(f"  Submuestreo activo   : {subset:.0%} -> {len(inventario)} imagenes")

    inventarios = dividir(inventario, config, verboso=verboso)
    datasets = {
        # train y val se repiten indefinidamente y el script les pasa
        # steps_per_epoch / validation_steps calculados con pasos_por_epoca().
        # Es lo que evita el aviso "your input ran out of data" que produce la
        # cardinalidad desconocida de ignore_errors().
        "train": crear_dataset(
            inventarios["train"], config, entrenamiento=True, batch_size=batch_size
        ),
        "val": crear_dataset(
            inventarios["val"], config, entrenamiento=False, batch_size=batch_size, repetir=True
        ),
        # test NO se repite: se recorre una sola vez para evaluar, y una
        # repeticion infinita colgaria el bucle de prediccion.
        "test": crear_dataset(
            inventarios["test"], config, entrenamiento=False, batch_size=batch_size, repetir=False
        ),
    }
    return datasets, inventarios


def cargar_propias(
    config: dict[str, Any], batch_size: int | None = None
) -> tuple[Any, Inventario] | tuple[None, None]:
    """Carga las fotografias propias del equipo como conjunto de prueba final.

    Este conjunto **nunca** se usa para entrenar ni para elegir hiperparametros:
    es la unica medida honesta de generalizacion fuera de la distribucion del
    dataset publico, y es lo que exige el docente.

    Args:
        config: Configuracion del proyecto.
        batch_size: Tamano de lote. ``None`` toma el del YAML.

    Returns:
        Tupla ``(dataset, inventario)``, o ``(None, None)`` si el directorio no
        existe o esta vacio. No lanza excepcion: es legitimo entrenar antes de
        haber tomado las fotos.
    """
    directorio = resolver(obtener(config, "rutas.datos_propias", "data/propias"))
    if not directorio.is_dir():
        return None, None
    try:
        inventario = construir_inventario(directorio, config)
    except (ValueError, FileNotFoundError):
        return None, None
    if len(inventario) == 0:
        return None, None

    ds = crear_dataset(inventario, config, entrenamiento=False, batch_size=batch_size)
    return ds, inventario
