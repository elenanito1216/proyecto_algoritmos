"""Captura desde webcam y estabilizacion temporal para el analisis en vivo.

Por que existe este modulo
--------------------------
El pipeline de la aplicacion esta pensado para una fotografia: se sube, se
analiza y se muestra un resultado. En video ese mismo pipeline produce una
salida **inservible por parpadeo**: la probabilidad de grieta oscila varias
decimas entre fotogramas consecutivos casi identicos, el angulo de desaplome
salta, y el semaforo de riesgo cambia de color varias veces por segundo.

La causa no es ruido del modelo sino ruido del sensor: el autoenfoque, el
balance de blancos automatico y el ruido de lectura hacen que dos fotogramas
seguidos no sean la misma imagen. Un clasificador que responde 0.48 y 0.52
alternativamente esta siendo coherente; el problema es que 0.50 es el umbral.

La solucion es **agregacion temporal**: en lugar de decidir con el fotograma
actual, se decide con la mediana de los ultimos N. Se usa la mediana y no la
media por la misma razon que en la inclinometria (``src/vision/inclinacion.py``):
un fotograma movido o sobreexpuesto es un valor atipico, y la media se lo traga
mientras que la mediana lo ignora.

Separacion de responsabilidades
-------------------------------
Todo lo que se puede probar sin una webcam conectada -suavizado, recorte,
medicion de tasa de fotogramas- vive aqui como funciones y clases puras. La
apertura del dispositivo se aisla al final del modulo.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np


class SuavizadorTemporal:
    """Mantiene una ventana deslizante de valores y devuelve su mediana.

    Se usa para estabilizar la probabilidad de grieta y el angulo de desaplome
    en el analisis de video. Acepta ``None`` como valor ausente -por ejemplo
    cuando la inclinometria no encuentra ningun elemento vertical- y lo excluye
    de la ventana en lugar de contaminarla con un cero.

    Attributes:
        ventana: Numero maximo de valores que se conservan.
    """

    def __init__(self, ventana: int = 7) -> None:
        """Inicializa el suavizador.

        Args:
            ventana: Tamano de la ventana deslizante. Valores tipicos: 5 a 9.
                Mas grande estabiliza mas pero anade retardo perceptible al
                mover la camara.

        Raises:
            ValueError: Si la ventana es menor que 1.
        """
        if ventana < 1:
            raise ValueError(f"La ventana debe ser >= 1; se recibio {ventana}.")
        self.ventana = int(ventana)
        self._valores: deque[float] = deque(maxlen=self.ventana)

    def agregar(self, valor: float | None) -> float | None:
        """Anade un valor a la ventana y devuelve la mediana actualizada.

        Args:
            valor: Nuevo valor observado, o ``None`` si la medida no esta
                disponible en este fotograma.

        Returns:
            Mediana de los valores acumulados, o ``None`` si la ventana esta
            vacia porque aun no se ha observado ningun valor valido.
        """
        if valor is not None and np.isfinite(valor):
            self._valores.append(float(valor))
        return self.valor()

    def valor(self) -> float | None:
        """Devuelve la mediana actual sin modificar la ventana.

        Returns:
            Mediana de la ventana, o ``None`` si esta vacia.
        """
        if not self._valores:
            return None
        return float(np.median(self._valores))

    def estabilidad(self) -> float:
        """Mide cuanto varian los valores de la ventana.

        Es el analogo temporal de la dispersion espacial que ya reporta la
        inclinometria: si la ventana esta llena de valores dispares, la lectura
        no es de fiar aunque su mediana parezca razonable.

        Returns:
            Desviacion absoluta mediana de la ventana. Cero si hay menos de dos
            valores.
        """
        if len(self._valores) < 2:
            return 0.0
        arreglo = np.asarray(self._valores, dtype=float)
        return float(np.median(np.abs(arreglo - np.median(arreglo))))

    def lleno(self) -> bool:
        """Indica si la ventana ya acumulo su capacidad completa.

        Returns:
            ``True`` cuando hay tantos valores como el tamano de la ventana.
        """
        return len(self._valores) == self.ventana

    def __len__(self) -> int:
        """Numero de valores validos acumulados en la ventana.

        Returns:
            Cantidad de muestras, entre 0 y ``ventana``.
        """
        return len(self._valores)

    def reiniciar(self) -> None:
        """Vacia la ventana. Se llama al cambiar de modelo o de camara."""
        self._valores.clear()


class MedidorFPS:
    """Calcula la tasa de fotogramas sobre una ventana deslizante.

    Se reporta en la interfaz porque es la demostracion mas directa del
    argumento de eficiencia del proyecto: con el TFLite int8 el sistema sostiene
    video fluido y con el modelo Keras no.
    """

    def __init__(self, ventana: int = 30) -> None:
        """Inicializa el medidor.

        Args:
            ventana: Cuantas duraciones de fotograma se promedian.
        """
        self._duraciones: deque[float] = deque(maxlen=max(int(ventana), 1))
        self._ultimo: float | None = None

    def marcar(self) -> None:
        """Registra el instante actual como final de un fotograma."""
        ahora = time.perf_counter()
        if self._ultimo is not None:
            self._duraciones.append(ahora - self._ultimo)
        self._ultimo = ahora

    def fps(self) -> float:
        """Devuelve la tasa de fotogramas estimada.

        Returns:
            Fotogramas por segundo, o 0.0 si aun no hay suficientes muestras.
        """
        if not self._duraciones:
            return 0.0
        medio = float(np.mean(self._duraciones))
        return 1.0 / medio if medio > 0 else 0.0

    def ms_por_fotograma(self) -> float:
        """Devuelve la duracion media de un fotograma en milisegundos.

        Returns:
            Milisegundos por fotograma, o 0.0 si no hay muestras.
        """
        if not self._duraciones:
            return 0.0
        return float(np.mean(self._duraciones)) * 1000.0


class LectorAsincrono:
    """Lee una fuente de video en un hilo aparte y conserva solo el ultimo fotograma.

    El problema que resuelve
    ------------------------
    Un stream de red entrega fotogramas a su propio ritmo (~30 FPS) mientras que
    el bucle de analisis los consume mas despacio. La diferencia **no se pierde:
    se acumula** en el bufer de FFMPEG. Cada segundo de desfase mete quince o
    veinte fotogramas atrasados en la cola, y el retardo entre mover el telefono
    y verlo en pantalla crece sin limite: a los treinta segundos van varios
    segundos por detras.

    ``CAP_PROP_BUFFERSIZE`` deberia evitarlo, pero la mayoria de backends lo
    ignoran en fuentes de red.

    La solucion es desacoplar produccion de consumo. Este hilo lee sin descanso y
    **descarta** todo lo que no sea el fotograma mas reciente; el bucle principal
    toma siempre el ultimo disponible. El retardo pasa a estar acotado por la red
    mas un fotograma, en lugar de crecer con el tiempo.

    Es el mismo criterio que el resto del proyecto aplica a los datos viejos: no
    interesa procesarlos, interesa procesar el actual.
    """

    def __init__(self, captura: Any) -> None:
        """Arranca el hilo lector sobre una captura ya abierta.

        Args:
            captura: Objeto ``cv2.VideoCapture`` abierto.
        """
        self._captura = captura
        self._cerrojo = threading.Lock()
        self._fotograma: np.ndarray | None = None
        self._instante: float = 0.0
        self._fallos = 0
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _bucle(self) -> None:
        """Lee sin parar y guarda unicamente el fotograma mas reciente."""
        while self._activo:
            leido, fotograma = self._captura.read()
            if not leido or fotograma is None:
                self._fallos += 1
                # Un stream de red puede tener cortes puntuales; solo se
                # abandona si son persistentes.
                if self._fallos > 30:
                    break
                time.sleep(0.01)
                continue
            self._fallos = 0
            with self._cerrojo:
                self._fotograma = fotograma
                self._instante = time.perf_counter()

    def leer(self) -> tuple[bool, np.ndarray | None, float]:
        """Devuelve el fotograma mas reciente y su antiguedad.

        Returns:
            Tupla ``(hay_fotograma, fotograma, antiguedad_ms)``. La antiguedad es
            el tiempo transcurrido desde que el hilo lo capturo, y es la medida
            honesta del retardo: si crece, el consumo va por detras de la fuente.
        """
        with self._cerrojo:
            if self._fotograma is None:
                return False, None, 0.0
            edad = (time.perf_counter() - self._instante) * 1000.0
            return True, self._fotograma.copy(), edad

    def esta_vivo(self) -> bool:
        """Indica si el hilo lector sigue en marcha.

        Returns:
            ``True`` mientras el hilo lea correctamente.
        """
        return self._activo and self._hilo.is_alive()

    def detener(self) -> None:
        """Para el hilo y libera el dispositivo."""
        self._activo = False
        if self._hilo.is_alive():
            self._hilo.join(timeout=2.0)
        # suppress: liberar el dispositivo es limpieza; si falla, no debe
        # impedir que el usuario detenga la camara.
        with contextlib.suppress(Exception):
            self._captura.release()

    def isOpened(self) -> bool:  # noqa: N802 - imita la interfaz de cv2.VideoCapture
        """Compatibilidad con la interfaz de ``cv2.VideoCapture``.

        Returns:
            ``True`` si la captura subyacente esta abierta.
        """
        return bool(self._captura.isOpened())


def recortar_centro(imagen: np.ndarray, fraccion: float) -> np.ndarray:
    """Recorta la region central cuadrada de un fotograma.

    Es necesario por un desajuste de dominio que no es evidente: el modelo se
    entreno con **parches en primer plano** de una superficie, mientras que una
    webcam entrega una vista amplia de la habitacion. Redimensionar el fotograma
    completo a 160x160 deja cualquier grieta reducida a dos o tres pixeles y
    ademas deforma la relacion de aspecto.

    Recortando el centro se acerca el encuadre al del entrenamiento y se le da
    al usuario un objetivo claro donde apuntar.

    Args:
        imagen: Fotograma ``(H, W, C)``.
        fraccion: Lado del recorte como fraccion del lado menor, en ``(0, 1]``.

    Returns:
        Recorte cuadrado centrado. Si ``fraccion`` es >= 1 se devuelve el
        cuadrado central completo.

    Raises:
        ValueError: Si la fraccion no es positiva.
    """
    if fraccion <= 0:
        raise ValueError(f"La fraccion debe ser > 0; se recibio {fraccion}.")

    alto, ancho = imagen.shape[:2]
    lado = max(int(min(alto, ancho) * min(fraccion, 1.0)), 1)

    y0 = (alto - lado) // 2
    x0 = (ancho - lado) // 2
    return imagen[y0 : y0 + lado, x0 : x0 + lado]


def rectangulo_centro(imagen: np.ndarray, fraccion: float) -> tuple[int, int, int, int]:
    """Calcula el rectangulo que :func:`recortar_centro` extraeria.

    Sirve para dibujarlo sobre el fotograma mostrado, de modo que el usuario vea
    exactamente que region esta analizando el modelo en lugar de adivinarlo.

    Args:
        imagen: Fotograma ``(H, W, C)``.
        fraccion: Misma fraccion que se pasaria a :func:`recortar_centro`.

    Returns:
        Tupla ``(x0, y0, x1, y1)`` en coordenadas de la imagen.
    """
    alto, ancho = imagen.shape[:2]
    lado = max(int(min(alto, ancho) * min(max(fraccion, 1e-6), 1.0)), 1)
    x0 = (ancho - lado) // 2
    y0 = (alto - lado) // 2
    return x0, y0, x0 + lado, y0 + lado


def listar_camaras(max_indice: int = 3) -> list[int]:
    """Sondea que indices de camara responden en este equipo.

    OpenCV no ofrece enumeracion de dispositivos, asi que la unica via es
    intentar abrir cada indice. En Windows se fuerza el backend DirectShow
    (``CAP_DSHOW``): el predeterminado (MSMF) tarda varios segundos en fallar
    cuando el indice no existe, lo que haria el sondeo insoportablemente lento.

    Args:
        max_indice: Ultimo indice a probar (exclusivo).

    Returns:
        Lista de indices que se pudieron abrir y de los que se leyo un
        fotograma valido.
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    disponibles: list[int] = []

    for indice in range(max(int(max_indice), 0)):
        captura = cv2.VideoCapture(indice, backend)
        try:
            if captura.isOpened():
                leido, fotograma = captura.read()
                if leido and fotograma is not None:
                    disponibles.append(indice)
        finally:
            captura.release()

    return disponibles


def diagnosticar_fotograma(fotograma: np.ndarray | None, ms_lectura: float) -> str | None:
    """Detecta si la camara esta entregando video real o esta bloqueada.

    Existe por un fallo observado y diagnosticado en este proyecto. Cuando dos
    procesos abren la misma webcam en Windows -por ejemplo dos servidores de
    Streamlit a la vez, o la aplicacion y otra videollamada- el segundo **no
    recibe un error**: recibe fotogramas completamente negros a exactamente
    1 FPS. Medido en el equipo de desarrollo: con la camara libre, ``read()``
    tarda 31 ms (32 FPS); con otro proceso reteniendola, 1000 ms y brillo 0.

    Sin esta comprobacion el sintoma es indistinguible de "el modelo va lento",
    que es la conclusion equivocada y lleva a optimizar lo que no toca.

    Args:
        fotograma: Fotograma leido, o ``None`` si la lectura fallo.
        ms_lectura: Milisegundos que tardo ``read()``.

    Returns:
        Mensaje explicando el problema, o ``None`` si el fotograma parece sano.
    """
    if fotograma is None:
        return (
            "La camara no entrego ningun fotograma. Suele significar que otra "
            "aplicacion la tiene abierta."
        )

    brillo = float(fotograma.mean())

    if ms_lectura > 500 and brillo < 5.0:
        return (
            f"La camara entrega fotogramas negros a {1000 / max(ms_lectura, 1):.1f} FPS "
            f"(brillo {brillo:.1f}/255, lectura {ms_lectura:.0f} ms). Casi con seguridad "
            "**otro proceso la tiene abierta**: otra pestana con esta misma aplicacion, "
            "un segundo servidor de Streamlit, o una videollamada. Cierralos y vuelve a "
            "pulsar Iniciar. No es lentitud del modelo."
        )

    if brillo < 5.0:
        return (
            f"Los fotogramas llegan casi negros (brillo {brillo:.1f}/255). Comprueba el "
            "obturador de privacidad de la camara -muchos portatiles llevan una tapa "
            "deslizante- y la iluminacion de la sala."
        )

    if ms_lectura > 300:
        return (
            f"La captura va a {1000 / max(ms_lectura, 1):.1f} FPS ({ms_lectura:.0f} ms por "
            "fotograma). Con la camara libre deberian ser ~30 FPS: revisa si hay otra "
            "aplicacion usandola."
        )

    return None


def es_fuente_de_red(fuente: int | str) -> bool:
    """Indica si una fuente de video es una URL de red en vez de un dispositivo.

    Args:
        fuente: Indice de dispositivo, o URL del stream.

    Returns:
        ``True`` si es una cadena que parece una URL de stream.

    Example:
        >>> es_fuente_de_red(0)
        False
        >>> es_fuente_de_red("http://192.168.1.40:8080/video")
        True
    """
    if isinstance(fuente, int):
        return False
    texto = str(fuente).strip().lower()
    return texto.startswith(("http://", "https://", "rtsp://", "rtmp://"))


def normalizar_url_celular(texto: str, puerto_por_defecto: int = 8080) -> str:
    """Completa una URL de stream escrita a medias.

    Las aplicaciones de camara IP muestran en pantalla algo como
    ``192.168.1.40:8080`` y el usuario lo copia tal cual. Esta funcion acepta
    esa forma abreviada y la convierte en una URL utilizable, en lugar de
    fallar con un mensaje sobre protocolos.

    Args:
        texto: Lo que escribio el usuario.
        puerto_por_defecto: Puerto que se anade si no hay ninguno.

    Returns:
        URL completa. Cadena vacia si la entrada estaba vacia.

    Example:
        >>> normalizar_url_celular("192.168.1.40:8080")
        'http://192.168.1.40:8080/video'
        >>> normalizar_url_celular("http://192.168.1.40:4747/video")
        'http://192.168.1.40:4747/video'
    """
    texto = (texto or "").strip()
    if not texto:
        return ""

    if not texto.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
        texto = "http://" + texto

    resto = texto.split("://", 1)[1]
    if ":" not in resto.split("/", 1)[0]:
        anfitrion, _, camino = resto.partition("/")
        texto = f"{texto.split('://', 1)[0]}://{anfitrion}:{puerto_por_defecto}"
        if camino:
            texto += "/" + camino

    # Sin ruta, se asume la de IP Webcam, que es la aplicacion mas extendida.
    if texto.count("/") == 2:
        texto += "/video"

    return texto


def abrir_camara(
    fuente: int | str = 0, ancho: int = 640, alto: int = 480, timeout_ms: int = 5000
) -> Any:
    """Abre un dispositivo de captura local o un stream de video por red.

    Acepta dos tipos de fuente, y esa es la razon de ser de la funcion:

    - **Un indice entero**: la webcam del propio equipo, o una camara virtual
      instalada por un driver (Iriun, DroidCam), que aparece como un dispositivo
      mas.
    - **Una URL**: el stream MJPEG que publican las aplicaciones de camara IP
      del movil. Es la via para usar la camara del **celular** sin anadir
      ninguna dependencia de Python: OpenCV decodifica MJPEG sobre HTTP con el
      mismo ``VideoCapture``, y al capturar la app nativa del telefono no hace
      falta HTTPS, que es lo que bloquearia a un navegador movil.

    Se limita la resolucion a proposito: Canny y Hough son lineales en el numero
    de pixeles, y una camara en 1920x1080 multiplica por seis el trabajo del
    modulo clasico sin aportar ninguna recta que no estuviera ya en 640x480.

    Args:
        fuente: Indice del dispositivo local, o URL del stream de red.
        ancho: Ancho solicitado. Solo se aplica a dispositivos locales; en un
            stream de red la resolucion la decide quien emite.
        alto: Alto solicitado, con la misma salvedad.
        timeout_ms: Milisegundos de espera al abrir y al leer un stream de red.
            Sin este limite, una URL equivocada dejaria la interfaz colgada.

    Returns:
        Objeto ``cv2.VideoCapture``. Hay que comprobar ``isOpened()`` antes de
        usarlo: esta funcion no lanza excepcion si la fuente no responde,
        porque en la interfaz eso es un estado que se muestra, no un error.
    """
    if es_fuente_de_red(fuente):
        # Opciones de baja latencia para FFMPEG. Deben fijarse ANTES de crear la
        # captura, porque se leen al abrir el flujo:
        #   nobuffer  - no acumular paquetes antes de entregarlos
        #   low_delay - no esperar a poder reordenar fotogramas
        # Reducen el retardo en origen; el hilo lector se encarga del resto.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

        # FFMPEG es el backend que sabe de HTTP y MJPEG. DirectShow no.
        captura = cv2.VideoCapture(str(fuente), cv2.CAP_FFMPEG)
        # Estas dos propiedades son las que evitan que la app se quede colgada
        # esperando indefinidamente a una direccion que no contesta.
        captura.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms))
        captura.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms))
        if captura.isOpened():
            captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return captura

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    captura = cv2.VideoCapture(int(fuente), backend)

    if captura.isOpened():
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, int(ancho))
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, int(alto))
        # Buffer minimo: sin esto el driver acumula fotogramas y el video se ve
        # con varios segundos de retardo respecto a lo que apunta la camara.
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return captura
