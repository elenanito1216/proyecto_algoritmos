r"""Resolucion de rutas del proyecto.

Todo el codigo del repositorio construye sus rutas a traves de este modulo. El
objetivo es que el proyecto funcione igual en Windows, Linux y macOS y desde
cualquier directorio de trabajo, sin una sola ruta absoluta escrita a mano ni un
separador de Windows (``\``) incrustado en el codigo.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[2]


def resolver(ruta_relativa: str | Path) -> Path:
    """Convierte una ruta del ``config.yaml`` en una ruta absoluta del sistema.

    Las rutas absolutas se devuelven intactas, de modo que un usuario puede
    apuntar ``datos_raw`` a un disco externo sin tocar el codigo.

    Args:
        ruta_relativa: Ruta tal como aparece en la configuracion, con
            separadores ``/`` (estilo POSIX) o como objeto ``Path``.

    Returns:
        Ruta absoluta y normalizada para el sistema operativo actual.
    """
    ruta = Path(ruta_relativa)
    if ruta.is_absolute():
        return ruta
    return (RAIZ_PROYECTO / ruta).resolve()


def asegurar_directorio(ruta: str | Path) -> Path:
    """Crea un directorio (y sus padres) si no existe y devuelve su ruta absoluta.

    Args:
        ruta: Ruta del directorio, relativa a la raiz del proyecto o absoluta.

    Returns:
        Ruta absoluta del directorio existente.
    """
    destino = resolver(ruta)
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def registrar_raiz_en_path() -> None:
    """Inserta la raiz del proyecto en ``sys.path``.

    Necesario para que ``import src.<modulo>`` funcione tanto en los scripts de
    ``scripts/`` como en el notebook de EDA, sin instalar el paquete y sin
    depender de la variable de entorno ``PYTHONPATH``.
    """
    raiz = str(RAIZ_PROYECTO)
    if raiz not in sys.path:
        sys.path.insert(0, raiz)
