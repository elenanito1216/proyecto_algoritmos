"""Verificacion del entorno: cierre de la Fase 0.

Comprueba en un solo comando que el entorno virtual esta bien montado antes de
perder tiempo depurando un entrenamiento que en realidad falla por una
dependencia ausente.

Uso:
    python scripts/verificar_entorno.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEPENDENCIAS: list[tuple[str, str, str]] = [
    ("numpy", "NumPy", "__version__"),
    ("pandas", "Pandas", "__version__"),
    ("yaml", "PyYAML", "__version__"),
    ("sklearn", "Scikit-learn", "__version__"),
    ("matplotlib", "Matplotlib", "__version__"),
    ("seaborn", "Seaborn", "__version__"),
    ("plotly", "Plotly", "__version__"),
    ("cv2", "OpenCV", "__version__"),
    ("tensorflow", "TensorFlow", "__version__"),
    ("streamlit", "Streamlit", "__version__"),
    ("PIL", "Pillow", "__version__"),
    ("psutil", "psutil", "__version__"),
]

MARCA_OK = "[ OK ]"
MARCA_FALLO = "[FALLO]"


def _verificar_interprete() -> bool:
    """Comprueba que se esta ejecutando dentro del entorno virtual del proyecto.

    Returns:
        ``True`` si el interprete parece ser el del ``.venv`` local.
    """
    raiz = Path(__file__).resolve().parents[1]
    ejecutable = Path(sys.executable).resolve()
    dentro = str(raiz / ".venv") in str(ejecutable)

    print(f"  Python      : {platform.python_version()}")
    print(f"  Interprete  : {ejecutable}")
    if dentro:
        print(f"  {MARCA_OK} El interprete pertenece al .venv del proyecto.")
    else:
        print(
            f"  {MARCA_FALLO} El interprete NO es el del .venv del proyecto.\n"
            "         Actívalo antes de continuar:\n"
            "           Windows  : .\\.venv\\Scripts\\Activate.ps1\n"
            "           Linux/Mac: source .venv/bin/activate"
        )
    if sys.version_info < (3, 10):  # noqa: UP036
        print(f"  {MARCA_FALLO} Se requiere Python 3.10 o superior.")
        return False
    if sys.version_info >= (3, 13):
        print(
            f"  {MARCA_FALLO} TensorFlow 2.19 no publica ruedas para Python "
            f"{platform.python_version()}. Usa Python 3.10-3.12."
        )
        return False
    return dentro


def _verificar_dependencias() -> list[str]:
    """Importa cada dependencia critica y reporta su version.

    Returns:
        Lista de nombres de las dependencias que fallaron.
    """
    fallidas: list[str] = []
    for modulo, nombre, atributo in DEPENDENCIAS:
        try:
            importado = importlib.import_module(modulo)
            version = getattr(importado, atributo, "?")
            print(f"  {MARCA_OK} {nombre:<14} {version}")
        except Exception as error:  # noqa: BLE001
            print(f"  {MARCA_FALLO} {nombre:<14} {type(error).__name__}: {error}")
            fallidas.append(nombre)
    return fallidas


def _verificar_proyecto() -> list[str]:
    """Comprueba que los modulos del proyecto se importan y que existe el config.

    Returns:
        Lista de problemas encontrados.
    """
    problemas: list[str] = []
    try:
        from src.utils.config import cargar_config
        from src.utils.rutas import RAIZ_PROYECTO

        config = cargar_config()
        print(f"  {MARCA_OK} config.yaml     cargado ({len(config)} secciones)")
        print(f"  {MARCA_OK} Raiz detectada  {RAIZ_PROYECTO}")

        for clave in ("datos_raw", "datos_propias"):
            from src.utils.config import obtener
            from src.utils.rutas import resolver

            destino = resolver(obtener(config, f"rutas.{clave}"))
            if destino.is_dir():
                n = sum(1 for p in destino.rglob("*") if p.is_file())
                estado = f"{n} archivo(s)" if n else "vacio"
                print(f"  {MARCA_OK} rutas.{clave:<12} {destino.name}/ ({estado})")
            else:
                print(f"  ...... rutas.{clave:<12} aun no existe: {destino}")
    except Exception as error:  # noqa: BLE001
        print(f"  {MARCA_FALLO} No se pudo cargar la configuracion: {error}")
        problemas.append("config")

    for modulo in (
        "src.data.loader",
        "src.models.baseline",
        "src.models.transfer",
        "src.vision.inclinacion",
        "src.risk.reglas",
        "src.eval.metricas",
        "src.eval.complejidad",
    ):
        try:
            importlib.import_module(modulo)
            print(f"  {MARCA_OK} {modulo}")
        except Exception as error:  # noqa: BLE001
            print(f"  {MARCA_FALLO} {modulo}: {type(error).__name__}: {error}")
            problemas.append(modulo)
    return problemas


def _verificar_funcional() -> list[str]:
    """Ejecuta una prueba minima de OpenCV y de TensorFlow de extremo a extremo.

    Returns:
        Lista de problemas encontrados.
    """
    problemas: list[str] = []
    try:
        import cv2
        import numpy as np

        lienzo = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.line(lienzo, (100, 20), (104, 180), (255, 255, 255), 3)
        from src.utils.config import cargar_config
        from src.vision.inclinacion import estimar_inclinacion

        resultado = estimar_inclinacion(lienzo, cargar_config())
        print(
            f"  {MARCA_OK} OpenCV + Hough  linea sintetica -> "
            f"{resultado.angulo_grados if resultado.angulo_grados is not None else 'n/d'}"
        )
    except Exception as error:  # noqa: BLE001
        print(f"  {MARCA_FALLO} Prueba de OpenCV: {type(error).__name__}: {error}")
        problemas.append("opencv")

    try:
        from src.models.baseline import compilar, construir_baseline
        from src.utils.config import cargar_config, forma_entrada
        from src.utils.dispositivo import detectar_dispositivo

        config = cargar_config()
        modelo = compilar(construir_baseline(config), 1e-3)
        import numpy as np

        entrada = np.zeros((1, *forma_entrada(config)), dtype=np.float32)
        salida = modelo(entrada, training=False)
        print(f"  {MARCA_OK} TensorFlow      pasada directa -> forma {tuple(salida.shape)}")
        detectar_dispositivo(verboso=False)
    except Exception as error:  # noqa: BLE001
        print(f"  {MARCA_FALLO} Prueba de TensorFlow: {type(error).__name__}: {error}")
        problemas.append("tensorflow")

    try:
        from src.risk.reglas import evaluar_riesgo
        from src.utils.config import cargar_config

        evaluacion = evaluar_riesgo(
            0.9, cargar_config(), elemento="columna", orientacion_grieta="diagonal"
        )
        print(f"  {MARCA_OK} Motor de reglas  nivel de prueba -> {evaluacion.nivel}")
    except Exception as error:  # noqa: BLE001
        print(f"  {MARCA_FALLO} Motor de reglas: {type(error).__name__}: {error}")
        problemas.append("reglas")

    return problemas


def main() -> int:
    """Ejecuta todas las verificaciones y devuelve un codigo de salida.

    Returns:
        0 si todo esta correcto, 1 si hubo algun fallo.
    """
    print("=" * 72)
    print("  VERIFICACION DEL ENTORNO - Fase 0")
    print("=" * 72)

    print("\n1. Interprete de Python")
    interprete_ok = _verificar_interprete()

    print("\n2. Dependencias")
    fallidas = _verificar_dependencias()

    print("\n3. Proyecto")
    problemas_proyecto = _verificar_proyecto()

    print("\n4. Prueba funcional")
    problemas_funcionales = _verificar_funcional()

    print("\n" + "=" * 72)
    total = len(fallidas) + len(problemas_proyecto) + len(problemas_funcionales)
    if total == 0 and interprete_ok:
        print("  ENTORNO CORRECTO. Fase 0 superada.")
        print("  Siguiente paso: coloca el dataset en data/raw/ y abre")
        print("  notebooks/01_eda.ipynb, o lanza el ensayo rapido:")
        print("    python scripts/train_transfer.py --config config.yaml --subset 0.05 --epochs 2")
        print("=" * 72)
        return 0

    print(f"  SE ENCONTRARON {total} PROBLEMA(S).")
    if fallidas:
        print(f"  Dependencias que fallan: {', '.join(fallidas)}")
        print("  Solucion: pip install -r requirements.txt")
    if not interprete_ok:
        print("  Recuerda activar el entorno virtual antes de ejecutar los scripts.")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
