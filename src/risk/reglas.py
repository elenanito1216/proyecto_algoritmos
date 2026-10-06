"""Motor de reglas de riesgo estructural.

Este modulo es deliberadamente **una funcion pura**: no lee archivos, no imprime,
no depende de TensorFlow ni de OpenCV. Recibe las salidas de los dos modulos de
percepcion (la CNN y la inclinometria) mas el contexto del elemento inspeccionado,
y devuelve un nivel de riesgo **junto con la lista de reglas que lo produjeron**.

Por que reglas y no un segundo modelo
-------------------------------------
Un clasificador entrenado sobre "riesgo" necesitaria etiquetas de riesgo, que no
existen en el dataset y que solo un ingeniero estructural puede emitir. Un motor
de reglas, en cambio, es **explicable y auditable**: cada nivel viene con la
justificacion de ingenieria que lo motiva, se puede discutir con un experto y se
puede recalibrar cambiando un numero en ``config.yaml`` sin reentrenar nada.

Contexto normativo
------------------
Las reglas se apoyan en criterios de la **NSR-10** (Reglamento Colombiano de
Construccion Sismo Resistente, Decreto 926 de 2010) y en la practica habitual de
inspeccion visual post-sismo. Se citan en el campo ``justificacion`` de cada
regla. **No son limites normativos de aceptacion**: son criterios de tamizaje
para priorizar que elementos debe revisar una persona.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NIVELES: tuple[str, str, str] = ("Bajo", "Medio", "Alto")

SEVERIDAD_BAJA = 0
SEVERIDAD_MEDIA = 1
SEVERIDAD_ALTA = 2


@dataclass(frozen=True)
class ReglaDisparada:
    """Una regla que se activo durante la evaluacion.

    Attributes:
        codigo: Identificador corto y estable, p. ej. ``"R3"``. Sirve para
            trazar el resultado en el informe sin depender del texto.
        titulo: Enunciado breve de la regla.
        detalle: Explicacion con los valores concretos que la dispararon.
        justificacion: Criterio de ingenieria que motiva la regla.
        severidad: 0 = informativa, 1 = riesgo medio, 2 = riesgo alto.
    """

    codigo: str
    titulo: str
    detalle: str
    justificacion: str
    severidad: int


@dataclass
class EvaluacionRiesgo:
    """Resultado completo de la evaluacion de riesgo.

    Attributes:
        nivel: ``"Bajo"``, ``"Medio"`` o ``"Alto"``.
        severidad: Indice numerico del nivel (0, 1 o 2).
        reglas: Reglas que se dispararon, en orden de aparicion.
        resumen: Frase unica lista para mostrar en la interfaz.
        entradas: Copia de los valores de entrada, para trazabilidad.
        advertencias: Avisos que no alteran el nivel pero el usuario debe leer.
    """

    nivel: str
    severidad: int
    reglas: list[ReglaDisparada] = field(default_factory=list)
    resumen: str = ""
    entradas: dict[str, Any] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)

    def reglas_criticas(self) -> list[ReglaDisparada]:
        """Filtra las reglas que determinaron el nivel final.

        Returns:
            Reglas cuya severidad coincide con la severidad global.
        """
        return [r for r in self.reglas if r.severidad == self.severidad]

    def a_dict(self) -> dict[str, Any]:
        """Serializa la evaluacion a un diccionario JSON-compatible.

        Returns:
            Diccionario con todos los campos, con las reglas expandidas.
        """
        return {
            "nivel": self.nivel,
            "severidad": self.severidad,
            "resumen": self.resumen,
            "entradas": self.entradas,
            "advertencias": self.advertencias,
            "reglas": [
                {
                    "codigo": r.codigo,
                    "titulo": r.titulo,
                    "detalle": r.detalle,
                    "justificacion": r.justificacion,
                    "severidad": r.severidad,
                }
                for r in self.reglas
            ],
        }


def evaluar_riesgo(
    probabilidad_grieta: float,
    config: dict[str, Any],
    elemento: str = "muro_portante",
    orientacion_grieta: str | None = None,
    angulo_desaplome: float | None = None,
    confianza_inclinacion: int = 0,
) -> EvaluacionRiesgo:
    """Evalua el nivel de riesgo de un elemento estructural. Funcion pura.

    El nivel final es la **maxima severidad** entre las reglas disparadas, con
    una regla adicional de composicion (R7) que eleva el resultado cuando
    coinciden dos indicios de severidad media: dos senales medias simultaneas
    describen un cuadro peor que cualquiera de ellas por separado.

    Args:
        probabilidad_grieta: Probabilidad de que exista grieta, en ``[0, 1]``,
            tal como la devuelve la CNN.
        config: Configuracion del proyecto; se leen los umbrales de la seccion
            ``riesgo``. Ningun umbral esta incrustado en este modulo.
        elemento: Elemento inspeccionado. Valores esperados: ``"columna"``,
            ``"viga"``, ``"muro_portante"``, ``"muro_divisorio"``, ``"losa"``.
        orientacion_grieta: ``"vertical"``, ``"horizontal"``, ``"diagonal"``,
            ``"indeterminada"`` o ``None`` si no se estimo.
        angulo_desaplome: Desviacion respecto a la vertical en grados. ``None``
            si no se pudo medir. Se usa su valor absoluto.
        confianza_inclinacion: Numero de lineas coherentes que sustentan el
            angulo. Por debajo de ``riesgo.min_confianza_inclinacion`` el angulo
            se ignora para decidir.

    Returns:
        Objeto :class:`EvaluacionRiesgo` con el nivel y las reglas disparadas.

    Raises:
        ValueError: Si ``probabilidad_grieta`` esta fuera de ``[0, 1]``.

    Example:
        >>> cfg = {"riesgo": {"umbral_grieta": 0.5, "umbral_grieta_alta": 0.85,
        ...                   "desaplome_atencion_grados": 1.0,
        ...                   "desaplome_severo_grados": 2.0,
        ...                   "elementos_criticos": ["columna"],
        ...                   "orientaciones_graves": {"columna": ["horizontal"]},
        ...                   "min_confianza_inclinacion": 3}}
        >>> ev = evaluar_riesgo(0.1, cfg, elemento="columna")
        >>> ev.nivel
        'Bajo'
    """
    if not 0.0 <= float(probabilidad_grieta) <= 1.0:
        raise ValueError(
            f"probabilidad_grieta debe estar en [0, 1]; se recibio {probabilidad_grieta}."
        )

    cfg = config.get("riesgo", {})
    umbral_grieta = float(cfg.get("umbral_grieta", 0.5))
    umbral_alta = float(cfg.get("umbral_grieta_alta", 0.85))
    desaplome_atencion = float(cfg.get("desaplome_atencion_grados", 1.0))
    desaplome_severo = float(cfg.get("desaplome_severo_grados", 2.0))
    criticos = [str(e).lower() for e in cfg.get("elementos_criticos", [])]
    graves: dict[str, list[str]] = cfg.get("orientaciones_graves", {})
    min_confianza = int(cfg.get("min_confianza_inclinacion", 3))

    elemento_norm = str(elemento).lower().strip()
    orientacion_norm = (orientacion_grieta or "").lower().strip() or None
    prob = float(probabilidad_grieta)

    reglas: list[ReglaDisparada] = []
    advertencias: list[str] = []

    hay_grieta = prob >= umbral_grieta

    if hay_grieta:
        reglas.append(
            ReglaDisparada(
                codigo="R1",
                titulo="Presencia de grieta detectada",
                detalle=(
                    f"El clasificador asigna P(grieta) = {prob:.3f}, por encima del "
                    f"umbral de decision {umbral_grieta:.2f}."
                ),
                justificacion=(
                    "Toda fisura visible en un elemento estructural exige registro y "
                    "seguimiento. La NSR-10 (Titulo A, evaluacion e intervencion) "
                    "considera la fisuracion como indicio de que el elemento ha "
                    "superado su estado limite de servicio."
                ),
                severidad=SEVERIDAD_MEDIA,
            )
        )
    else:
        reglas.append(
            ReglaDisparada(
                codigo="R0",
                titulo="Sin evidencia de grieta",
                detalle=(f"P(grieta) = {prob:.3f}, por debajo del umbral {umbral_grieta:.2f}."),
                justificacion=(
                    "La ausencia de fisuracion visible no descarta dano interno, pero "
                    "no aporta evidencia de riesgo en la inspeccion visual."
                ),
                severidad=SEVERIDAD_BAJA,
            )
        )

    if prob >= umbral_alta:
        reglas.append(
            ReglaDisparada(
                codigo="R2",
                titulo="Evidencia fuerte de fisuracion",
                detalle=(
                    f"P(grieta) = {prob:.3f} supera el umbral de alta confianza "
                    f"{umbral_alta:.2f}."
                ),
                justificacion=(
                    "Una deteccion de alta confianza reduce la probabilidad de falso "
                    "positivo y justifica escalar la inspeccion sin esperar a otra "
                    "senal concurrente."
                ),
                severidad=SEVERIDAD_MEDIA,
            )
        )

    if hay_grieta and orientacion_norm and orientacion_norm != "indeterminada":
        orientaciones_graves_elemento = [str(o).lower() for o in graves.get(elemento_norm, [])]
        if orientacion_norm in orientaciones_graves_elemento:
            reglas.append(
                ReglaDisparada(
                    codigo="R3",
                    titulo=f"Grieta {orientacion_norm} en {elemento_norm.replace('_', ' ')}",
                    detalle=(
                        f"La orientacion dominante estimada es {orientacion_norm}, "
                        f"catalogada como grave para el elemento '{elemento_norm}'."
                    ),
                    justificacion=_justificacion_orientacion(elemento_norm, orientacion_norm),
                    severidad=SEVERIDAD_ALTA,
                )
            )
        else:
            reglas.append(
                ReglaDisparada(
                    codigo="R4",
                    titulo=f"Grieta {orientacion_norm} (patron no critico)",
                    detalle=(
                        f"La orientacion {orientacion_norm} no figura entre los patrones "
                        f"de alta gravedad para '{elemento_norm}'."
                    ),
                    justificacion=(
                        "Las fisuras verticales finas en muros suelen asociarse a "
                        "retraccion del mortero o a cambios termicos, no a perdida de "
                        "capacidad portante. Requieren seguimiento, no alarma."
                    ),
                    severidad=SEVERIDAD_MEDIA,
                )
            )

    desaplome_fiable = angulo_desaplome is not None and confianza_inclinacion >= min_confianza
    magnitud = abs(float(angulo_desaplome)) if angulo_desaplome is not None else None

    if angulo_desaplome is None:
        advertencias.append(
            "No se pudo medir la inclinacion: no se detecto ningun elemento vertical "
            "fiable en la imagen. El nivel de riesgo se basa solo en la fisuracion."
        )
    elif not desaplome_fiable:
        advertencias.append(
            f"El desaplome estimado ({magnitud:.2f} grados) se apoya en solo "
            f"{confianza_inclinacion} linea(s), por debajo del minimo de {min_confianza}. "
            "No se ha usado para determinar el nivel de riesgo."
        )
    elif magnitud is not None and magnitud >= desaplome_severo:
        reglas.append(
            ReglaDisparada(
                codigo="R5",
                titulo="Desaplome severo",
                detalle=(
                    f"Desviacion de {magnitud:.2f} grados respecto a la vertical, por "
                    f"encima del umbral severo de {desaplome_severo:.2f} grados "
                    f"(~{_grados_a_deriva(magnitud):.1f}% de deriva equivalente)."
                ),
                justificacion=(
                    "La NSR-10 (Titulo A.6) limita la deriva maxima de piso al 1.0% de "
                    "la altura de entrepiso para estructuras de concreto reforzado, "
                    "equivalente a ~0.57 grados. Una desviacion permanente que triplica "
                    "ese limite indica deformacion residual y posible perdida de "
                    "verticalidad del elemento portante."
                ),
                severidad=SEVERIDAD_ALTA,
            )
        )
    elif magnitud is not None and magnitud >= desaplome_atencion:
        reglas.append(
            ReglaDisparada(
                codigo="R6",
                titulo="Desaplome apreciable",
                detalle=(
                    f"Desviacion de {magnitud:.2f} grados respecto a la vertical, entre "
                    f"el umbral de atencion ({desaplome_atencion:.2f}) y el severo "
                    f"({desaplome_severo:.2f})."
                ),
                justificacion=(
                    "Un desaplome por encima del limite de deriva de la NSR-10 "
                    "(~0.57 grados) sugiere deformacion permanente. La medida "
                    "fotografica tiene error de perspectiva, por lo que se fija el "
                    "umbral de atencion en 1.0 grado para absorber ese margen."
                ),
                severidad=SEVERIDAD_MEDIA,
            )
        )

    if hay_grieta and elemento_norm in criticos:
        reglas.append(
            ReglaDisparada(
                codigo="R7",
                titulo="Fisuracion en elemento critico",
                detalle=(
                    f"'{elemento_norm.replace('_', ' ')}' pertenece al sistema de "
                    "resistencia sismica; su falla compromete la estabilidad global."
                ),
                justificacion=(
                    "La NSR-10 (Titulo C) distingue los elementos del sistema de "
                    "resistencia sismica de los no estructurales. El dano en columnas, "
                    "vigas y muros portantes se evalua con criterios mas exigentes "
                    "porque su falla puede desencadenar colapso progresivo."
                ),
                severidad=SEVERIDAD_MEDIA,
            )
        )

    hay_desaplome_medio = any(r.codigo == "R6" for r in reglas)
    if hay_grieta and hay_desaplome_medio:
        reglas.append(
            ReglaDisparada(
                codigo="R8",
                titulo="Concurrencia de fisuracion y desaplome",
                detalle=(
                    f"Coinciden grieta detectada (P = {prob:.3f}) y desaplome de "
                    f"{magnitud:.2f} grados en el mismo elemento."
                ),
                justificacion=(
                    "Fisuracion y perdida de verticalidad simultaneas son compatibles "
                    "con un mecanismo de dano activo (asentamiento diferencial o "
                    "deformacion residual post-sismo), no con dos defectos "
                    "independientes de origen constructivo."
                ),
                severidad=SEVERIDAD_ALTA,
            )
        )

    severidad = max((r.severidad for r in reglas), default=SEVERIDAD_BAJA)
    nivel = NIVELES[severidad]

    return EvaluacionRiesgo(
        nivel=nivel,
        severidad=severidad,
        reglas=reglas,
        resumen=_construir_resumen(nivel, hay_grieta, magnitud, desaplome_fiable, elemento_norm),
        entradas={
            "probabilidad_grieta": prob,
            "elemento": elemento_norm,
            "orientacion_grieta": orientacion_norm,
            "angulo_desaplome": (None if angulo_desaplome is None else float(angulo_desaplome)),
            "confianza_inclinacion": int(confianza_inclinacion),
        },
        advertencias=advertencias,
    )


def _grados_a_deriva(grados: float) -> float:
    """Convierte una desviacion angular en deriva porcentual equivalente.

    Args:
        grados: Desviacion respecto a la vertical.

    Returns:
        Deriva equivalente en porcentaje (``tan(theta) * 100``), que es la
        magnitud con la que la NSR-10 expresa sus limites.
    """
    import math

    return math.tan(math.radians(abs(grados))) * 100.0


def _justificacion_orientacion(elemento: str, orientacion: str) -> str:
    """Devuelve el criterio de ingenieria que hace grave una orientacion concreta.

    Args:
        elemento: Elemento estructural normalizado.
        orientacion: Orientacion dominante de la grieta.

    Returns:
        Texto de justificacion. Si la combinacion no tiene una explicacion
        especifica catalogada, se devuelve una generica.
    """
    catalogo: dict[tuple[str, str], str] = {
        ("muro_portante", "diagonal"): (
            "Las fisuras diagonales en muros portantes son la firma tipica de la "
            "falla por cortante ante carga lateral (sismo). La NSR-10 (Titulo D, "
            "mamposteria estructural) trata el agrietamiento diagonal como "
            "indicador de agotamiento de la resistencia a cortante del muro."
        ),
        ("muro_divisorio", "diagonal"): (
            "Aun en un muro no portante, la fisura diagonal indica que el muro esta "
            "recibiendo carga lateral para la que no fue disenado. Es un riesgo de "
            "desprendimiento y caida (elemento no estructural, NSR-10 Titulo A.9)."
        ),
        ("columna", "horizontal"): (
            "La fisura horizontal en una columna es perpendicular al eje de "
            "compresion y sugiere flexo-traccion o falla incipiente del recubrimiento. "
            "Es una de las senales mas graves en inspeccion visual post-sismo."
        ),
        ("columna", "diagonal"): (
            "La fisura diagonal en columna indica esfuerzo cortante elevado, "
            "asociado a columnas cortas o a confinamiento transversal insuficiente "
            "(NSR-10 Titulo C.21, requisitos de confinamiento)."
        ),
        ("viga", "diagonal"): (
            "La fisura diagonal cerca del apoyo de una viga corresponde al patron de "
            "agrietamiento por cortante, que en la NSR-10 (Titulo C.11) exige "
            "verificacion inmediata del refuerzo transversal."
        ),
        ("viga", "vertical"): (
            "La fisura vertical en el centro de la luz de una viga corresponde a "
            "traccion por flexion. Si atraviesa mas de la mitad del canto, indica "
            "que el refuerzo longitudinal esta trabajando cerca de su limite."
        ),
        ("losa", "diagonal"): (
            "El agrietamiento diagonal en losa cerca de columnas es compatible con "
            "punzonamiento, un modo de falla fragil y sin aviso previo."
        ),
    }
    return catalogo.get(
        (elemento, orientacion),
        (
            f"El patron {orientacion} en '{elemento.replace('_', ' ')}' esta catalogado "
            "como grave en config.yaml. Requiere verificacion por un ingeniero "
            "estructural."
        ),
    )


def _construir_resumen(
    nivel: str,
    hay_grieta: bool,
    magnitud_desaplome: float | None,
    desaplome_fiable: bool,
    elemento: str,
) -> str:
    """Redacta la frase de resumen que se muestra en la interfaz.

    Args:
        nivel: Nivel de riesgo determinado.
        hay_grieta: Si se supero el umbral de deteccion de grieta.
        magnitud_desaplome: Desviacion absoluta en grados, o ``None``.
        desaplome_fiable: Si la medida de desaplome alcanzo la confianza minima.
        elemento: Elemento estructural normalizado.

    Returns:
        Frase unica en espanol.
    """
    partes: list[str] = []
    partes.append("con fisuracion detectada" if hay_grieta else "sin fisuracion detectada")
    if desaplome_fiable and magnitud_desaplome is not None:
        partes.append(f"desaplome de {magnitud_desaplome:.2f} grados")
    else:
        partes.append("sin medida fiable de desaplome")
    return f"Riesgo {nivel} en {elemento.replace('_', ' ')}: " + " y ".join(partes) + "."
