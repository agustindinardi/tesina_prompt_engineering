"""Experimento 8 — Síntesis multi-fuente con detección de contradicciones (tarea compleja).

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Evalúa seis técnicas de prompting (Zero-shot, Few-shot, Chain-of-Thought,
Chain-of-Thought limitado, Structured Prompting JSON y Role Prompting) sobre
el modelo gpt-4o-mini, en una tarea de síntesis de tres fragmentos breves
(estilo informe/noticia) que se solapan parcialmente y se contradicen en
al menos un dato. El modelo debe integrar lo consistente entre las tres
fuentes y señalar explícitamente la o las contradicciones sin resolverlas
arbitrariamente. A diferencia del resumen (compresión de una única
fuente), esta tarea exige mantener un modelo mental de qué afirma cada
fuente por separado y compararlas entre sí.

Al igual que en experimento7 y 9, cada input tiene una
referencia (qué contradicciones son reales y cuál es el consenso
esperado entre las fuentes) que se inyecta ÚNICAMENTE en el prompt del
juez, nunca en el prompt que recibe el modelo generador: sin esa
referencia, el juez debería descubrir las contradicciones por su cuenta,
arriesgando el mismo tipo de error que se busca medir en el modelo
evaluado.

exactitud_funcional es GRADUADA en esta tarea (no binaria): detectar una
contradicción parcialmente, o atribuirla a la fuente equivocada, admite
grados intermedios de corrección, a diferencia de scheduling (donde un
cronograma es válido o no) o de la resolución matemática (donde el
resultado final es correcto o no).

La calidad de cada respuesta se evalúa mediante un esquema de doble juez
(GPT-4o y Claude Haiku 4.5) en cuatro dimensiones:
exactitud funcional, completitud funcional, claridad comunicacional y
profundidad conceptual.

El experimento se repite ``--num-runs`` veces (10 por defecto) para poder
calcular la variabilidad de los resultados entre corridas. Cada corrida se
guarda como un archivo JSON independiente: ``results8.0.json``,
``results8.1.json``, ..., ``results8.9.json``.

Uso:
    python experimento8.py

Variables de entorno requeridas:
    OPENAI_API_KEY   Clave de API de OpenAI (genera las respuestas y actúa
                     como uno de los dos jueces).
    CLAUDE_API_KEY   Clave de API de Anthropic (actúa como el segundo juez).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from openai import OpenAI

# =============================================================================
# Configuración
# =============================================================================

TASK_DESCRIPTION = "síntesis de información multi-fuente con detección de contradicciones"

GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL_OPENAI = "gpt-4o"
JUDGE_MODEL_CLAUDE = "claude-haiku-4-5-20251001"

# Cantidad máxima de intentos por juez cuando omite alguna técnica en su
# respuesta (mismo mecanismo agregado en experimento7_scheduling.py tras
# detectar omisiones silenciosas con prompts largos).
MAX_JUDGE_RETRIES = 3

QUALITY_DIMENSIONS = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]

ALLOWED_LIKERT_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
ALLOWED_BINARY_VALUES = [0.0, 1.0]

# Escala permitida para exactitud_funcional en ESTA tarea: graduada, igual
# que en traducción y resumen. Detectar una contradicción parcialmente, o
# atribuirla a la fuente equivocada, admite grados intermedios de
# corrección — a diferencia de scheduling o matemática, donde hay un
# resultado final binario correcto/incorrecto.
EXACTITUD_ALLOWED_VALUES = ALLOWED_LIKERT_VALUES

CONSIGNA_SINTESIS = (
    "Redactá una síntesis que integre la información "
    "consistente entre las tres fuentes y señale explícitamente los puntos "
    "en que se contradicen, indicando qué fuente sostiene qué, sin decidir "
    "arbitrariamente cuál tiene razón."
)

# Tareas de entrada evaluadas: 3 casos de 3 fuentes breves cada uno, con 2
# contradicciones y un consenso esperado por caso. El campo "reference"
# NUNCA se incluye en el prompt que recibe el modelo generador — solo se
# usa para construir el prompt de evaluación de los jueces.
#
# Diseño del set: input 1 y 2 comparten estructura (empresa minimiza /
# prensa denuncia / auditoría media) para mantener dificultad comparable;
# input 3 es el más exigente porque una de las "contradicciones" es en
# realidad una reconciliación parcial entre fuentes, lo que exige un nivel
# de comprensión conceptual mayor que simplemente listar discrepancias.
INPUTS: list[dict[str, str]] = [
    {
        "text": (
            "Fuente 1, comunicado de la empresa: \"El 12 de marzo detectamos "
            "un acceso no autorizado a un servidor de staging. No se "
            "comprometieron datos de clientes. El vector de ataque fue una "
            "credencial filtrada de un empleado. Ya se rotaron todas las "
            "credenciales y se reforzó el monitoreo.\"\n\n"
            "Fuente 2, medio especializado: \"Según investigadores "
            "independientes, el acceso ocurrió el 10 de marzo, dos días "
            "antes de lo declarado, y alcanzó también la base de datos de "
            "producción, donde se habrían expuesto unos 40.000 registros de "
            "usuarios.\"\n\n"
            "Fuente 3, informe de auditoría posterior: \"El primer acceso "
            "registrado data del 10 de marzo. El servidor de producción fue "
            "efectivamente alcanzado, pero los registros estaban cifrados y "
            "no hay evidencia de exfiltración legible. El vector fue una "
            "credencial expuesta en un repositorio público.\"\n\n"
            f"{CONSIGNA_SINTESIS}"
        ),
        "reference": (
            "Contradicciones reales: (1) FECHA del acceso — Fuente 1 dice "
            "12 de marzo; Fuentes 2 y 3 coinciden en 10 de marzo (la "
            "auditoría respalda a la fuente 2). (2) ALCANCE/exposición de "
            "datos — Fuente 1 dice que no se comprometieron datos; Fuente 2 "
            "dice 40.000 registros expuestos; Fuente 3 dice que el servidor "
            "de producción fue alcanzado pero los registros estaban "
            "cifrados sin evidencia de exfiltración (posición intermedia "
            "entre las otras dos). CONSENSO esperado: las tres fuentes "
            "coinciden en que hubo un acceso no autorizado y en que el "
            "vector fue una credencial comprometida (difieren en si fue "
            "'filtrada de un empleado' o 'expuesta en un repositorio "
            "público', variante menor del mismo consenso). Una síntesis "
            "correcta debe señalar AMBAS contradicciones (fecha y alcance) "
            "atribuyendo qué fuente sostiene qué, e integrar el consenso "
            "sobre el acceso no autorizado y la credencial comprometida."
        ),
    },
    {
        "text": (
            "Fuente 1, comunicado de la empresa: \"Retiramos preventivamente "
            "el lote L-2024-118 de nuestro producto por una posible "
            "desviación de calidad detectada en controles internos. No "
            "existe ningún caso de enfermedad confirmado asociado al "
            "producto. El lote se distribuyó únicamente en la provincia de "
            "Córdoba.\"\n\n"
            "Fuente 2, agencia sanitaria: \"Se confirmaron tres casos de "
            "salmonelosis vinculados por trazabilidad al lote L-2024-118. "
            "La distribución del lote alcanzó cuatro provincias: Córdoba, "
            "Santa Fe, Mendoza y Buenos Aires. Se ordenó el retiro "
            "inmediato.\"\n\n"
            "Fuente 3, medio nacional: \"Fuentes hospitalarias indican que "
            "los tres casos siguen en estudio y aún no tienen confirmación "
            "bacteriológica definitiva. El retiro habría comenzado cinco "
            "días después de la primera alerta interna de la empresa. La "
            "distribución en cuatro provincias fue confirmada por "
            "distribuidores mayoristas.\"\n\n"
            f"{CONSIGNA_SINTESIS}"
        ),
        "reference": (
            "Contradicciones reales: (1) ESTADO de los casos de enfermedad "
            "— Fuente 1 dice que no hay casos confirmados; Fuente 2 dice "
            "que hay tres casos confirmados; Fuente 3 dice que los tres "
            "casos siguen en estudio, sin confirmación definitiva (matiza "
            "a la fuente 2, no la contradice del todo). (2) ALCANCE "
            "geográfico — Fuente 1 dice una sola provincia (Córdoba); "
            "Fuentes 2 y 3 coinciden en cuatro provincias (la fuente 3 "
            "respalda a la fuente 2 en este punto). Dato exclusivo de la "
            "fuente 3, sin contradicción pero relevante para completitud: "
            "el retiro habría comenzado con una demora de cinco días "
            "respecto de la alerta interna. CONSENSO esperado: existe un "
            "lote (L-2024-118) retirado por un problema de calidad/"
            "seguridad alimentaria. Una síntesis correcta debe señalar "
            "ambas contradicciones (estado de los casos y alcance "
            "geográfico); mencionar el dato de la demora de la fuente 3 sí "
            "suma completitud pero no es obligatorio; no debe decidir "
            "arbitrariamente si hubo o no casos confirmados."
        ),
    },
    {
        "text": (
            "Fuente 1, página de estado del proveedor: \"El servicio "
            "experimentó una interrupción de 45 minutos causada por un "
            "error de configuración de DNS. No hubo pérdida de datos. "
            "Todos los sistemas operan con normalidad.\"\n\n"
            "Fuente 2, monitor externo independiente: \"Nuestras mediciones "
            "registraron indisponibilidad total o parcial durante 2 horas y "
            "10 minutos en al menos dos regiones, considerablemente más que "
            "lo informado oficialmente.\"\n\n"
            "Fuente 3, postmortem técnico publicado una semana después: "
            "\"La caída global duró 47 minutos, seguida de degradación "
            "parcial durante aproximadamente 90 minutos adicionales en dos "
            "regiones. La causa raíz no fue el DNS en sí, sino un cambio de "
            "configuración que activó un bug latente en el balanceador de "
            "carga.\"\n\n"
            f"{CONSIGNA_SINTESIS}"
        ),
        "reference": (
            "Este es el input más exigente del set porque una de las "
            "aparentes contradicciones es en realidad una RECONCILIACIÓN "
            "PARCIAL, no una contradicción irreconciliable. DURACIÓN: "
            "Fuente 1 dice 45 minutos; Fuente 2 dice 2h10 (130 min); Fuente "
            "3 dice 47 minutos de caída global + 90 minutos adicionales de "
            "degradación parcial (47+90=137 min, muy cercano a lo medido "
            "por la fuente 2). Es decir, la fuente 3 RECONCILIA a las "
            "otras dos: la caída 'dura' (45-47 min) coincide entre F1 y F3, "
            "y el tiempo total con degradación coincide aproximadamente con "
            "lo medido por F2 — no son en rigor tres cifras contradictorias "
            "sino tres formas de medir fenómenos distintos (caída total vs. "
            "caída dura vs. caída + degradación). CAUSA: Fuente 1 dice que "
            "la causa fue un error de configuración de DNS; Fuente 3 dice "
            "que la causa raíz NO fue el DNS sino un bug latente en el "
            "balanceador de carga activado por un cambio de configuración "
            "(esta sí es una contradicción genuina sobre la causa raíz). "
            "Una síntesis que solo señale 'discrepancia en la duración' sin "
            "notar que la fuente 3 reconcilia a F1 y F2 debe puntuarse en "
            "un nivel intermedio de exactitud_funcional (detecta que hay "
            "discrepancia pero no su naturaleza real); una síntesis que sí "
            "note la reconciliación debe puntuarse en la parte alta de la "
            "escala."
        ),
    },
]

DEFAULT_NUM_RUNS = 10
DEFAULT_OUTPUT_PREFIX = "results8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experimento8")


# =============================================================================
# Clientes de API
# =============================================================================

def _require_env(var_name: str) -> str:
    """Lee una variable de entorno o corta la ejecución con un error claro."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"La variable de entorno {var_name} no está configurada")
    return value


openai_client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=_require_env("CLAUDE_API_KEY"))


# =============================================================================
# Mapeo input -> referencia (para pasarle al juez sin tocar el prompt del
# modelo generador)
# =============================================================================

INPUT_REFERENCE_BY_TEXT: dict[str, str] = {item["text"]: item["reference"] for item in INPUTS}


# =============================================================================
# Normalización de puntajes del juez
# =============================================================================

def _closest_allowed(value: Any, allowed_values: list[float]) -> float:
    """Redondea un puntaje al valor permitido más cercano (0, 0.3, 0.5, 0.7, 1)."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return allowed_values[0]
    return min(allowed_values, key=lambda allowed: (abs(allowed - numeric_value), allowed))


def normalize_evaluation_scores(scores: dict[str, Any]) -> dict[str, float]:
    """Normaliza las 4 dimensiones de un puntaje crudo del juez a la escala permitida."""
    return {
        "exactitud_funcional": _closest_allowed(scores.get("exactitud_funcional", 0), EXACTITUD_ALLOWED_VALUES),
        "completitud_funcional": _closest_allowed(scores.get("completitud_funcional", 0), ALLOWED_LIKERT_VALUES),
        "claridad_comunicacional": _closest_allowed(scores.get("claridad_comunicacional", 0), ALLOWED_LIKERT_VALUES),
        "profundidad_conceptual": _closest_allowed(scores.get("profundidad_conceptual", 0), ALLOWED_LIKERT_VALUES),
    }


def _empty_scores() -> dict[str, float]:
    return {dim: 0.0 for dim in QUALITY_DIMENSIONS}


# =============================================================================
# Técnicas de prompting
# =============================================================================
# Nota: igual que en experimento9_restricciones.py, "text" ya es la consigna
# completa (las 3 fuentes + la instrucción de síntesis), no un dato en
# bruto al que hay que agregarle instrucción. Por eso zero_shot no le
# agrega nada adicional.

def zero_shot(text: str) -> str:
    return text


def few_shot(text: str) -> str:
    # NOTA METODOLÓGICA: el ejemplo usa un caso genérico y mucho más simple
    # (una sola contradicción numérica) que los casos reales de la tarea
    # (2 contradicciones por input, atribución a 3 fuentes). Se mantiene
    # así a propósito para no regalar la estructura completa de la
    # solución; queda documentado como limitación del diseño.
    example = (
        "Ejemplo de síntesis con detección de contradicciones:\n"
        "Fuente 1: 'El evento tuvo 200 asistentes.' Fuente 2 (organizadores): "
        "'Asistieron 350 personas.'\n"
        "Síntesis de ejemplo: 'El evento reunió una cantidad considerable de "
        "asistentes, aunque las fuentes difieren: la Fuente 1 reporta 200 "
        "personas, mientras que la Fuente 2 (organizadores) indica 350.'\n\n"
    )
    return f"{example}Ahora resolvé la siguiente consigna:\n\n{text}"


def cot(text: str) -> str:
    return (
        f"{text}\n\n"
        "Antes de escribir la síntesis final, explicá brevemente tu "
        "razonamiento. Luego entregá la síntesis final."
    )


def cot_limited(text: str) -> str:
    return (
        f"{text}\n\n"
        "Antes de escribir la síntesis final, explicá tu razonamiento y "
        "limitalo a máximo 100 palabras; luego entregá la síntesis final."
    )


def structured_prompting_json(text: str) -> str:
    prompt = {
        "instruction": CONSIGNA_SINTESIS,
        "input": text,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text: str) -> str:
    # NOTA METODOLÓGICA (corregido tras revisión): la versión anterior de
    # este rol agregaba "experto en detectar y señalar contradicciones
    # entre fuentes sin tomar partido por ninguna", que es casi una
    # paráfrasis literal de la consigna misma — le daba a esta técnica una
    # ayuda que ninguna de las otras cinco recibía. El rol ahora solo nombra
    # el dominio de expertise, igual que en el resto de los scripts
    # (traductor, profesor, editor, gestor de proyectos, redactor).
    return (
        "Actuá como un periodista de datos especializado en verificación de "
        "fuentes (fact-checking).\n\n"
        f"{text}"
    )


TECHNIQUES: dict[str, Callable[[str], str]] = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "cot": cot,
    "cot_limited": cot_limited,
    "structured_prompting_json": structured_prompting_json,
    "role_prompting": role_prompting,
}


# =============================================================================
# Evaluación mediante doble juez (GPT-4o + Claude Haiku 4.5)
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """Realiza una evaluación comparativa de las siguientes respuestas para la tarea de {task_description}: "{input_text}"

REFERENCIA DE CONTRADICCIONES Y CONSENSO (esto es solo para tu evaluación; el modelo evaluado NUNCA vio esta referencia, solo las 3 fuentes):
{reference}

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones, usando la referencia para verificar la corrección.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para {task_description}:

EXACTITUD FUNCIONAL: ¿La síntesis identifica correctamente las contradicciones reales entre las fuentes (según la referencia) y atribuye correctamente cada afirmación a la fuente que la sostiene, sin inventar ni omitir contradicciones?
- 1.0: Identifica todas las contradicciones reales de la referencia y las atribuye correctamente a cada fuente
- 0.7: Identifica las contradicciones principales con alguna atribución imprecisa o un matiz faltante
- 0.5: Identifica solo una parte de las contradicciones, o las señala de forma vaga sin atribución clara a las fuentes
- 0.3: Detecta que hay discrepancias pero no logra identificar cuáles son, o atribuye mal la mayoría
- 0: No detecta ninguna contradicción real, o inventa una contradicción que no existe según la referencia

COMPLETITUD FUNCIONAL: ¿Integra correctamente la información consistente (el consenso) entre las tres fuentes, sin omitir aspectos relevantes?
- 1.0: Integra todo el consenso relevante de forma completa
- 0.7: Integra el consenso con alguna omisión menor
- 0.5: Integra el consenso de forma parcial
- 0.3: Integración muy incompleta del consenso
- 0: No integra el consenso relevante

CLARIDAD COMUNICACIONAL: ¿Qué tan clara, ordenada y fácil de seguir es la síntesis?
- 1.0: Muy clara, ordenada y fácil de seguir
- 0.7: Clara con detalles mejorables
- 0.5: Comprensible pero confusa por momentos
- 0.3: Difícil de seguir
- 0: Incoherente

PROFUNDIDAD CONCEPTUAL: ¿Refleja una comprensión adecuada de cómo se relacionan las tres fuentes entre sí, más allá de listar contradicciones aisladas (incluyendo si alguna fuente reconcilia parcialmente a las otras, cuando corresponda)?
- 1.0: Comprensión profunda; distingue contradicciones genuinas de discrepancias reconciliables
- 0.7: Buena comprensión con matices menores
- 0.5: Comprensión básica de las relaciones entre fuentes
- 0.3: Comprensión superficial
- 0: Sin comprensión útil de las relaciones entre fuentes

RESTRICCIONES DE PUNTUACIÓN:
- Usa únicamente estos valores: 0, 0.3, 0.5, 0.7, 1
- No inventes valores intermedios
- Si dudas entre dos valores, elige uno de esos dos permitidos

RECORDATORIO ANTES DE RESPONDER (releé la referencia del principio si hace falta):
- La REFERENCIA de más arriba es la que define qué contradicciones son reales y cuál es el consenso esperado; no evalúes "a ojo" sin volver a chequearla.
- Tu JSON debe incluir una entrada en "evaluations" para CADA UNA de estas {n_techniques} técnicas, sin omitir ninguna: {technique_list}.

Responde en formato JSON con estructura:
{{
  "ranking": ["mejor_técnica", "segunda", "tercera", "peor_técnica"],
  "evaluations": {{
    "nombre_técnica": {{
            "exactitud_funcional": 0.7,
            "completitud_funcional": 0.7,
            "claridad_comunicacional": 0.5,
            "profundidad_conceptual": 0.3
    }},
    ...
  }}
}}

Solo devuelve el JSON, sin explicaciones adicionales."""


def _build_judge_prompt(input_text: str, reference: str, responses_dict: dict[str, dict]) -> str:
    """Arma el prompt de evaluación comparativa para un input dado.

    Incluye el texto completo (las 3 fuentes + consigna, sin truncar) y la
    referencia de contradicciones/consenso, indispensable para que el juez
    pueda verificar la exactitud sin tener que descubrir las contradicciones
    desde cero.
    """
    responses_text = "\n\n".join(
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
        input_text=input_text,
        reference=reference,
        responses_text=responses_text,
        n_techniques=len(responses_dict),
        technique_list=", ".join(responses_dict.keys()),
    )


def _extract_json_object(text: str) -> str | None:
    """Extrae el primer objeto JSON balanceado del texto, contando llaves.

    Más robusto que un regex greedy: corta en la llave de cierre que
    realmente balancea la primera llave de apertura, en vez de tomar todo
    hasta la última '}' del texto (lo cual se rompe si el juez agrega
    cualquier comentario con llaves fuera del bloque JSON).
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_judge_response(response_text: str, responses_dict: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Extrae y normaliza las 4 dimensiones de calidad para cada técnica.

    Devuelve además la lista de técnicas que el juez NO incluyó en su
    respuesta (completadas con puntaje 0 como fallback), para que quien
    llama pueda decidir si reintentar la consulta en vez de aceptar el 0
    silencioso.
    """
    results: dict[str, dict] = {}
    try:
        json_text = _extract_json_object(response_text)
        if json_text:
            evaluation_data = json.loads(json_text)
            evaluations = evaluation_data.get("evaluations", {})
            for tech, scores in evaluations.items():
                if tech in responses_dict:
                    results[tech] = normalize_evaluation_scores(scores)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Error parseando evaluación del juez: %s", exc)

    missing = [tech for tech in responses_dict if tech not in results]
    for tech in missing:
        results[tech] = _empty_scores()

    return results, missing


def _evaluate_with_retries(
    call_judge: Callable[[], str],
    judge_name: str,
    responses_dict: dict[str, dict],
) -> dict[str, dict]:
    """Llama al juez y reintenta si omitió alguna técnica en su respuesta."""
    results: dict[str, dict] = {}
    missing: list[str] = list(responses_dict.keys())

    for attempt in range(1, MAX_JUDGE_RETRIES + 1):
        response_text = call_judge()
        results, missing = _parse_judge_response(response_text, responses_dict)
        if not missing:
            return results
        logger.warning(
            "Intento %d/%d: %s no evaluó estas técnicas: %s. Reintentando...",
            attempt, MAX_JUDGE_RETRIES, judge_name, ", ".join(missing),
        )

    logger.warning(
        "%s no evaluó %s tras %d intentos; se completa con 0 en las 4 dimensiones.",
        judge_name, ", ".join(missing), MAX_JUDGE_RETRIES,
    )
    return results


def evaluate_with_gpt4o(input_text: str, reference: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando GPT-4o como juez."""
    prompt = _build_judge_prompt(input_text, reference, responses_dict)

    def _call() -> str:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL_OPENAI,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    return _evaluate_with_retries(_call, "GPT-4o", responses_dict)


def evaluate_with_claude(input_text: str, reference: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando Claude Haiku 4.5 como juez."""
    prompt = _build_judge_prompt(input_text, reference, responses_dict)

    def _call() -> str:
        response = claude_client.messages.create(
            model=JUDGE_MODEL_CLAUDE,
            max_tokens=20000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    return _evaluate_with_retries(_call, "Claude Haiku 4.5", responses_dict)


def evaluate_comparative(input_text: str, reference: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Combina las evaluaciones de ambos jueces y calcula el promedio por dimensión."""
    logger.info("  Evaluando con GPT-4o...")
    gpt_results = evaluate_with_gpt4o(input_text, reference, responses_dict)

    logger.info("  Evaluando con Claude Haiku 4.5...")
    claude_results = evaluate_with_claude(input_text, reference, responses_dict)

    combined: dict[str, dict] = {}
    for tech in responses_dict:
        gpt_scores = gpt_results.get(tech, _empty_scores())
        claude_scores = claude_results.get(tech, _empty_scores())
        combined[tech] = {
            "gpt4o": gpt_scores,
            "claude": claude_scores,
            "promedio": {
                dim: (gpt_scores.get(dim, 0) + claude_scores.get(dim, 0)) / 2
                for dim in QUALITY_DIMENSIONS
            },
        }
    return combined


# =============================================================================
# Ejecución de una corrida completa del experimento
# =============================================================================

def _generate_responses() -> tuple[list[dict], dict[str, dict]]:
    """Genera las respuestas de las 6 técnicas para cada input con gpt-4o-mini.

    El modelo generador solo recibe item["text"] (las 3 fuentes + la
    consigna de síntesis); la referencia de contradicciones nunca se le
    muestra.
    """
    results: list[dict] = []
    responses_by_input: dict[str, dict] = {}

    for technique_name, build_prompt in TECHNIQUES.items():
        for item in INPUTS:
            text = item["text"]
            prompt = build_prompt(text)

            response = openai_client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            output = response.choices[0].message.content
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens

            logger.info(
                "[%s] %.50s... | In: %d tok | Out: %d tok",
                technique_name, text, tokens_in, tokens_out,
            )

            results.append({
                "input": text,
                "technique": technique_name,
                "prompt": prompt,
                "output": output,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            })

            responses_by_input.setdefault(text, {"responses": {}})
            responses_by_input[text]["responses"][technique_name] = {
                "prompt": prompt,
                "output": output,
            }

    return results, responses_by_input


def _attach_evaluations(results: list[dict], responses_by_input: dict[str, dict]) -> None:
    """Evalúa cada input de forma comparativa y adjunta los puntajes a `results` in-place."""
    for input_text, input_data in responses_by_input.items():
        logger.info("Evaluando input: %.50s...", input_text)
        reference = INPUT_REFERENCE_BY_TEXT[input_text]
        evaluations = evaluate_comparative(input_text, reference, input_data["responses"])

        for record in results:
            if record["input"] != input_text:
                continue
            eval_data = evaluations.get(record["technique"])
            if not eval_data:
                continue

            record["gpt4o_evaluacion"] = eval_data["gpt4o"]
            record["claude_evaluacion"] = eval_data["claude"]
            promedio = eval_data["promedio"]
            for dim in QUALITY_DIMENSIONS:
                record[dim] = promedio.get(dim, 0)
            record["calidad"] = sum(record[dim] for dim in QUALITY_DIMENSIONS) / len(QUALITY_DIMENSIONS)


def _build_summary(results: list[dict]) -> dict[str, Any]:
    """Calcula tokens, calidad y eficiencia agregados por técnica a partir de `results`."""
    techniques_stats: dict[str, dict] = {}
    technique_tokens: dict[str, dict] = {}
    json_results: list[dict] = []
    total_tokens_used = 0

    for record in results:
        total_tokens_used += record["total_tokens"]
        tech = record["technique"]

        tokens = technique_tokens.setdefault(tech, {"total_input": 0, "total_output": 0, "count": 0})
        tokens["total_input"] += record["tokens_input"]
        tokens["total_output"] += record["tokens_output"]
        tokens["count"] += 1

        stats = techniques_stats.setdefault(tech, {
            "count": 0,
            "total_tokens": 0,
            "total_calidad": 0.0,
            "total_calidad_gpt4o": 0.0,
            "total_calidad_claude": 0.0,
            **{f"total_{dim}": 0.0 for dim in QUALITY_DIMENSIONS},
        })

        gpt4o_calidad = sum(record["gpt4o_evaluacion"].values()) / len(record["gpt4o_evaluacion"])
        claude_calidad = sum(record["claude_evaluacion"].values()) / len(record["claude_evaluacion"])

        stats["count"] += 1
        stats["total_tokens"] += record["total_tokens"]
        stats["total_calidad"] += record["calidad"]
        stats["total_calidad_gpt4o"] += gpt4o_calidad
        stats["total_calidad_claude"] += claude_calidad
        for dim in QUALITY_DIMENSIONS:
            stats[f"total_{dim}"] += record[dim]

        json_results.append({
            "input": record["input"],
            "technique": tech,
            "output": record["output"],
            "tokens_input": int(record["tokens_input"]),
            "tokens_output": int(record["tokens_output"]),
            "total_tokens": int(record["total_tokens"]),
            "evaluacion_gpt4o": {dim: round(record["gpt4o_evaluacion"][dim], 2) for dim in QUALITY_DIMENSIONS},
            "evaluacion_claude": {dim: round(record["claude_evaluacion"][dim], 2) for dim in QUALITY_DIMENSIONS},
            "promedio": {
                **{dim: round(record[dim], 2) for dim in QUALITY_DIMENSIONS},
                "calidad": round(record["calidad"], 2),
            },
        })

    efficiency: dict[str, dict] = {}
    for tech, stats in techniques_stats.items():
        count = stats["count"]
        stats["avg_calidad"] = round(stats["total_calidad"] / count, 2)
        stats["avg_calidad_gpt4o"] = round(stats["total_calidad_gpt4o"] / count, 2)
        stats["avg_calidad_claude"] = round(stats["total_calidad_claude"] / count, 2)
        for dim in QUALITY_DIMENSIONS:
            stats[f"avg_{dim}"] = round(stats[f"total_{dim}"] / count, 2)

        tokens = stats["total_tokens"]
        efficiency[tech] = {
            "gpt4o": round(stats["avg_calidad_gpt4o"] / tokens, 6) if tokens else 0,
            "claude": round(stats["avg_calidad_claude"] / tokens, 6) if tokens else 0,
            "promedio": round(stats["avg_calidad"] / tokens, 6) if tokens else 0,
        }

    return {
        "results": json_results,
        "summary": {
            "total_results": len(json_results),
            "total_tokens_used": int(total_tokens_used),
            "techniques_stats": {
                tech: {
                    "count": stats["count"],
                    "tokens_input": int(technique_tokens[tech]["total_input"]),
                    "tokens_output": int(technique_tokens[tech]["total_output"]),
                    "total_tokens": int(stats["total_tokens"]),
                    "avg_calidad_promedio": stats["avg_calidad"],
                    "avg_calidad_gpt4o": stats["avg_calidad_gpt4o"],
                    "avg_calidad_claude": stats["avg_calidad_claude"],
                    "dimensions": {dim: stats[f"avg_{dim}"] for dim in QUALITY_DIMENSIONS},
                    "efficiency_scores": efficiency[tech],
                }
                for tech, stats in techniques_stats.items()
            },
        },
    }


def run_single_experiment(run_index: int, output_dir: Path, output_prefix: str) -> Path:
    """Corre el experimento completo una vez y guarda el resultado en disco."""
    logger.info("=" * 60)
    logger.info("CORRIDA %d", run_index)
    logger.info("=" * 60)

    results, responses_by_input = _generate_responses()
    _attach_evaluations(results, responses_by_input)
    payload = _build_summary(results)

    output_path = output_dir / f"{output_prefix}.{run_index}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Resultados guardados en %s", output_path)
    ranking = sorted(
        payload["summary"]["techniques_stats"].items(),
        key=lambda item: item[1]["avg_calidad_promedio"],
        reverse=True,
    )
    for position, (tech, stats) in enumerate(ranking, start=1):
        logger.info(
            "  %d. %-28s calidad=%.2f  eficiencia=%.6f",
            position, tech, stats["avg_calidad_promedio"], stats["efficiency_scores"]["promedio"],
        )

    return output_path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--num-runs", type=int, default=DEFAULT_NUM_RUNS,
        help=f"Cantidad de corridas independientes a ejecutar (default: {DEFAULT_NUM_RUNS})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Carpeta donde se guardan los archivos de resultados (default: carpeta actual)",
    )
    parser.add_argument(
        "--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX,
        help=f"Prefijo de los archivos generados (default: '{DEFAULT_OUTPUT_PREFIX}')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [
        run_single_experiment(run_index, args.output_dir, args.output_prefix)
        for run_index in range(args.num_runs)
    ]

    logger.info("=" * 60)
    logger.info("%d corridas completadas", args.num_runs)
    logger.info("=" * 60)
    for path in generated_files:
        logger.info("  - %s", path)


if __name__ == "__main__":
    main()