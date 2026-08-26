"""Experimento 7 — Planificación con restricciones / scheduling (tarea compleja).

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Evalúa seis técnicas de prompting (Zero-shot, Few-shot, Chain-of-Thought,
Chain-of-Thought limitado, Structured Prompting JSON y Role Prompting) sobre
el modelo gpt-4o-mini, en una tarea de planificación de cronogramas con
restricciones de dependencias, recursos compartidos y ventana de tiempo
disponible (scheduling). A diferencia de las tareas anteriores, cada input
tiene una solución de referencia (cronograma válido, o el cálculo que
demuestra infactibilidad) que se inyecta ÚNICAMENTE en el prompt del juez,
nunca en el prompt que recibe el modelo generador: sin esa referencia, el
juez debe razonar sobre restricciones de dependencias/recursos/paralelismo
desde cero, lo cual introduce el mismo riesgo de error que se busca medir en
el modelo evaluado y degrada el acuerdo inter-juez.

La calidad de cada respuesta se evalúa mediante un esquema de doble juez
(GPT-4o y Claude Haiku 4.5) en cuatro dimensiones:
exactitud funcional, completitud funcional, claridad comunicacional y
profundidad conceptual. exactitud_funcional es BINARIA en esta tarea: un
cronograma cumple todas las restricciones (o la infactibilidad está bien
justificada) o no, no hay grados intermedios de "más o menos válido".

El experimento se repite ``--num-runs`` veces (10 por defecto) para poder
calcular la variabilidad de los resultados entre corridas. Cada corrida se
guarda como un archivo JSON independiente: ``results7.0.json``,
``results7.1.json``, ..., ``results7.9.json``.

Uso:
    python experimento7.py

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

TASK_DESCRIPTION = "planificación de cronogramas con restricciones (scheduling)"

# Criterio de exactitud funcional: binario, igual que en matemática, porque
# un cronograma tiene una respuesta objetivamente verificable (cumple todas
# las restricciones, o la infactibilidad está correctamente justificada).
EXACTITUD_CRITERIO = (
    "¿El cronograma propuesto cumple TODAS las restricciones de dependencias, "
    "recursos y ventana de tiempo, o identifica y justifica correctamente que "
    "no existe cronograma factible cuando ese es el caso?"
)

GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL_OPENAI = "gpt-4o"
JUDGE_MODEL_CLAUDE = "claude-haiku-4-5-20251001"

# Cantidad máxima de intentos por juez cuando omite alguna técnica en su
# respuesta (prompt largo -> más riesgo de que el juez "se olvide" de incluir
# una entrada en "evaluations"). Si tras estos intentos sigue faltando, se
# completa esa técnica con 0 en las 4 dimensiones y queda logueado.
MAX_JUDGE_RETRIES = 3

QUALITY_DIMENSIONS = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]

ALLOWED_LIKERT_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
ALLOWED_BINARY_VALUES = [0.0, 1.0]

# Escala permitida para exactitud_funcional en ESTA tarea: binaria.
EXACTITUD_ALLOWED_VALUES = ALLOWED_BINARY_VALUES

# Tareas de entrada evaluadas: 3 problemas de scheduling, cada uno con una
# solución de referencia para el juez. El campo "reference" NUNCA se incluye
# en el prompt que recibe el modelo generador (gpt-4o-mini) — solo se usa
# para construir el prompt de evaluación de los jueces.
#
# Diseño del set: input 1 es factible y ajustado (sin holgura), input 2 es
# infactible (el modelo debe detectarlo, no inventar un cronograma que
# "cierra" mal), input 3 es factible pero exige paralelizar dos tareas para
# entrar en la ventana disponible.
INPUTS: list[dict[str, str]] = [
    {
        "text": (
            "Cinco tareas de un despliegue de software: A (backup, 2h), "
            "B (migración de BD, 3h, requiere A terminada), "
            "C (actualización de API, 2h, requiere B), "
            "D (pruebas de regresión, 4h, requiere C), "
            "E (comunicación a usuarios, 1h, debe terminar antes de que empiece B). "
            "Hay una sola persona para A, B y C (no pueden solaparse entre sí); "
            "D y E pueden hacerlas otras personas en paralelo con cualquier tarea. "
            "La ventana de mantenimiento va de 22:00 a 09:00. Indicá hora de "
            "inicio y fin de cada tarea, o justificá que no entra en la ventana."
        ),
        "reference": (
            "Cronograma válido: A 22:00-00:00, B 00:00-03:00, C 03:00-05:00, "
            "D 05:00-09:00; E en cualquier momento entre 22:00 y 00:00 (antes de "
            "que empiece B). La cadena obligatoria A->B->C->D dura exactamente "
            "2+3+2+4 = 11h, igual a la ventana completa (22:00 a 09:00): no hay "
            "holgura, así que cualquier corrimiento de horario invalida el "
            "cronograma. Una respuesta correcta debe respetar estos horarios "
            "(o un corrimiento equivalente sin holgura) y la restricción de E."
        ),
    },
    {
        "text": (
            "Migración de un datacenter: A (inventario y etiquetado, 3h), "
            "B (apagado ordenado de servidores, 2h, requiere A), "
            "C (traslado físico, 4h, requiere B), "
            "D (reencendido y verificación, 3h, requiere C), "
            "E (prueba de conectividad de red, 2h, requiere C y debe terminar "
            "antes de que empiece D). Todas las tareas son secuenciales sobre "
            "el mismo equipamiento (ninguna puede solaparse con otra). La "
            "ventana es de 20:00 a 08:00. Indicá hora de inicio y fin de cada "
            "tarea, o justificá que no entra en la ventana."
        ),
        "reference": (
            "NO existe cronograma factible. La cadena obligatoria es "
            "A->B->C->E->D (E requiere C y debe terminar antes de D), con "
            "duración total 3+2+4+2+3 = 14h. La ventana disponible es de 12h "
            "(20:00 a 08:00). Como 14h > 12h, no hay forma de completar la "
            "secuencia dentro de la ventana. Una respuesta correcta DEBE "
            "concluir explícitamente que es infactible y mostrar un cálculo "
            "equivalente a este (duración total de la cadena vs. duración de "
            "la ventana). Un cronograma que asigna horarios sin señalar la "
            "infactibilidad es incorrecto, sin importar qué tan prolijo se vea."
        ),
    },
    {
        "text": (
            "Montaje de un evento con dos técnicos disponibles: "
            "A (armado de escenario, 3h, requiere a los dos técnicos juntos), "
            "B (instalación de sonido, 2h, requiere A terminada, ocupa un "
            "técnico), C (instalación de luces, 2h, requiere A terminada, "
            "ocupa un técnico), D (prueba general, 1h, requiere B y C "
            "terminadas, requiere a los dos técnicos). La ventana de trabajo "
            "es de 08:00 a 15:00. Indicá hora de inicio y fin de cada tarea y "
            "qué técnico(s) la ejecutan, o justificá que no alcanza el tiempo."
        ),
        "reference": (
            "Cronograma válido: A 08:00-11:00 (ambos técnicos juntos), luego "
            "B y C EN PARALELO de 11:00 a 13:00 (un técnico distinto en cada "
            "una), y D 13:00-14:00 (ambos técnicos juntos). El punto clave es "
            "que B y C pueden y deben paralelizarse porque hay dos técnicos "
            "disponibles y ninguna depende de la otra. Si un modelo serializa "
            "B y C (una después de la otra), el cronograma termina a las "
            "16:00, fuera de la ventana (08:00-15:00), y concluir que es "
            "infactible en ese caso sería INCORRECTO: la tarea sí entra en la "
            "ventana si se paralelizan B y C."
        ),
    },
]

DEFAULT_NUM_RUNS = 10
DEFAULT_OUTPUT_PREFIX = "results7"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experimento7")


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
# IMPORTANTE: ninguna de estas funciones recibe ni expone la solución de
# referencia. El modelo generador solo ve el enunciado del problema, igual
# que un usuario real.

def zero_shot(text: str) -> str:
    return f"Resolvé el siguiente problema de planificación de tareas:\n\n{text}"


def few_shot(text: str) -> str:
    # NOTA METODOLÓGICA: el ejemplo es un caso factible. Esto puede introducir
    # una asimetría hacia "siempre hay cronograma" que perjudique la
    # detección de infactibilidad (input 2). Queda documentado como
    # limitación en vez de corregido, tal como se decidió al diseñar la tarea.
    examples = (
        "Ejemplo:\n"
        "Problema: Dos tareas, X (1h) e Y (1h, requiere X terminada), ventana "
        "de 08:00 a 10:00.\n"
        "Solución: X 08:00-09:00, Y 09:00-10:00. Cronograma válido, sin "
        "holgura.\n\n"
    )
    return f"{examples}Ahora resolvé el siguiente problema de planificación de tareas:\n\n{text}"


def cot(text: str) -> str:
    return (
        "Resolvé el siguiente problema de planificación de tareas.\n"
        "Explicá tu razonamiento paso a paso y luego da la respuesta "
        "final:\n\n"
        f"{text}"
    )


def cot_limited(text: str) -> str:
    return (
        "Resolvé el siguiente problema de planificación de tareas.\n"
        "Explicá tu razonamiento y limitalo a máximo 100 palabras; "
        f"luego da la respuesta final:\n\n{text}"
    )


def structured_prompting_json(text: str) -> str:
    prompt = {
        "instruction": "Resolvé el siguiente problema de planificación de tareas.",
        "input": text,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text: str) -> str:
    return (
        "Actuá como un gestor de proyectos experto en planificación de "
        "tareas con restricciones de recursos y dependencias.\n"
        "Resolvé el siguiente problema:\n\n"
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

SOLUCIÓN DE REFERENCIA (esto es solo para tu evaluación; el modelo evaluado NUNCA vio esta referencia):
{reference_solution}

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones, usando la solución de referencia para verificar la corrección.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para {task_description}:

EXACTITUD FUNCIONAL: {exactitud_criterio}
- Para esta tarea, usa solo 1 si el cronograma es válido (o la infactibilidad está correctamente justificada, según la solución de referencia) y 0 si es incorrecto.

COMPLETITUD FUNCIONAL: ¿Especifica horario de inicio y fin (y recursos/personas asignadas cuando corresponda) para cada tarea, o la justificación completa de infactibilidad?
- 1.0: Completa, especifica todo lo necesario para verificar la solución
- 0.7: Casi completa, con omisiones menores (ej. falta un horario puntual)
- 0.5: Incompleta en aspectos importantes
- 0.3: Muy incompleta
- 0: No permite verificar ni reproducir la solución

CLARIDAD COMUNICACIONAL: ¿Qué tan clara y ordenada es la explicación del cronograma o de la infactibilidad?
- 1.0: Muy clara, ordenada y fácil de seguir
- 0.7: Clara con detalles mejorables
- 0.5: Comprensible pero confusa por momentos
- 0.3: Difícil de seguir
- 0: Incoherente

PROFUNDIDAD CONCEPTUAL: ¿El razonamiento sobre dependencias, recursos compartidos y posibilidad de paralelismo es explícito y correcto?
- 1.0: Razonamiento sólido, considera dependencias, recursos y paralelismo correctamente
- 0.7: Buen razonamiento con pequeñas carencias
- 0.5: Básico pero funcional
- 0.3: Superficial, ignora alguna restricción relevante
- 0: Sin razonamiento útil

RESTRICCIONES DE PUNTUACIÓN:
- Usa únicamente estos valores: 0, 0.3, 0.5, 0.7, 1
- No inventes valores intermedios
- Si dudas entre dos valores, elige uno de esos dos permitidos

RECORDATORIO ANTES DE RESPONDER (releé la solución de referencia del principio si hace falta):
- La SOLUCIÓN DE REFERENCIA de más arriba es la que define qué cronograma es válido (o si la infactibilidad está bien justificada); no evalúes "a ojo" la corrección sin volver a chequearla contra esa referencia.
- Tu JSON debe incluir una entrada en "evaluations" para CADA UNA de estas {n_techniques} técnicas, sin omitir ninguna: {technique_list}.

Responde en formato JSON con estructura:
{{
  "ranking": ["mejor_técnica", "segunda", "tercera", "peor_técnica"],
  "evaluations": {{
    "nombre_técnica": {{
            "exactitud_funcional": 1,
            "completitud_funcional": 0.7,
            "claridad_comunicacional": 0.5,
            "profundidad_conceptual": 0.3
    }},
    ...
  }}
}}

Solo devuelve el JSON, sin explicaciones adicionales."""


def _build_judge_prompt(input_text: str, reference_solution: str, responses_dict: dict[str, dict]) -> str:
    """Arma el prompt de evaluación comparativa para un input dado.

    Incluye el enunciado completo (sin truncar) y la solución de referencia,
    indispensable en esta tarea para que el juez pueda verificar dependencias,
    recursos y paralelismo sin tener que resolver el problema desde cero.
    """
    responses_text = "\n\n".join(
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
        exactitud_criterio=EXACTITUD_CRITERIO,
        input_text=input_text,
        reference_solution=reference_solution,
        responses_text=responses_text,
        n_techniques=len(responses_dict),
        technique_list=", ".join(responses_dict.keys()),
    )


def _extract_json_object(text: str) -> str | None:
    """Extrae el primer objeto JSON balanceado del texto, contando llaves.

    Más robusto que un regex greedy (``\\{[\\s\\S]*\\}``): ese regex toma todo
    desde la primera '{' hasta la ÚLTIMA '}' del texto, lo cual se rompe si el
    juez agrega cualquier texto con llaves fuera del bloque JSON (más probable
    en esta tarea, donde el prompt es largo y el juez a veces agrega comentarios).
    Esta función en cambio corta en la llave de cierre que realmente balancea
    la primera llave de apertura.
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
    respuesta (completadas con puntaje 0 como fallback), para que quien llama
    pueda decidir si reintentar la consulta en vez de aceptar el 0 silencioso.
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
    """Llama al juez y reintenta si omitió alguna técnica en su respuesta.

    ``call_judge`` recibe el prompt y devuelve el texto crudo de la respuesta
    del juez (ya extraído de la estructura propia de cada API). Reintenta
    hasta MAX_JUDGE_RETRIES veces antes de aceptar el fallback en 0.
    """
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


def evaluate_with_gpt4o(input_text: str, reference_solution: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando GPT-4o como juez.

    Reintenta hasta MAX_JUDGE_RETRIES veces si el juez omite alguna técnica.
    """
    prompt = _build_judge_prompt(input_text, reference_solution, responses_dict)

    def _call() -> str:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL_OPENAI,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    return _evaluate_with_retries(_call, "GPT-4o", responses_dict)


def evaluate_with_claude(input_text: str, reference_solution: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando Claude Haiku 4.5 como juez.

    Reintenta hasta MAX_JUDGE_RETRIES veces si el juez omite alguna técnica.
    """
    prompt = _build_judge_prompt(input_text, reference_solution, responses_dict)

    def _call() -> str:
        response = claude_client.messages.create(
            model=JUDGE_MODEL_CLAUDE,
            max_tokens=20000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    return _evaluate_with_retries(_call, "Claude Haiku 4.5", responses_dict)


def evaluate_comparative(input_text: str, reference_solution: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Combina las evaluaciones de ambos jueces y calcula el promedio por dimensión."""
    logger.info("  Evaluando con GPT-4o...")
    gpt_results = evaluate_with_gpt4o(input_text, reference_solution, responses_dict)

    logger.info("  Evaluando con Claude Haiku 4.5...")
    claude_results = evaluate_with_claude(input_text, reference_solution, responses_dict)

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

    El modelo generador solo recibe item["text"]; la referencia no se le
    muestra nunca.
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
        reference_solution = INPUT_REFERENCE_BY_TEXT[input_text]
        evaluations = evaluate_comparative(input_text, reference_solution, input_data["responses"])

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