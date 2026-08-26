"""Experimento 5 — Resolución de problemas matemáticos (tarea intermedia).

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Evalúa seis técnicas de prompting (Zero-shot, Few-shot, Chain-of-Thought,
Chain-of-Thought limitado, Structured Prompting JSON y Role Prompting) sobre
el modelo gpt-4o-mini, en una tarea de resolución de problemas matemáticos de
porcentajes y repartos proporcionales. La calidad de cada respuesta se evalúa
mediante un esquema de doble juez (GPT-4o y Claude Haiku 4.5) en cuatro
dimensiones: exactitud funcional, completitud
funcional, claridad comunicacional y profundidad conceptual.

El experimento se repite ``--num-runs`` veces (10 por defecto) para poder
calcular la variabilidad de los resultados entre corridas. Cada corrida se
guarda como un archivo JSON independiente: ``results5.0.json``,
``results5.1.json``, ..., ``results5.9.json``.

Uso:
    python experimento5_matematica.py

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
import re
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from openai import OpenAI

# =============================================================================
# Configuración
# =============================================================================

# Descripción de la tarea, usada en el prompt del juez y en los logs.
TASK_DESCRIPTION = "resolución de problemas matemáticos"

# Criterio de exactitud funcional específico de esta tarea. A diferencia de
# traducción (donde la exactitud es graduable porque el significado se puede
# preservar "más o menos"), un problema matemático tiene un único resultado
# final correcto, igual que las tareas simples. Por eso exactitud_funcional
# vuelve a ser binaria acá: evalúa solo si el resultado final es correcto,
# no la calidad del razonamiento (eso ya lo capturan completitud_funcional
# y profundidad_conceptual, que sí son graduadas).
EXACTITUD_CRITERIO = "¿Es correcto el resultado final del problema?"

# Modelo generador y modelos jueces.
GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL_OPENAI = "gpt-4o"
JUDGE_MODEL_CLAUDE = "claude-haiku-4-5-20251001"

# Las 4 dimensiones de calidad usadas en toda la tesina.
# Se definen en un único lugar para que el prompt del juez y el parseo de su
# respuesta usen siempre las mismas claves.
QUALITY_DIMENSIONS = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]

ALLOWED_LIKERT_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
ALLOWED_BINARY_VALUES = [0.0, 1.0]

# Escala permitida para exactitud_funcional en ESTA tarea: binaria (ver nota
# en EXACTITUD_CRITERIO más arriba).
EXACTITUD_ALLOWED_VALUES = ALLOWED_BINARY_VALUES

# Tareas de entrada evaluadas: 3 problemas de porcentajes y repartos
# proporcionales.
INPUTS = [
    "Un comercio aplica primero un 20% de descuento sobre un producto de $250. "
    "Luego agrega un impuesto del 21%. Si el cliente paga en efectivo recibe "
    "además un descuento adicional del 5% sobre el precio final. ¿Cuánto "
    "termina pagando?",
    "Una cooperativa recibe una donación de $36.000 que debe repartirse entre "
    "tres escuelas según la cantidad de estudiantes. La Escuela A tiene 180 "
    "alumnos, la Escuela B tiene 240 alumnos y la Escuela C tiene 300 alumnos. "
    "¿Cuánto dinero recibe cada escuela?",
    "Una empresa reparte un bono de $48.000 entre tres empleados según las "
    "horas trabajadas durante el mes. Ana trabajó 120 horas, Bruno 150 horas "
    "y Carla 210 horas. ¿Cuánto recibe cada uno?",
]

DEFAULT_NUM_RUNS = 10
DEFAULT_OUTPUT_PREFIX = "results5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experimento5")


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

def zero_shot(text: str) -> str:
    return f"Resuelve el siguiente problema matemático:\n\n{text}"


def few_shot(text: str) -> str:
    examples = (
        "Ejemplo:\n"
        "Un artículo de $50 con 10% de descuento: $50 * 0,9 = $45\n"
        "Luego con 5% de impuesto: $45 * 1,05 = $47,25\n"
        "Resultado: $47,25\n\n"
    )
    return f"{examples}Ahora resuelve el siguiente problema matemático:\n\n{text}"


def cot(text: str) -> str:
    return (
        "Resuelve el siguiente problema matemático.\n"
        "Explica tu razonamiento paso a paso y luego da la respuesta final:\n\n"
        f"{text}"
    )


def cot_limited(text: str) -> str:
    return (
        "Resuelve el siguiente problema matemático.\n"
        "Explica tu razonamiento y limitalo a máximo 100 palabras; "
        f"luego da la respuesta final:\n\n{text}"
    )


def structured_prompting_json(text: str) -> str:
    prompt = {
        "instruction": "Resuelve el siguiente problema matemático.",
        "input": text,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text: str) -> str:
    return (
        "Actúa como un profesor de matemática.\n"
        "Resuelve el siguiente problema matemático.\n\n"
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

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para {task_description}:

EXACTITUD FUNCIONAL: {exactitud_criterio}
- Para esta tarea, usa solo 1 si el resultado final es correcto y 0 si es incorrecto.

COMPLETITUD FUNCIONAL: ¿Incluye los pasos necesarios para reproducir la solución?
- 1.0: Completa, con todos los pasos esenciales
- 0.7: Casi completa, con omisiones menores
- 0.5: Incompleta en aspectos importantes
- 0.3: Muy incompleta
- 0: No permite reproducir la solución

CLARIDAD COMUNICACIONAL: ¿Qué tan clara y ordenada es la explicación?
- 1.0: Muy clara, ordenada y fácil de seguir
- 0.7: Clara con detalles mejorables
- 0.5: Comprensible pero confusa por momentos
- 0.3: Difícil de seguir
- 0: Incoherente

PROFUNDIDAD CONCEPTUAL: ¿El nivel de razonamiento matemático es adecuado al problema?
- 1.0: Razonamiento sólido y bien justificado
- 0.7: Buen razonamiento con pequeñas carencias
- 0.5: Básico pero funcional
- 0.3: Superficial
- 0: Sin razonamiento útil

RESTRICCIONES DE PUNTUACIÓN:
- Usa únicamente estos valores: 0, 0.3, 0.5, 0.7, 1
- No inventes valores intermedios
- Si dudas entre dos valores, elige uno de esos dos permitidos

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


def _build_judge_prompt(input_text: str, responses_dict: dict[str, dict]) -> str:
    """Arma el prompt de evaluación comparativa para un input dado.

    Usa el enunciado completo del problema (sin truncar): el juez necesita
    ver los datos numéricos completos para poder verificar si el resultado
    final de cada técnica es correcto, no solo las salidas generadas.
    """
    responses_text = "\n\n".join(
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
        exactitud_criterio=EXACTITUD_CRITERIO,
        input_text=input_text,
        responses_text=responses_text,
    )


def _parse_judge_response(response_text: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Extrae y normaliza las 4 dimensiones de calidad para cada técnica.

    Usa siempre las claves canónicas de QUALITY_DIMENSIONS, tanto al leer el
    JSON del juez como al completar los valores faltantes, para evitar que
    una dimensión quede silenciosamente en 0 por un desajuste de nombres.
    """
    results: dict[str, dict] = {}
    try:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            evaluation_data = json.loads(json_match.group())
            evaluations = evaluation_data.get("evaluations", {})
            for tech, scores in evaluations.items():
                if tech in responses_dict:
                    results[tech] = normalize_evaluation_scores(scores)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Error parseando evaluación del juez: %s", exc)

    for tech in responses_dict:
        results.setdefault(tech, _empty_scores())

    return results


def evaluate_with_gpt4o(input_text: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando GPT-4o como juez."""
    prompt = _build_judge_prompt(input_text, responses_dict)
    response = openai_client.chat.completions.create(
        model=JUDGE_MODEL_OPENAI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _parse_judge_response(response.choices[0].message.content.strip(), responses_dict)


def evaluate_with_claude(input_text: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando Claude Haiku 4.5 como juez."""
    prompt = _build_judge_prompt(input_text, responses_dict)
    response = claude_client.messages.create(
        model=JUDGE_MODEL_CLAUDE,
        max_tokens=20000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_response(response.content[0].text.strip(), responses_dict)


def evaluate_comparative(input_text: str, responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Combina las evaluaciones de ambos jueces y calcula el promedio por dimensión."""
    logger.info("  Evaluando con GPT-4o...")
    gpt_results = evaluate_with_gpt4o(input_text, responses_dict)

    logger.info("  Evaluando con Claude Haiku 4.5...")
    claude_results = evaluate_with_claude(input_text, responses_dict)

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
    """Genera las respuestas de las 6 técnicas para cada input con gpt-4o-mini."""
    results: list[dict] = []
    responses_by_input: dict[str, dict] = {}

    for technique_name, build_prompt in TECHNIQUES.items():
        for text in INPUTS:
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
        evaluations = evaluate_comparative(input_text, input_data["responses"])

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
    """Corre el experimento completo una vez y guarda el resultado en disco.

    Args:
        run_index: Número de corrida (0-indexado), usado en el nombre del archivo.
        output_dir: Carpeta donde se guarda el archivo de resultados.
        output_prefix: Prefijo del archivo (ej. "results5" -> "results5.0.json").

    Returns:
        Ruta del archivo JSON generado.
    """
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