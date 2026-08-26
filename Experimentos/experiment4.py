"""Experimento 4 — Traducción técnica (tarea intermedia).

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Evalúa seis técnicas de prompting (Zero-shot, Few-shot, Chain-of-Thought,
Chain-of-Thought limitado, Structured Prompting JSON y Role Prompting) sobre
el modelo gpt-4o-mini, en una tarea de traducción técnica de textos en inglés
sobre LLMs al español. La calidad de cada respuesta se evalúa mediante un
esquema de doble juez (GPT-4o y Claude Haiku 4.5) en cuatro dimensiones: exactitud funcional, completitud funcional,
claridad comunicacional y profundidad conceptual.

El experimento se repite ``--num-runs`` veces (10 por defecto) para poder
calcular la variabilidad de los resultados entre corridas. Cada corrida se
guarda como un archivo JSON independiente: ``results4.0.json``,
``results4.1.json``, ..., ``results4.9.json``.

Uso:
    python experimento4_traduccion.py

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
TASK_DESCRIPTION = "traducción técnica del inglés al español"

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

# Escala permitida para exactitud_funcional en ESTA tarea. En tareas simples
# (una única respuesta correcta) esta constante sería ALLOWED_BINARY_VALUES;
# en traducción, donde la exactitud es graduable, es la misma escala Likert
# que el resto de las dimensiones.
EXACTITUD_ALLOWED_VALUES = ALLOWED_LIKERT_VALUES

# Tareas de entrada evaluadas: 3 párrafos técnicos en inglés sobre LLMs.
INPUTS = [
    """The self-attention mechanism allows each token in a sequence to weigh the relevance of every other token when computing its own representation, rather than relying solely on fixed-size context windows as earlier recurrent architectures did. In practice, this is implemented through three learned projections—queries, keys, and values—whose dot products, scaled by the square root of the key dimension, are passed through a softmax function to produce attention weights. Stacking multiple attention heads in parallel lets the model attend to different types of relationships simultaneously, such as syntactic dependencies in one head and long-range semantic associations in another. This design choice, combined with residual connections and layer normalization, is what allows transformer-based models to scale to billions of parameters without suffering from the vanishing gradient problems that plagued earlier deep recurrent networks. However, the quadratic cost of self-attention with respect to sequence length remains a significant bottleneck for processing very long documents.""",
    """Reinforcement learning from human feedback has become the dominant approach for aligning large language models with human preferences after pretraining. The process typically unfolds in three stages: first, human annotators rank multiple model outputs for the same prompt according to helpfulness, honesty, and harmlessness; second, these rankings are used to train a separate reward model that learns to predict which responses humans would prefer; and third, the original language model is fine-tuned using reinforcement learning, most commonly proximal policy optimization, to maximize the reward model's score. Critics of this pipeline point out that reward models can be gamed through a phenomenon known as reward hacking, where the policy learns to exploit quirks of the reward model rather than genuinely improving response quality. More recent alternatives, such as direct preference optimization, attempt to sidestep the need for a separate reward model entirely by reformulating the alignment objective as a classification problem over preference pairs, which simplifies the training pipeline considerably.""",
    """Empirical scaling laws suggest that a model's loss on held-out data decreases predictably as a power-law function of three factors: the number of parameters, the size of the training dataset, and the amount of compute used, measured in floating-point operations (FLOPs). A widely cited study found that for a fixed compute budget, the optimal allocation is roughly 20 training tokens per parameter, which implied that many earlier models in the 175B-to-540B parameter range were substantially undertrained relative to their size. This finding motivated a shift toward smaller, more heavily trained models that match or exceed the performance of larger predecessors while requiring far less compute at inference time. On standard benchmarks such as MMLU and HumanEval, this approach has closed much of the gap between models an order of magnitude apart in parameter count, though performance on tasks requiring genuine multi-step reasoning still correlates more strongly with scale than with training-token density alone."""
    ]

DEFAULT_NUM_RUNS = 10
DEFAULT_OUTPUT_PREFIX = "results4"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experimento4")


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
    return f"Traduce el siguiente texto técnico del inglés al español:\n\n{text}"


def few_shot(text: str) -> str:
    examples = (
        "Ejemplos de traducción:\n"
        "'machine learning' -> 'aprendizaje automático'\n"
        "'neural networks' -> 'redes neuronales'\n"
        "'cloud computing' -> 'computación en la nube'\n"
        "'energy consumption' -> 'consumo de energía'\n\n"
    )
    return f"{examples}Ahora traduce el siguiente texto técnico del inglés al español:\n\n{text}"


def cot(text: str) -> str:
    return (
        "Traduce el siguiente texto técnico del inglés al español.\n"
        "Explica tu estrategia paso a paso y luego proporciona la traducción completa:\n\n"
        f"{text}"
    )


def cot_limited(text: str) -> str:
    return (
        "Traduce el siguiente texto técnico del inglés al español.\n"
        "Explica tu estrategia y limitala a máximo 100 palabras; "
        f"luego proporciona la traducción completa:\n\n{text}"
    )


def structured_prompting_json(text: str) -> str:
    prompt = {
        "instruction": "Traduce el siguiente texto técnico del inglés al español.",
        "input": text,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text: str) -> str:
    return (
        "Actúa como un traductor técnico especializado en informática.\n"
        "Traduce el siguiente texto al español.\n\n"
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

EXACTITUD FUNCIONAL: ¿La traducción preserva correctamente el significado del texto original?
- 1.0: Totalmente correcta y precisa
- 0.7: Mayormente correcta con errores menores
- 0.5: Parcialmente correcta con algunos errores de significado
- 0.3: Varios errores relevantes
- 0: Incorrecta o distorsiona el significado

COMPLETITUD FUNCIONAL: ¿Responde completamente y de forma clara lo solicitado?
- 1.0: Responde completamente y deja clara la respuesta
- 0.7: Responde bien con omisiones menores
- 0.5: Respuesta parcial o algo incompleta
- 0.3: Respuesta débil o muy breve
- 0: No responde a la tarea

CLARIDAD COMUNICACIONAL: ¿Qué tan clara, estructurada y fácil de interpretar es?
- 1.0: Muy clara, bien organizada y fácil de seguir
- 0.7: Clara con estructura adecuada
- 0.5: Comprensible pero con problemas leves de claridad
- 0.3: Difícil de seguir o con organización pobre
- 0: Muy confusa o incoherente

PROFUNDIDAD CONCEPTUAL: ¿El nivel de detalle y razonamiento es apropiado para la tarea?
- 1.0: Razonamiento muy apropiado y bien contextualizado
- 0.7: Buen nivel de profundidad
- 0.5: Profundidad suficiente pero básica
- 0.3: Profundidad limitada o superficial
- 0: Sin profundidad útil

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

    Usa el texto de entrada completo (sin truncar): el juez necesita ver el
    original íntegro para poder evaluar correctamente exactitud y
    completitud de la traducción, no solo las salidas generadas.
    """
    responses_text = "\n\n".join(
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
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
        output_prefix: Prefijo del archivo (ej. "results4" -> "results4.0.json").

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