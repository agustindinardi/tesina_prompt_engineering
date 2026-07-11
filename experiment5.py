import os
import json
import re
import tiktoken
from openai import OpenAI
from anthropic import Anthropic

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY no está configurada")

claude_api_key = os.getenv("CLAUDE_API_KEY")
if not claude_api_key:
    raise ValueError("CLAUDE_API_KEY no está configurada")

openai_client = OpenAI(api_key=openai_api_key)
claude_client = Anthropic(api_key=claude_api_key)

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

ALLOWED_LIKERT_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]


def normalize_score(value, allowed_values):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return allowed_values[0]

    return min(allowed_values, key=lambda allowed: (abs(allowed - numeric_value), allowed))


def normalize_evaluation_scores(scores):
    return {
        "exactitud_funcional": normalize_score(scores.get("exactitud_funcional", 0), ALLOWED_LIKERT_VALUES),
        "completitud_funcional": normalize_score(scores.get("completitud_funcional", 0), ALLOWED_LIKERT_VALUES),
        "claridad_comunicacional": normalize_score(scores.get("claridad_comunicacional", 0), ALLOWED_LIKERT_VALUES),
        "profundidad_conceptual": normalize_score(scores.get("profundidad_conceptual", 0), ALLOWED_LIKERT_VALUES)
    }


def count_tokens(text):
    return len(enc.encode(text))


def get_default_scores():
    return {
        "exactitud_funcional": 0,
        "completitud_funcional": 0,
        "claridad_comunicacional": 0,
        "profundidad_conceptual": 0
    }


# Función para evaluación comparativa de respuestas con GPT-4o
def evaluate_responses_with_gpt4o(input_text, responses_dict):
    """
    Evalúa comparativamente todas las respuestas de una misma entrada.
    responses_dict: {technique_name: {"prompt": ..., "output": ...}}
    Retorna: {technique_name: {dimensiones}}
    Rankea las respuestas de mejor a peor
    """

    responses_text = "\n\n".join([
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    ])

    evaluation_prompt = f"""Realiza una evaluación comparativa de las siguientes respuestas para la tarea de resolución matemática: "{input_text[:100]}"

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para problemas matemáticos:

EXACTITUD FUNCIONAL: ¿Es correcto el resultado final?
- 1.0: Correcto y consistente
- 0.7: Correcto con errores menores en pasos intermedios
- 0.5: Parcialmente correcto
- 0.3: Contiene errores significativos
- 0: Incorrecto

COMPLETITUD FUNCIONAL: ¿Incluye pasos suficientes para reproducir la solución?
- 1.0: Completa, con pasos esenciales
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
      "exactitud_funcional": 0.7,
      "completitud_funcional": 1,
      "claridad_comunicacional": 0.7,
      "profundidad_conceptual": 0.5
    }},
    ...
  }}
}}

Solo devuelve el JSON, sin explicaciones adicionales."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0
    )

    results = {}
    try:
        response_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            evaluation_data = json.loads(json_match.group())
            evaluations = evaluation_data.get("evaluations", {})

            for tech, scores in evaluations.items():
                if tech in responses_dict:
                    results[tech] = normalize_evaluation_scores(scores)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  Error parsing GPT-4o evaluation: {e}")

    for tech in responses_dict:
        if tech not in results:
            results[tech] = {
                "exactitud_funcional": 0,
                "completitud_funcional": 0,
                "usabilidad_comprensibilidad": 0,
                "eficiencia_comunicacional": 0,
                "confiabilidad_semantica": 0,
                "adecuacion_contextual": 0
            }

    return results


# Función para evaluación comparativa de respuestas con Claude
def evaluate_responses_with_claude(input_text, responses_dict):
    """
    Evalúa comparativamente todas las respuestas usando Claude como juez.
    """
    responses_text = "\n\n".join([
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    ])

    evaluation_prompt = f"""Realiza una evaluación comparativa de las siguientes respuestas para la tarea de resolución matemática: "{input_text[:100]}"

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para problemas matemáticos:

EXACTITUD FUNCIONAL: ¿Es correcto el resultado final?
- 1.0: Correcto y consistente
- 0.7: Correcto con errores menores en pasos intermedios
- 0.5: Parcialmente correcto
- 0.3: Contiene errores significativos
- 0: Incorrecto

COMPLETITUD FUNCIONAL: ¿Incluye pasos suficientes para reproducir la solución?
- 1.0: Completa, con pasos esenciales
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
      "exactitud_funcional": 0.7,
      "completitud_funcional": 1,
      "claridad_comunicacional": 0.7,
      "profundidad_conceptual": 0.5
    }},
    ...
  }}
}}

Solo devuelve el JSON, sin explicaciones adicionales."""

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": evaluation_prompt}]
    )

    results = {}
    try:
        response_text = response.content[0].text.strip()
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            evaluation_data = json.loads(json_match.group())
            evaluations = evaluation_data.get("evaluations", {})

            for tech, scores in evaluations.items():
                if tech in responses_dict:
                    results[tech] = normalize_evaluation_scores(scores)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  Error parsing Claude evaluation: {e}")

    for tech in responses_dict:
        if tech not in results:
            results[tech] = {
                "exactitud_funcional": 0,
                "completitud_funcional": 0,
                "usabilidad_comprensibilidad": 0,
                "eficiencia_comunicacional": 0,
                "confiabilidad_semantica": 0,
                "adecuacion_contextual": 0
            }

    return results


# Función wrapper que combina evaluaciones de ambos jueces
def evaluate_responses_comparative(input_text, responses_dict):
    """
    Evalúa comparativamente usando tanto GPT-4o como Claude.
    Retorna las evaluaciones de ambos jueces y el promedio.
    """
    print("    - Evaluando con GPT-4o...")
    gpt_results = evaluate_responses_with_gpt4o(input_text, responses_dict)

    print("    - Evaluando con Claude...")
    claude_results = evaluate_responses_with_claude(input_text, responses_dict)

    combined_results = {}
    for tech in responses_dict:
        gpt_scores = gpt_results.get(tech, {})
        claude_scores = claude_results.get(tech, {})

        combined_results[tech] = {
            "gpt4o": gpt_scores,
            "claude": claude_scores,
            "promedio": {
                "exactitud_funcional": (gpt_scores.get("exactitud_funcional", 0) + claude_scores.get("exactitud_funcional", 0)) / 2,
                "completitud_funcional": (gpt_scores.get("completitud_funcional", 0) + claude_scores.get("completitud_funcional", 0)) / 2,
                "claridad_comunicacional": (gpt_scores.get("claridad_comunicacional", 0) + claude_scores.get("claridad_comunicacional", 0)) / 2,
                "profundidad_conceptual": (gpt_scores.get("profundidad_conceptual", 0) + claude_scores.get("profundidad_conceptual", 0)) / 2
            }
        }

    return combined_results


# Tareas: Solo Matemática
inputs = [
    "Un comercio aplica primero un 20% de descuento sobre un producto de $250. Luego agrega un impuesto del 21%. Si el cliente paga en efectivo recibe además un descuento adicional del 5% sobre el precio final. ¿Cuánto termina pagando?",
    "Una cooperativa recibe una donación de $36.000 que debe repartirse entre tres escuelas según la cantidad de estudiantes. La Escuela A tiene 180 alumnos, la Escuela B tiene 240 alumnos y la Escuela C tiene 300 alumnos. ¿Cuánto dinero recibe cada escuela?",
    "Una empresa reparte un bono de $48.000 entre tres empleados según las horas trabajadas durante el mes. Ana trabajó 120 horas, Bruno 150 horas y Carla 210 horas. ¿Cuánto recibe cada uno?"
]


# Técnicas
def zero_shot(text):
    return f"Resuelve el siguiente problema matemático:\n\n{text}"


def few_shot(text):
    examples = (
        "Ejemplos:\n"
        "Un artículo de $50 con 10% descuento: $50 * 0.9 = $45\n"
        "Luego con 5% impuesto: $45 * 1.05 = $47.25\n"
        "Resultado: $47.25\n\n"
    )
    return f"{examples}Resuelve el siguiente problema matemático:\n\n{text}"


def cot(text):
    return f"""Resuelve el siguiente problema matemático.
    Explica tu razonamiento paso a paso y luego da la respuesta final.
    {text}"""


def cot_limited(text):
    return f"""Resuelve el siguiente problema matemático.
    Explica tu razonamiento en un máximo de 100 palabras y luego da la respuesta final.
    {text}"""


def structured_prompting_json(text):
    prompt = {
        "instruction": "Resuelve el siguiente problema matemático.",
        "input": text
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text):
    return f"""Actúa como un profesor de matemática.
    Resuelve el siguiente problema matemático.
    {text}"""


techniques = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "cot": cot,
    "cot_limited": cot_limited,
    "structured_prompting_json": structured_prompting_json,
    "role_prompting": role_prompting
}

results = []
responses_by_input = {}

for name, func in techniques.items():
    for text in inputs:
        prompt = func(text)

        if openai_client is None:
            raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens

        print(f"[{name}] Input: {text[:50]}... | Request ID: {response.id} | In: {tokens_in} tokens | Out: {tokens_out} tokens")

        results.append({
            "input": text,
            "technique": name,
            "prompt": prompt,
            "output": output,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "total_tokens": tokens_in + tokens_out
        })

        if text not in responses_by_input:
            responses_by_input[text] = {
                "responses": {}
            }
        responses_by_input[text]["responses"][name] = {
            "prompt": prompt,
            "output": output
        }


print("\nEvaluando respuestas de forma comparativa...")

for input_text, input_data in responses_by_input.items():
    print(f"  Evaluando input: {input_text[:50]}...")

    evaluations = evaluate_responses_comparative(
        input_text,
        input_data["responses"]
    )

    for i, r in enumerate(results):
        if r["input"] == input_text:
            tech = r["technique"]
            if tech in evaluations:
                eval_data = evaluations[tech]

                gpt4o_scores = eval_data.get("gpt4o", {})
                r["gpt4o_evaluacion"] = {
                    "exactitud_funcional": gpt4o_scores.get("exactitud_funcional", 0),
                    "completitud_funcional": gpt4o_scores.get("completitud_funcional", 0),
                    "claridad_comunicacional": gpt4o_scores.get("claridad_comunicacional", 0),
                    "profundidad_conceptual": gpt4o_scores.get("profundidad_conceptual", 0)
                }

                claude_scores = eval_data.get("claude", {})
                r["claude_evaluacion"] = {
                    "exactitud_funcional": claude_scores.get("exactitud_funcional", 0),
                    "completitud_funcional": claude_scores.get("completitud_funcional", 0),
                    "claridad_comunicacional": claude_scores.get("claridad_comunicacional", 0),
                    "profundidad_conceptual": claude_scores.get("profundidad_conceptual", 0)
                }

                promedio_scores = eval_data.get("promedio", {})
                r["exactitud_funcional"] = promedio_scores.get("exactitud_funcional", 0)
                r["completitud_funcional"] = promedio_scores.get("completitud_funcional", 0)
                r["claridad_comunicacional"] = promedio_scores.get("claridad_comunicacional", 0)
                r["profundidad_conceptual"] = promedio_scores.get("profundidad_conceptual", 0)
                r["calidad"] = (
                    (promedio_scores.get("exactitud_funcional", 0) +
                     promedio_scores.get("completitud_funcional", 0) +
                     promedio_scores.get("claridad_comunicacional", 0) +
                     promedio_scores.get("profundidad_conceptual", 0)) / 4
                )


json_results = []
total_tokens_used = 0
total_tokens_input = 0
total_tokens_output = 0
techniques_stats = {}
technique_tokens = {}

for r in results:
    total_tokens_used += r["total_tokens"]
    total_tokens_input += r["tokens_input"]
    total_tokens_output += r["tokens_output"]

    tech = r["technique"]
    if tech not in technique_tokens:
        technique_tokens[tech] = {
            "total_input": 0,
            "total_output": 0,
            "count": 0
        }
    technique_tokens[tech]["total_input"] += r["tokens_input"]
    technique_tokens[tech]["total_output"] += r["tokens_output"]
    technique_tokens[tech]["count"] += 1

    if tech not in techniques_stats:
        techniques_stats[tech] = {
            "count": 0,
            "total_tokens": 0,
            "avg_calidad": 0,
            "total_calidad": 0,
            "total_calidad_gpt4o": 0,
            "total_calidad_claude": 0,
            "total_exactitud_funcional": 0,
            "total_completitud_funcional": 0,
            "total_claridad_comunicacional": 0,
            "total_profundidad_conceptual": 0
        }

    gpt4o_scores = r["gpt4o_evaluacion"]
    gpt4o_calidad = sum(gpt4o_scores.values()) / len(gpt4o_scores)

    claude_scores = r["claude_evaluacion"]
    claude_calidad = sum(claude_scores.values()) / len(claude_scores)

    techniques_stats[tech]["count"] += 1
    techniques_stats[tech]["total_tokens"] += r["total_tokens"]
    techniques_stats[tech]["total_calidad"] += r["calidad"]
    techniques_stats[tech]["total_calidad_gpt4o"] += gpt4o_calidad
    techniques_stats[tech]["total_calidad_claude"] += claude_calidad
    techniques_stats[tech]["total_exactitud_funcional"] += r["exactitud_funcional"]
    techniques_stats[tech]["total_completitud_funcional"] += r["completitud_funcional"]
    techniques_stats[tech]["total_claridad_comunicacional"] += r["claridad_comunicacional"]
    techniques_stats[tech]["total_profundidad_conceptual"] += r["profundidad_conceptual"]

    json_results.append({
        "input": r["input"],
        "technique": r["technique"],
        "output": r["output"],
        "tokens_input": int(r["tokens_input"]),
        "tokens_output": int(r["tokens_output"]),
        "total_tokens": int(r["total_tokens"]),
        "evaluacion_gpt4o": {
            "exactitud_funcional": round(r["gpt4o_evaluacion"]["exactitud_funcional"], 2),
            "completitud_funcional": round(r["gpt4o_evaluacion"]["completitud_funcional"], 2),
            "claridad_comunicacional": round(r["gpt4o_evaluacion"]["claridad_comunicacional"], 2),
            "profundidad_conceptual": round(r["gpt4o_evaluacion"]["profundidad_conceptual"], 2)
        },
        "evaluacion_claude": {
            "exactitud_funcional": round(r["claude_evaluacion"]["exactitud_funcional"], 2),
            "completitud_funcional": round(r["claude_evaluacion"]["completitud_funcional"], 2),
            "claridad_comunicacional": round(r["claude_evaluacion"]["claridad_comunicacional"], 2),
            "profundidad_conceptual": round(r["claude_evaluacion"]["profundidad_conceptual"], 2)
        },
        "promedio": {
            "exactitud_funcional": round(r["exactitud_funcional"], 2),
            "completitud_funcional": round(r["completitud_funcional"], 2),
            "claridad_comunicacional": round(r["claridad_comunicacional"], 2),
            "profundidad_conceptual": round(r["profundidad_conceptual"], 2),
            "calidad": round(r["calidad"], 2)
        }
    })


for tech in techniques_stats:
    count = techniques_stats[tech]["count"]
    techniques_stats[tech]["avg_calidad"] = round(
        techniques_stats[tech]["total_calidad"] / count, 2
    )
    techniques_stats[tech]["avg_calidad_gpt4o"] = round(
        techniques_stats[tech]["total_calidad_gpt4o"] / count, 2
    )
    techniques_stats[tech]["avg_calidad_claude"] = round(
        techniques_stats[tech]["total_calidad_claude"] / count, 2
    )
    techniques_stats[tech]["avg_exactitud_funcional"] = round(
        techniques_stats[tech]["total_exactitud_funcional"] / count, 2
    )
    techniques_stats[tech]["avg_completitud_funcional"] = round(
        techniques_stats[tech]["total_completitud_funcional"] / count, 2
    )
    techniques_stats[tech]["avg_claridad_comunicacional"] = round(
        techniques_stats[tech]["total_claridad_comunicacional"] / count, 2
    )
    techniques_stats[tech]["avg_profundidad_conceptual"] = round(
        techniques_stats[tech]["total_profundidad_conceptual"] / count, 2
    )


efficiency_by_technique = {}
efficiency_gpt4o_by_technique = {}
efficiency_claude_by_technique = {}

for tech in techniques_stats:
    tokens = techniques_stats[tech]["total_tokens"]
    quality_promedio = techniques_stats[tech]["avg_calidad"]
    quality_gpt4o = techniques_stats[tech]["avg_calidad_gpt4o"]
    quality_claude = techniques_stats[tech]["avg_calidad_claude"]

    efficiency = quality_promedio / tokens if tokens > 0 else 0
    efficiency_gpt4o = quality_gpt4o / tokens if tokens > 0 else 0
    efficiency_claude = quality_claude / tokens if tokens > 0 else 0

    efficiency_by_technique[tech] = round(efficiency, 6)
    efficiency_gpt4o_by_technique[tech] = round(efficiency_gpt4o, 6)
    efficiency_claude_by_technique[tech] = round(efficiency_claude, 6)

final_payload = {
    "results": json_results,
    "summary": {
        "total_results": len(json_results),
        "total_tokens_used": int(total_tokens_used),
        "total_tokens_input": int(total_tokens_input),
        "total_tokens_output": int(total_tokens_output),
        "techniques_stats": {
            tech: {
                "count": techniques_stats[tech]["count"],
                "tokens_input": int(technique_tokens[tech]["total_input"]),
                "tokens_output": int(technique_tokens[tech]["total_output"]),
                "total_tokens": int(techniques_stats[tech]["total_tokens"]),
                "avg_calidad_promedio": round(techniques_stats[tech]["avg_calidad"], 2),
                "avg_calidad_gpt4o": round(techniques_stats[tech]["avg_calidad_gpt4o"], 2),
                "avg_calidad_claude": round(techniques_stats[tech]["avg_calidad_claude"], 2),
                "dimensions": {
                    "exactitud_funcional": techniques_stats[tech]["avg_exactitud_funcional"],
                    "completitud_funcional": techniques_stats[tech]["avg_completitud_funcional"],
                    "claridad_comunicacional": techniques_stats[tech]["avg_claridad_comunicacional"],
                    "profundidad_conceptual": techniques_stats[tech]["avg_profundidad_conceptual"]
                },
                "efficiency_scores": {
                    "gpt4o": efficiency_gpt4o_by_technique[tech],
                    "claude": efficiency_claude_by_technique[tech],
                    "promedio": efficiency_by_technique[tech]
                }
            }
            for tech in techniques_stats
        }
    }
}

with open("results5.json", "w", encoding="utf-8") as f:
    json.dump(final_payload, f, ensure_ascii=False, indent=2)

print("\n✓ Resultados guardados en results5.json")
print("\nResumen de Evaluación Comparativa (Resolución Matemática):")
print(f"  Total de tokens utilizados (gpt-4o-mini): {total_tokens_used}")
print(f"  Total tokens input (gpt-4o-mini): {total_tokens_input}")
print(f"  Total tokens output (gpt-4o-mini): {total_tokens_output}")
print("  Nota: GPT-4o y Claude son usados solo como jueces (tokens de evaluación no incluidos)")
print("\nRanking de técnicas por calidad promedio:")

ranking = sorted(techniques_stats.items(), key=lambda x: x[1]["avg_calidad"], reverse=True)
for position, (tech, stats) in enumerate(ranking, 1):
    print(f"\n{position}. {tech.upper()}")
    print(f"    - Consultas: {stats['count']}")
    print(f"    - Tokens input: {int(technique_tokens[tech]['total_input'])}")
    print(f"    - Tokens output: {int(technique_tokens[tech]['total_output'])}")
    print(f"    - Tokens totales: {int(stats['total_tokens'])}")
    print(f"    - Calidad GPT-4o: {stats['avg_calidad_gpt4o']}")
    print(f"    - Calidad Claude: {stats['avg_calidad_claude']}")
    print(f"    - Calidad Promedio: {stats['avg_calidad']}")
    print(f"    - Efficiency GPT-4o: {efficiency_gpt4o_by_technique[tech]}")
    print(f"    - Efficiency Claude: {efficiency_claude_by_technique[tech]}")
    print(f"    - Efficiency Promedio: {efficiency_by_technique[tech]}")
    print("    - Dimensiones de calidad (ISO 25010):")
    print(f"      • Exactitud Funcional: {stats['avg_exactitud_funcional']}")
    print(f"      • Completitud Funcional: {stats['avg_completitud_funcional']}")
    print(f"      • Claridad Comunicacional: {stats['avg_claridad_comunicacional']}")
    print(f"      • Profundidad Conceptual: {stats['avg_profundidad_conceptual']}")