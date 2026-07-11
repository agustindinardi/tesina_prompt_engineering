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

# Función para evaluación comparativa de respuestas con GPT-4o
def evaluate_responses_with_gpt4o(input_text, responses_dict):
    """
    Evalúa comparativamente todas las respuestas de una misma entrada.
    responses_dict: {technique_name: {"prompt": ..., "output": ...}}
    Retorna: {technique_name: {dimensiones}}
    Rankea las respuestas de mejor a peor
    """

    # Construir el contexto de comparación
    responses_text = "\n\n".join([
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    ])

    evaluation_prompt = f"""Realiza una evaluación comparativa de las siguientes respuestas para la tarea de traducción: "{input_text[:100]}"

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para traducción técnica:

EXACTITUD FUNCIONAL: ¿La traducción preserva correctamente el significado del original?
- 1.0: Totalmente correcta y precisa
- 0.7: Mayormente correcta con errores menores
- 0.5: Parcialmente correcta con algunos errores de significado
- 0.3: Varios errores relevantes
- 0: Incorrecta o distorsiona el significado

COMPLETITUD FUNCIONAL: ¿Traduce todo el contenido importante sin omisiones?
- 1.0: Completa, sin omisiones relevantes
- 0.7: Casi completa, omisiones menores
- 0.5: Incompleta en algunos puntos
- 0.3: Muy incompleta
- 0: No cumple con la traducción solicitada

CLARIDAD COMUNICACIONAL: ¿Qué tan clara y natural es la redacción en español?
- 1.0: Muy clara, fluida y bien estructurada
- 0.7: Clara con pequeños problemas de estilo
- 0.5: Comprensible pero con redacción mejorable
- 0.3: Difícil de leer o confusa
- 0: Incoherente

PROFUNDIDAD CONCEPTUAL: ¿Mantiene adecuadamente la terminología y matices técnicos?
- 1.0: Excelente manejo de terminología y contexto técnico
- 0.7: Buen manejo técnico con fallas menores
- 0.5: Nivel técnico aceptable pero superficial
- 0.3: Manejo técnico pobre
- 0: Pierde totalmente el contexto técnico

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

    evaluation_prompt = f"""Realiza una evaluación comparativa de las siguientes respuestas para la tarea de traducción: "{input_text[:100]}"

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones.

DIMENSIONES DE EVALUACIÓN (ISO 25010) para traducción técnica:

EXACTITUD FUNCIONAL: ¿La traducción preserva correctamente el significado del original?
- 1.0: Totalmente correcta y precisa
- 0.7: Mayormente correcta con errores menores
- 0.5: Parcialmente correcta con algunos errores de significado
- 0.3: Varios errores relevantes
- 0: Incorrecta o distorsiona el significado

COMPLETITUD FUNCIONAL: ¿Traduce todo el contenido importante sin omisiones?
- 1.0: Completa, sin omisiones relevantes
- 0.7: Casi completa, omisiones menores
- 0.5: Incompleta en algunos puntos
- 0.3: Muy incompleta
- 0: No cumple con la traducción solicitada

CLARIDAD COMUNICACIONAL: ¿Qué tan clara y natural es la redacción en español?
- 1.0: Muy clara, fluida y bien estructurada
- 0.7: Clara con pequeños problemas de estilo
- 0.5: Comprensible pero con redacción mejorable
- 0.3: Difícil de leer o confusa
- 0: Incoherente

PROFUNDIDAD CONCEPTUAL: ¿Mantiene adecuadamente la terminología y matices técnicos?
- 1.0: Excelente manejo de terminología y contexto técnico
- 0.7: Buen manejo técnico con fallas menores
- 0.5: Nivel técnico aceptable pero superficial
- 0.3: Manejo técnico pobre
- 0: Pierde totalmente el contexto técnico

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

    # Promediar los resultados de ambos jueces
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


# Tareas: Solo Traducción
inputs = [
    """While measuring traditional software impact in terms of
emissions is well-established [1], [7], assessing LLMs consumption is still challenging, as High-Performance Computing
(HPC) clusters are often required to run the training process,
which can last for weeks or even months. Therefore, measuring
the energy consumption in terms of carbon emissions is
particularly challenging in those environments due to several
factors, e.g., parallel jobs or the non-exclusive use of the
cluster.
Moreover, even well-maintained LLMs leaderboard benchmarks [19]–[21] do not report energy consumption, focusing
instead on accuracy metrics. Figure 1 shows the carbon
emissions of the GPT-3 model in different server regions for
three big IT players, i.e., Google, Amazon, and Microsoft. For
instance, some models emit carbon that is equivalent to the
average of five cars over their lifetimes [22], thus underlining
significant sustainability concerns, especially when considering the growing scope of LLM-based implementations and
their integration into everyday life. This highlights the need
to reduce the carbon footprint of LLMs and to examine the
details that contribute to the reported figures.
To address the environmental impact of software, a range of
energy monitoring tools [5], [6] has been recently developed to
measure the carbon emissions associated with code execution.
Among these, the CodeCarbon tool [16] is a widely adopted
Python library that estimates the energy consumption of code
executions. It can also calculate the carbon footprint by
measuring the electricity power consumption of the underlying
hardware architecture, i.e., GPU, CPU, and RAM. In addition,
it can estimate the carbon intensity of the region where
the computing is done. This study focuses on the energy
consumption related to GPU usage without considering the
carbon emission.""",
    """Nowadays, LLMs are posing a significant impact on
the AI community, and the advent of ChatGPT and GPT-4
leads to the rethinking of the possibilities of artificial general
intelligence (AGI). OpenAI has published a technical article
entitled “Planning for AGI and beyond”, which discusses
the short-term and long-term plans to approach AGI [40],
and a more recent paper has argued that GPT-4 might be
considered as an early version of an AGI system [41]. The
research areas of AI are being revolutionized by the rapid
progress of LLMs. In the field of NLP, LLMs can serve as a
general-purpose language task solver (to some extent), and
the research paradigm has been shifting towards the use
of LLMs. In the field of IR, traditional search engines are
challenged by the new information seeking way through AI
chatbots (i.e., ChatGPT), and New Bing3 presents an initial
attempt that enhances the search results based on LLMs. In
the field of CV, the researchers try to develop ChatGPT-like
vision-language models that can better serve multimodal
dialogues [42–45], and GPT-4 [46] has supported multimodal input by integrating the visual information. This new
wave of technology would potentially lead to a prosperous
ecosystem of real-world applications based on LLMs. For
instance, Microsoft 365 is being empowered by LLMs (i.e.,
Copilot) to automate the office work, and OpenAI supports
the use of plugins in ChatGPT for implementing special
functions""",
    """LLMs have revolutionized the field of artificial intelligence and have emerged as the de-facto tool for
many tasks. The current established technology of LLMs is to process input and generate output at
the token level. This is in sharp contrast to humans who operate at multiple levels of abstraction, well
beyond single words, to analyze information and to generate creative content. In this paper, we present
an attempt at an architecture which operates on an explicit higher-level semantic representation,
which we name a “concept”. Concepts are language- and modality-agnostic and represent a higher
level idea or action in a flow. Hence, we build a “Large Concept Model”. In this study, as
proof of feasibility, we assume that a concept corresponds to a sentence, and use an existing sentence
embedding space, SONAR, which supports up to 200 languages in both text and speech modalities.
The Large Concept Model is trained to perform autoregressive sentence prediction in an embedding
space. We explore multiple approaches, namely MSE regression, variants of diffusion-based generation,
and models operating in a quantized SONAR space. These explorations are performed using 1.6B
parameter models and training data in the order of 1.3T tokens. We then scale one architecture to a
model size of 7B parameters and training data of about 2.7T tokens. We perform an experimental
evaluation on several generative tasks, namely summarization and a new task of summary expansion.
Finally, we show that our model exhibits impressive zero-shot generalization performance to many
languages, outperforming existing LLMs of the same size. The training code of our models is freely
available."""
]


# Técnicas de prompting
def zero_shot(text):
    return f"Traduce el siguiente texto técnico del inglés al español:\n\n{text}"


def few_shot(text):
    examples = (
        "Ejemplos de traducción:\n"
        "'machine learning' -> 'aprendizaje automático'\n"
        "'neural networks' -> 'redes neuronales'\n"
        "'cloud computing' -> 'computación en la nube'\n"
        "'energy consumption' -> 'consumo de energía'\n\n"
    )
    return f"{examples}Ahora traduce el siguiente texto técnico del inglés al español:\n\n{text}"


def cot(text):
    return f"""Traduce el siguiente texto técnico del inglés al español.
    Explica tu estrategia paso a paso y luego proporciona la traducción completa:

    {text}"""


def cot_limited(text):
    return f"""Traduce el siguiente texto técnico del inglés al español.
    Explica tu estrategia limitándolo a máximo 100 palabras; luego proporciona la traducción:

    {text}"""


def structured_prompting_json(text):
    prompt = {
        "instruction": "Traduce el siguiente texto técnico del inglés al español.",
        "input": text
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text):
    return f"""Actúa como un traductor técnico especializado en informática.
    Traduce el siguiente texto al español.
    {text}
    """

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

        # Generar respuesta con el modelo principal
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens

        # Log request details
        print(f"[{name}] Input: {text[:50]}... | Request ID: {response.id} | In: {tokens_in} tokens | Out: {tokens_out} tokens")

        # Guardar resultado sin evaluar aún
        results.append({
            "input": text,
            "technique": name,
            "prompt": prompt,
            "output": output,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "total_tokens": tokens_in + tokens_out
        })

        # Agrupar por input
        if text not in responses_by_input:
            responses_by_input[text] = {
                "responses": {}
            }
        responses_by_input[text]["responses"][name] = {
            "prompt": prompt,
            "output": output
        }


# Ahora evaluar cada input de forma comparativa
print("\nEvaluando respuestas de forma comparativa...")

for input_text, input_data in responses_by_input.items():
    print(f"  Evaluando input: {input_text[:50]}...")

    # Evaluación comparativa
    evaluations = evaluate_responses_comparative(
        input_text,
        input_data["responses"]
    )

    # Agregar evaluaciones a los resultados
    for i, r in enumerate(results):
        if r["input"] == input_text:
            tech = r["technique"]
            if tech in evaluations:
                eval_data = evaluations[tech]

                # Guardar evaluaciones de GPT-4o
                gpt4o_scores = eval_data.get("gpt4o", {})
                r["gpt4o_evaluacion"] = {
                    "exactitud_funcional": gpt4o_scores.get("exactitud_funcional", 0),
                    "completitud_funcional": gpt4o_scores.get("completitud_funcional", 0),
                    "claridad_comunicacional": gpt4o_scores.get("claridad_comunicacional", 0),
                    "profundidad_conceptual": gpt4o_scores.get("profundidad_conceptual", 0)
                }

                # Guardar evaluaciones de Claude
                claude_scores = eval_data.get("claude", {})
                r["claude_evaluacion"] = {
                    "exactitud_funcional": claude_scores.get("exactitud_funcional", 0),
                    "completitud_funcional": claude_scores.get("completitud_funcional", 0),
                    "claridad_comunicacional": claude_scores.get("claridad_comunicacional", 0),
                    "profundidad_conceptual": claude_scores.get("profundidad_conceptual", 0)
                }

                # Guardar promedio
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


# Save results JSON with token metrics and quality evaluation
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

    # Contar tokens por técnica
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

    # Aggregate statistics by technique
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

    # Calcular calidad de GPT-4o y Claude por técnica
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


# Calculate average quality per technique
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


# Calculate efficiency metrics: E = C / T (quality / tokens)
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

with open("results4.json", "w", encoding="utf-8") as f:
    json.dump(final_payload, f, ensure_ascii=False, indent=2)

print("\n✓ Resultados guardados en results4.json")
print("\nResumen de Evaluación Comparativa (Traducciones Técnicas):")
print(f"  Total de tokens utilizados (gpt-4o-mini): {total_tokens_used}")
print(f"  Total tokens input (gpt-4o-mini): {total_tokens_input}")
print(f"  Total tokens output (gpt-4o-mini): {total_tokens_output}")
print("  Nota: GPT-4o y Claude son usados solo como jueces (tokens de evaluación no incluidos)")
print("\nRanking de técnicas por calidad promedio:")

# Crear ranking por calidad
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