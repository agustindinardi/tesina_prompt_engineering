"""Experimento 9 — Detección y explicación de bugs de lógica en código (tarea compleja).

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Evalúa seis técnicas de prompting (Zero-shot, Few-shot, Chain-of-Thought,
Chain-of-Thought limitado, Structured Prompting JSON y Role Prompting)
sobre el modelo gpt-4o-mini, en una tarea de debugging: se le entrega al
modelo una función Python que "pasa los tests del equipo" pero falla en
producción, con dos bugs de lógica independientes plantados (de tipos
distintos entre sí: control de flujo, mutación durante iteración,
aliasing de referencias, o contaminación de estado entre llamadas vía
argumento default mutable). El modelo debe encontrar AMBOS bugs, explicar
con un caso concreto cuándo se dispara cada uno, y corregirlos. A
diferencia de scheduling o síntesis, acá la habilidad exigida es simular
mentalmente la ejecución del código paso a paso para encontrar el estado
exacto en el que se rompe, en vez de sostener una cadena de restricciones
o mantener fuentes separadas.

Cada input tiene una referencia que lista los 2 bugs esperados, cada uno con su
descripción, un caso concreto que lo dispara (con el resultado incorrecto
real) y el fix esperado. Esa referencia se inyecta ÚNICAMENTE en el
prompt del juez, nunca en el prompt que recibe el modelo generador.

exactitud_funcional es GRADUADA pero OBJETIVAMENTE COMPUTABLE en esta
tarea: se calcula como la proporción de bugs reales correctamente identificados 
y explicados (no alcanza con mencionar el síntoma sin explicar la causa y el caso que la dispara). 
Con 2 bugs por input, los valores alcanzables (0, 0.5, 1) caen exactamente
sobre la escala graduada permitida, sin necesidad de redondeo real.

La calidad de cada respuesta se evalúa mediante un esquema de doble juez
(GPT-4o y Claude Haiku 4.5) en cuatro dimensiones:
exactitud funcional, completitud funcional, claridad comunicacional y
profundidad conceptual.

El experimento se repite ``--num-runs`` veces (10 por defecto) para poder
calcular la variabilidad de los resultados entre corridas. Cada corrida se
guarda como un archivo JSON independiente: ``results9.0.json``,
``results9.1.json``, ..., ``results9.9.json`` (se mantiene el prefijo
``results9`` porque este script ocupa el mismo lugar que la versión
anterior en la estructura de la tesina, sección 5.3.3).

Uso:
    python experimento9.py

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

TASK_DESCRIPTION = "detección y explicación de bugs de lógica en código Python"

# Criterio de exactitud funcional: graduada, pero computable de forma
# objetiva (proporción de bugs reales correctamente identificados).
EXACTITUD_CRITERIO = (
    "¿Identifica correctamente cada uno de los bugs reales indicados en la "
    "referencia, explicando el caso concreto que lo dispara, sin limitarse "
    "a describir el síntoma y sin inventar bugs que no están en la "
    "referencia?"
)

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

EXACTITUD_ALLOWED_VALUES = ALLOWED_LIKERT_VALUES

# =============================================================================
# Código con bugs plantados (verificado ejecutando cada caso antes de
# incluirlo; ver referencia de cada input más abajo para el detalle)
# =============================================================================

CODIGO_RACHA = '''def racha_mas_larga(temperaturas, umbral, ultimo_resultado_por_umbral={}):
    """
    Devuelve la longitud de la racha mas larga de dias consecutivos
    con temperatura estrictamente mayor al umbral.

    Si ya se proceso una serie con el mismo umbral en esta corrida del
    batch y no hubo un resultado mayor, reutiliza el ultimo calculado
    para ese umbral como piso (para no reportar retrocesos espurios
    por ventanas de datos incompletas).

    Ejemplo: [20, 31, 32, 33, 25, 31], umbral=30 -> 3
    """
    if not temperaturas:
        return 0
    racha_maxima = 0
    racha_actual = 0
    for temp in temperaturas:
        if temp > umbral:
            racha_actual += 1
        else:
            if racha_actual > racha_maxima:
                racha_maxima = racha_actual
            racha_actual = 0
    if umbral in ultimo_resultado_por_umbral:
        racha_maxima = max(racha_maxima, ultimo_resultado_por_umbral[umbral])
    ultimo_resultado_por_umbral[umbral] = racha_maxima
    return racha_maxima
'''

CODIGO_COLA = '''def depurar_cola_de_envios(cola):
    """
    Recibe la cola de notificaciones pendientes (lista de dicts) y
    elimina las que estan vencidas, canceladas o rechazadas, dejando
    en la cola solo las notificaciones activas. Modifica la cola in
    place y devuelve la cantidad de elementos eliminados.
    """
    eliminados = 0
    for notificacion in cola:
        if notificacion["estado"] in ("vencida", "cancelada"):
            cola.remove(notificacion)
            eliminados += 1
    return eliminados
'''

CODIGO_FACTURACION = '''def calcular_facturacion_mensual(pedidos, umbral_descuento=100, cache={}):
    """
    Calcula el total facturado por cliente en el mes, aplicando un
    descuento del 10% a los pedidos que superen el umbral de importe.
    Devuelve un diccionario {cliente: total_facturado}.
    Usa un cache interno para no recalcular clientes ya procesados
    en la misma corrida del proceso batch.
    """
    resultado = {}
    for pedido in pedidos:
        cliente = pedido["cliente"]
        monto = pedido["monto"]
        if cliente in cache:
            resultado[cliente] = cache[cliente]
            continue
        total = resultado.get(cliente, 0)
        if monto > umbral_descuento:
            monto = monto * 0.9
        total += monto
        resultado[cliente] = total
        cache[cliente] = total
    return resultado
'''

# Tareas de entrada evaluadas: 3 funciones Python, cada una con 2 bugs de
# lógica independientes plantados (verificados ejecutando el código real).
# El campo "bugs" (lista de 2 descripciones completas: síntoma + caso que
# lo dispara + fix esperado) se usa solo para armar el checklist que se le
# inyecta al juez; el modelo generador solo ve "text".
INPUTS: list[dict[str, Any]] = [
    {
        "text": (
            "Esta función pasa todos los tests del equipo, pero en "
            "producción se detectaron dos problemas: a veces devuelve "
            "rachas más cortas que las reales (incluso 0 cuando hubo días "
            "calurosos), y además, al procesar datos de una nueva "
            "estación, a veces devuelve una racha que en realidad "
            "corresponde a otra estación procesada antes en la misma "
            "corrida del proceso batch. Encontrá ambos problemas, explicá "
            "con un ejemplo concreto cuándo se dispara cada uno, y "
            "corregilos.\n\n"
            f"```python\n{CODIGO_RACHA}```"
        ),
        "bugs": [
            (
                "BUG DE CONTROL DE FLUJO: el máximo (racha_maxima) solo se "
                "actualiza en la rama 'else' del bucle, cuando se corta la "
                "racha actual. Si la racha más larga llega hasta el final "
                "de la lista sin cortarse, nunca se compara ni se guarda. "
                "Caso verificado: racha_mas_larga([20,25,31,32,33,34], 30, "
                "ultimo_resultado_por_umbral={}) devuelve 0 en vez de 4 "
                "(la racha de 4 días >30 ocupa el final de la lista). Fix "
                "esperado: comparar racha_actual contra racha_maxima "
                "también después de terminar el bucle (o llevar el máximo "
                "con max() en cada iteración)."
            ),
            (
                "BUG DE CONTAMINACIÓN ENTRE LLAMADAS (default mutable): "
                "'ultimo_resultado_por_umbral={}' es un argumento con "
                "valor por defecto mutable, que se comparte entre TODAS "
                "las llamadas a la función que no pasen su propio "
                "diccionario. Como el resultado de cada llamada se guarda "
                "ahí y se usa como piso en la siguiente, una estación con "
                "racha real baja puede heredar el resultado alto de una "
                "estación completamente distinta procesada antes con el "
                "mismo umbral. Caso verificado: racha_mas_larga("
                "[31,32,33,34,35,20], 30) (racha real y correctamente "
                "cerrada = 5) seguida de racha_mas_larga([20,31,20,20], "
                "30) (racha real = 1) hace que la segunda llamada devuelva "
                "5 en vez de 1. Fix esperado: usar "
                "'ultimo_resultado_por_umbral=None' como default y crear "
                "el diccionario dentro de la función si es None (o "
                "eliminar directamente el mecanismo de piso entre "
                "llamadas, ya que no es necesario para calcular la racha "
                "de una sola serie)."
            ),
        ],
    },
    {
        "text": (
            "Esta función pasó QA, pero en producción aparecieron dos "
            "problemas: algunas notificaciones vencidas o canceladas "
            "siguen enviándose cuando hay varias seguidas en la cola, y "
            "además las notificaciones con estado 'rechazada' nunca se "
            "eliminan pese a que el equipo espera que también se saquen "
            "de la cola. Encontrá ambos problemas, explicá con qué datos "
            "falla cada uno y por qué QA no los detectó, y corregilos.\n\n"
            f"```python\n{CODIGO_COLA}```"
        ),
        "bugs": [
            (
                "BUG DE MUTACIÓN DURANTE LA ITERACIÓN: al hacer "
                "'cola.remove(notificacion)' mientras se itera sobre "
                "'cola' con un for, la lista se reindexa en el momento y "
                "el iterador saltea el elemento siguiente al eliminado. "
                "Solo falla cuando hay dos o más notificaciones "
                "eliminables consecutivas. Caso verificado: cola = "
                "[{'estado':'activa'}, {'estado':'vencida'}, "
                "{'estado':'cancelada'}, {'estado':'activa'}] elimina 1 "
                "de las 2 eliminables (la 'cancelada' queda colgada en la "
                "cola) en vez de eliminar ambas. QA no lo detecta porque "
                "sus datos de prueba no tenían eliminables consecutivas. "
                "Fix esperado: iterar sobre una copia de la lista "
                "('for notificacion in list(cola)'), construir una lista "
                "nueva con las activas, o recorrer la lista al revés."
            ),
            (
                "BUG DE CASO FALTANTE: la condición solo chequea "
                "\"estado in ('vencida', 'cancelada')\", sin incluir "
                "'rechazada' pese a que el docstring de la función dice "
                "explícitamente que también debe eliminar las rechazadas. "
                "Caso verificado: cola = [{'estado':'activa'}, "
                "{'estado':'rechazada'}] elimina 0 en vez de 1 (la "
                "rechazada nunca se saca de la cola). QA no lo detecta si "
                "sus casos de prueba no incluyen notificaciones "
                "rechazadas. Fix esperado: agregar 'rechazada' a la tupla "
                "de estados a eliminar."
            ),
        ],
    },
    {
        "text": (
            "Este módulo pasa los tests de creación y de cálculo básico, "
            "pero en producción aparecieron dos problemas distintos: (1) "
            "algunos clientes con varios pedidos en el mismo mes terminan "
            "facturados de menos, y (2) al correr el proceso de un mes "
            "nuevo, algunos clientes arrastran el monto facturado del mes "
            "anterior. Encontrá ambos problemas, explicá con un caso "
            "concreto cuándo se dispara cada uno, y corregilos.\n\n"
            f"```python\n{CODIGO_FACTURACION}```"
        ),
        "bugs": [
            (
                "BUG DE LÓGICA DE CACHÉ DENTRO DE UNA MISMA CORRIDA: "
                "dentro de una sola llamada, si un cliente aparece en más "
                "de un pedido, el primero lo carga al 'cache' y los "
                "pedidos siguientes de ese mismo cliente se saltean por "
                "completo (rama 'if cliente in cache: continue') en vez "
                "de sumarse al total. Caso verificado: pedidos = "
                "[{'cliente':'ana','monto':50}, "
                "{'cliente':'ana','monto':60}] con cache={} da "
                "{'ana': 50} en vez de {'ana': 110}. Fix esperado: "
                "eliminar el corte temprano por caché dentro del bucle (no "
                "cumple ningún propósito real en una sola pasada) y "
                "acumular siempre 'total' para cada pedido del cliente."
            ),
            (
                "BUG DE CONTAMINACIÓN ENTRE LLAMADAS (default mutable): "
                "'cache={}' es un argumento con valor por defecto mutable "
                "que persiste entre llamadas sucesivas de la función si no "
                "se pasa explícitamente. Caso verificado: una llamada de "
                "'enero' con pedidos=[{'cliente':'pedro','monto':200}] da "
                "{'pedro': 180.0} (200*0.9); una llamada posterior de "
                "'febrero' con pedidos=[{'cliente':'pedro','monto':300}] "
                "debería dar {'pedro': 270.0} (300*0.9), pero devuelve "
                "{'pedro': 180.0}, arrastrado de la corrida de enero. Fix "
                "esperado: usar 'cache=None' como default y crear el "
                "diccionario dentro de la función si es None."
            ),
        ],
    },
]

DEFAULT_NUM_RUNS = 10
DEFAULT_OUTPUT_PREFIX = "results9"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experimento9")


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
# Mapeo input -> bugs (para armar el checklist del juez)
# =============================================================================

INPUT_BUGS_BY_TEXT: dict[str, list[str]] = {
    item["text"]: item["bugs"] for item in INPUTS
}


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


def normalize_evaluation_scores(scores: dict[str, Any], n_bugs: int) -> dict[str, float]:
    """Normaliza las 4 dimensiones de un puntaje crudo del juez a la escala permitida.

    exactitud_funcional NO se toma de un número que el juez haya escrito
    directamente: se calcula acá como (bugs marcados true en
    "bugs_detectados" / n_bugs). Esto evita el problema detectado en las
    primeras corridas de prueba, donde GPT-4o a veces asignaba valores
    como 0.7 sin que correspondieran a ninguna fracción real de bugs
    encontrados sobre 2 (probablemente por la instrucción genérica de
    "usa estos 5 valores" pisando la instrucción específica de calcular
    bugs/n_bugs). Al pedir un booleano por bug y calcular la fracción acá,
    se elimina esa fuente de error sin depender de que el juez "haga bien
    la cuenta" por su cuenta.
    """
    bugs_detectados = scores.get("bugs_detectados", [])
    if isinstance(bugs_detectados, list) and n_bugs:
        encontrados = sum(1 for marcado in bugs_detectados[:n_bugs] if marcado)
        exactitud = encontrados / n_bugs
    else:
        exactitud = 0.0

    return {
        "exactitud_funcional": _closest_allowed(exactitud, EXACTITUD_ALLOWED_VALUES),
        "completitud_funcional": _closest_allowed(scores.get("completitud_funcional", 0), ALLOWED_LIKERT_VALUES),
        "claridad_comunicacional": _closest_allowed(scores.get("claridad_comunicacional", 0), ALLOWED_LIKERT_VALUES),
        "profundidad_conceptual": _closest_allowed(scores.get("profundidad_conceptual", 0), ALLOWED_LIKERT_VALUES),
    }


def _empty_scores() -> dict[str, float]:
    return {dim: 0.0 for dim in QUALITY_DIMENSIONS}


# =============================================================================
# Técnicas de prompting
# =============================================================================
# Nota: "text" ya es la consigna completa (incluye el código con los bugs
# y la instrucción de qué hacer), no un dato en bruto al que hay que
# agregarle instrucción. Por eso zero_shot no le agrega nada.

def zero_shot(text: str) -> str:
    return text


def few_shot(text: str) -> str:
    # NOTA METODOLÓGICA: el ejemplo muestra un solo bug trivial (division
    # por cero), muy por debajo de los 2 bugs de lógica más sutiles que
    # exige la tarea real. Es intencional: mostrar un ejemplo con la
    # estructura completa de "2 bugs independientes en la misma función"
    # sería casi regalar la estrategia de resolución.
    example = (
        "Ejemplo de tarea de detección de bugs:\n"
        "Código:\n"
        "```python\n"
        "def promedio(numeros):\n"
        "    return sum(numeros) / len(numeros)\n"
        "```\n"
        "Bug: si 'numeros' es una lista vacía, 'len(numeros)' es 0 y la "
        "división lanza ZeroDivisionError. Caso que lo dispara: "
        "promedio([]).\n"
        "Fix: agregar 'if not numeros: return 0' antes de dividir.\n\n"
    )
    return f"{example}Ahora resolvé la siguiente consigna, encontrando TODOS los bugs indicados:\n\n{text}"


def cot(text: str) -> str:
    return (
        f"{text}\n\n"
        "Antes de escribir tu respuesta final, explicá brevemente tu "
        "razonamiento. Luego entregá tu respuesta final."
    )


def cot_limited(text: str) -> str:
    return (
        f"{text}\n\n"
        "Antes de escribir tu respuesta final, explicá tu razonamiento y "
        "limitalo a máximo 100 palabras; luego entregá tu respuesta final."
    )


def structured_prompting_json(text: str) -> str:
    prompt = {
        "instruction": "Encontrá y corregí TODOS los bugs de lógica indicados en el input, explicando el caso concreto que dispara cada uno.",
        "input": text,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def role_prompting(text: str) -> str:
    return (
        "Actuá como una persona ingeniera de software senior "
        "especializada en debugging y revisión de código.\n\n"
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

BUGS A VERIFICAR EN ESTE INPUT (verificá cada uno explícitamente, uno por uno; cada descripción ya incluye el caso concreto que lo dispara y el fix esperado, verificados ejecutando el código real):
{bugs_checklist}

RESPUESTAS A COMPARAR:
{responses_text}

Tu tarea es:
1. RANKEAR las respuestas de MEJOR a PEOR
2. Para CADA respuesta, evaluarla en estas 4 dimensiones.

DIMENSIONES DE EVALUACIÓN para {task_description}:

EXACTITUD FUNCIONAL: {exactitud_criterio}
- Para CADA técnica, completá "bugs_detectados": una lista de {n_bugs} valores true/false, en el MISMO ORDEN que el checklist de arriba (posición 1 = primer bug listado, etc.).
- Marcá true en la posición de un bug solo si la respuesta lo identifica correctamente Y explica el caso concreto que lo dispara, tal como se describe en la referencia — no alcanza con mencionar el síntoma (por ejemplo "la función a veces falla") sin explicar la causa real.
- Si la respuesta describe un problema real pero distinto (de estilo, nombres, validaciones genéricas) y no identifica el bug lógico específico de la referencia, marcá false en esa posición.
- IMPORTANTE: NO calcules vos mismo un número para exactitud_funcional. Tu única tarea en esta dimensión es completar el array de booleanos; el valor numérico se deriva automáticamente a partir de "bugs_detectados" y no debe incluirse en tu respuesta.

COMPLETITUD FUNCIONAL: ¿La respuesta explica por qué los tests o QA no detectaron el problema, y entrega una corrección funcional para cada bug identificado?
- 1.0: Explica por qué no se detectó y da una corrección funcional completa para todos los bugs identificados
- 0.7: Buena explicación/corrección con algún detalle menor faltante
- 0.5: Corrección parcial o explicación superficial de por qué no se detectó
- 0.3: Corrección o explicación muy incompleta
- 0: No explica ni corrige

CLARIDAD COMUNICACIONAL: ¿Qué tan clara, ordenada y fácil de seguir es la explicación?
- 1.0: Muy clara, ordenada y fácil de seguir
- 0.7: Clara con detalles mejorables
- 0.5: Comprensible pero confusa por momentos
- 0.3: Difícil de seguir
- 0: Incoherente

PROFUNDIDAD CONCEPTUAL: ¿La explicación aborda el mecanismo subyacente del bug (por qué ocurre a nivel de código) en vez de solo describir el síntoma observable?
- 1.0: Explica el mecanismo subyacente con precisión para todos los bugs identificados
- 0.7: Buena comprensión del mecanismo con algún matiz menor
- 0.5: Comprensión básica; se apoya más en el síntoma que en la causa
- 0.3: Comprensión superficial
- 0: Sin comprensión útil del mecanismo

RESTRICCIONES DE PUNTUACIÓN (aplican a completitud_funcional, claridad_comunicacional y profundidad_conceptual — NO a exactitud_funcional, que se completa como booleanos en "bugs_detectados"):
- Usa únicamente estos valores: 0, 0.3, 0.5, 0.7, 1
- No inventes valores intermedios
- Si dudas entre dos valores, elige uno de esos dos permitidos

RECORDATORIO ANTES DE RESPONDER:
- Volvé a chequear, bug por bug, si la respuesta lo identificó correctamente antes de marcar "bugs_detectados"; no lo estimes "a ojo".
- No premies bugs inventados que no están en la referencia de arriba.
- No incluyas un campo "exactitud_funcional" numérico en tu respuesta: alcanza con "bugs_detectados".
- Tu JSON debe incluir una entrada en "evaluations" para CADA UNA de estas {n_techniques} técnicas, sin omitir ninguna: {technique_list}.

Responde en formato JSON con estructura:
{{
  "ranking": ["mejor_técnica", "segunda", "tercera", "peor_técnica"],
  "evaluations": {{
    "nombre_técnica": {{
            "bugs_detectados": [true, false],
            "completitud_funcional": 0.7,
            "claridad_comunicacional": 0.5,
            "profundidad_conceptual": 0.3
    }},
    ...
  }}
}}

Solo devuelve el JSON, sin explicaciones adicionales."""


def _build_judge_prompt(input_text: str, bugs: list[str], responses_dict: dict[str, dict]) -> str:
    """Arma el prompt de evaluación comparativa para un input dado.

    Incluye el checklist explícito de bugs (con caso que los dispara y fix
    esperado, ya verificados ejecutando el código real) para que el juez
    los verifique uno por uno en vez de estimar el cumplimiento global de
    forma subjetiva.
    """
    responses_text = "\n\n".join(
        f"--- TÉCNICA: {tech} ---\n{data['output']}"
        for tech, data in responses_dict.items()
    )
    n_bugs = len(bugs)
    checklist = "\n\n".join(f"{i}. {b}" for i, b in enumerate(bugs, start=1))

    return JUDGE_PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
        exactitud_criterio=EXACTITUD_CRITERIO,
        input_text=input_text,
        bugs_checklist=checklist,
        n_bugs=n_bugs,
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


def _parse_judge_response(response_text: str, responses_dict: dict[str, dict], n_bugs: int) -> tuple[dict[str, dict], list[str]]:
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
                    results[tech] = normalize_evaluation_scores(scores, n_bugs)
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
    n_bugs: int,
) -> dict[str, dict]:
    """Llama al juez y reintenta si omitió alguna técnica en su respuesta."""
    results: dict[str, dict] = {}
    missing: list[str] = list(responses_dict.keys())

    for attempt in range(1, MAX_JUDGE_RETRIES + 1):
        response_text = call_judge()
        results, missing = _parse_judge_response(response_text, responses_dict, n_bugs)
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


def evaluate_with_gpt4o(input_text: str, bugs: list[str], responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando GPT-4o como juez."""
    prompt = _build_judge_prompt(input_text, bugs, responses_dict)

    def _call() -> str:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL_OPENAI,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    return _evaluate_with_retries(_call, "GPT-4o", responses_dict, len(bugs))


def evaluate_with_claude(input_text: str, bugs: list[str], responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Evalúa comparativamente todas las respuestas de un input usando Claude Haiku 4.5 como juez."""
    prompt = _build_judge_prompt(input_text, bugs, responses_dict)

    def _call() -> str:
        response = claude_client.messages.create(
            model=JUDGE_MODEL_CLAUDE,
            max_tokens=20000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    return _evaluate_with_retries(_call, "Claude Haiku 4.5", responses_dict, len(bugs))


def evaluate_comparative(input_text: str, bugs: list[str], responses_dict: dict[str, dict]) -> dict[str, dict]:
    """Combina las evaluaciones de ambos jueces y calcula el promedio por dimensión."""
    logger.info("  Evaluando con GPT-4o...")
    gpt_results = evaluate_with_gpt4o(input_text, bugs, responses_dict)

    logger.info("  Evaluando con Claude Haiku 4.5...")
    claude_results = evaluate_with_claude(input_text, bugs, responses_dict)

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

    El modelo generador solo recibe item["text"] (el código con los bugs y
    la consigna); nunca ve la lista de bugs esperados que se le arma al
    juez.
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
        bugs = INPUT_BUGS_BY_TEXT[input_text]
        evaluations = evaluate_comparative(input_text, bugs, input_data["responses"])

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