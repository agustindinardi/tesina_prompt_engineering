"""Promediar resultados agregados entre tareas — cierre de Nivel 2.

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

A diferencia de ``promediar_resultados.py`` (que promedia entre CORRIDAS de
una misma tarea), este script promedia entre TAREAS de un mismo nivel de
complejidad. Toma como entrada los JSON ya agregados de cada tarea (ej.
``resultados_promediados1.json``, ``resultados_promediados2.json``,
``resultados_promediados3.json`` para el nivel de tareas simples) y produce un
único JSON con la misma estructura de ``techniques_stats``, listo para
alimentar ``generar_graficos.py``.

La eficiencia se recalcula a partir de la calidad y los tokens ya
promediados entre tareas (E = calidad_promedio_entre_tareas /
tokens_promedio_entre_tareas), en vez de promediar directamente los
índices de eficiencia de cada tarea, para mantener la misma convención de
cálculo usada en los scripts de experimento (``_build_summary``).

Uso:
    python promediar_tareas.py resultados_promediados1.json resultados_promediados2.json resultados_promediados3.json --output promedio_simple.json --nivel-label "Tareas simples (5.1)"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("promediar_tareas")

# Las 4 dimensiones de calidad usadas en toda la tesina.
DIMENSIONS = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]


def load_techniques_stats(path: Path) -> dict[str, dict[str, Any]]:
    """Carga un JSON agregado de tarea y devuelve su diccionario `techniques_stats`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("techniques_stats")
    if stats is None:
        raise ValueError(f"El archivo {path} no contiene la clave 'techniques_stats' esperada.")
    return stats


def average_across_tasks(all_stats: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Promedia, técnica por técnica, los `techniques_stats` de varias tareas."""
    technique_names: set[str] = set()
    for stats in all_stats:
        technique_names.update(stats.keys())

    combined: dict[str, dict[str, Any]] = {}
    for tech in sorted(technique_names):
        entries = [stats[tech] for stats in all_stats if tech in stats]
        if len(entries) != len(all_stats):
            logger.warning(
                "La técnica '%s' no aparece en todas las tareas provistas (%d/%d); "
                "se promedia solo con las tareas donde está disponible.",
                tech, len(entries), len(all_stats),
            )

        n = len(entries)
        tokens_input = sum(e["tokens_input"] for e in entries) / n
        tokens_output = sum(e["tokens_output"] for e in entries) / n
        total_tokens = sum(e["total_tokens"] for e in entries) / n
        avg_calidad = sum(e["avg_calidad_promedio"] for e in entries) / n
        avg_calidad_gpt4o = sum(e["avg_calidad_gpt4o"] for e in entries) / n
        avg_calidad_claude = sum(e["avg_calidad_claude"] for e in entries) / n
        dimensions = {
            dim: sum(e["dimensions"][dim] for e in entries) / n
            for dim in DIMENSIONS
        }

        combined[tech] = {
            "n_tareas": n,
            "tokens_input": round(tokens_input, 2),
            "tokens_output": round(tokens_output, 2),
            "total_tokens": round(total_tokens, 2),
            "avg_calidad_promedio": round(avg_calidad, 3),
            "avg_calidad_gpt4o": round(avg_calidad_gpt4o, 3),
            "avg_calidad_claude": round(avg_calidad_claude, 3),
            "dimensions": {dim: round(val, 3) for dim, val in dimensions.items()},
            "efficiency_scores": {
                "gpt4o": round(avg_calidad_gpt4o / total_tokens, 6) if total_tokens else 0,
                "claude": round(avg_calidad_claude / total_tokens, 6) if total_tokens else 0,
                "promedio": round(avg_calidad / total_tokens, 6) if total_tokens else 0,
            },
        }

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_jsons", type=Path, nargs="+",
        help="Rutas a los JSON agregados de cada tarea (cada uno con clave 'techniques_stats').",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("nivel_agregado.json"),
        help="Ruta del JSON combinado a generar (default: 'nivel_agregado.json').",
    )
    parser.add_argument(
        "--nivel-label", type=str, default="",
        help="Etiqueta descriptiva del nivel combinado (ej. 'Tareas simples (5.1)'), guardada como metadato.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_stats = [load_techniques_stats(path) for path in args.input_jsons]
    combined = average_across_tasks(all_stats)

    payload = {
        "nivel_label": args.nivel_label,
        "n_tareas_combinadas": len(args.input_jsons),
        "archivos_fuente": [str(p) for p in args.input_jsons],
        "techniques_stats": combined,
    }

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Resultados combinados guardados en %s", args.output)

    ranking = sorted(combined.items(), key=lambda item: item[1]["avg_calidad_promedio"], reverse=True)
    for position, (tech, stats) in enumerate(ranking, start=1):
        logger.info(
            "  %d. %-28s calidad=%.3f  eficiencia=%.6f",
            position, tech, stats["avg_calidad_promedio"], stats["efficiency_scores"]["promedio"],
        )


if __name__ == "__main__":
    main()