"""Combina y promedia múltiples corridas de un mismo experimento.

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente" (Facultad de Informática, UNLP). Toma N archivos JSON de
resultados (una corrida cada uno, generados por los scripts de
experimento) y produce un único archivo agregado con:

  - el promedio de tokens, calidad y eficiencia por técnica de prompting
    (``techniques_stats``), y
  - la media y el desvío estándar de esas mismas métricas entre corridas
    (``variabilidad_entre_corridas``), usado para evaluar la
    reproducibilidad de los resultados (ver sección 3.8 de la tesina).

Uso:
    python promediar_resultados.py results1.0.json results1.1.json ... results1.9.json
    python promediar_resultados.py results1.*.json -o resultados_promediados1.json
    python promediar_resultados.py results1.*.json --incluir-detalle
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any

# Mismas 4 dimensiones de calidad usadas en los scripts de
# experimento, para que los nombres de clave coincidan en todo el repositorio.
QUALITY_DIMENSIONS = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("promediar_resultados")


# =============================================================================
# Carga y validación
# =============================================================================

def _natural_key(text: str) -> list[Any]:
    """Clave de orden natural: results1.2.json antes que results1.10.json."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def expandir_patrones(patrones: list[str]) -> list[Path]:
    """Expande patrones glob (ej. 'results1.*.json') a rutas concretas.

    Necesario porque, a diferencia de bash, PowerShell (Windows) no expande
    comodines antes de pasarlos al programa: el argumento le llega tal cual
    ("results1.*.json") y hay que resolverlo manualmente en Python para que
    el script funcione igual sin importar la terminal usada.
    """
    rutas: set[str] = set()
    for patron in patrones:
        coincidencias = glob.glob(patron)
        if coincidencias:
            rutas.update(coincidencias)
        else:
            # Sin coincidencias: se deja tal cual para que, si realmente no
            # existe, el error de "archivo no encontrado" señale ese nombre.
            rutas.add(patron)

    return sorted((Path(r) for r in rutas), key=lambda p: _natural_key(str(p)))


def cargar_corridas(paths: list[Path]) -> list[dict[str, Any]]:
    """Carga cada archivo JSON de corrida y valida que tenga el formato esperado."""
    corridas = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "summary" not in data or "techniques_stats" not in data.get("summary", {}):
            raise ValueError(
                f"El archivo {path} no tiene el formato esperado "
                "(falta la clave 'summary.techniques_stats')"
            )
        corridas.append(data)

    return corridas


def _tecnicas_presentes(corridas: list[dict[str, Any]]) -> list[str]:
    """Lista ordenada de todas las técnicas que aparecen en al menos una corrida."""
    return sorted({
        tech
        for corrida in corridas
        for tech in corrida["summary"]["techniques_stats"]
    })


# =============================================================================
# Estadística
# =============================================================================

def media(valores: list[float]) -> float:
    return round(statistics.mean(valores), 6) if valores else 0.0


def desvio(valores: list[float]) -> float:
    return round(statistics.stdev(valores), 6) if len(valores) > 1 else 0.0


def promediar_techniques_stats(corridas: list[dict[str, Any]]) -> dict[str, Any]:
    """Promedia, para cada técnica, las métricas de todas las corridas donde aparece."""
    stats: dict[str, Any] = {}

    for tech in _tecnicas_presentes(corridas):
        runs = [
            corrida["summary"]["techniques_stats"][tech]
            for corrida in corridas
            if tech in corrida["summary"]["techniques_stats"]
        ]

        dimensions = {
            dim: media([r["dimensions"][dim] for r in runs])
            for dim in QUALITY_DIMENSIONS
        }

        stats[tech] = {
            "count": round(media([r["count"] for r in runs]), 2),
            "tokens_input": round(media([r["tokens_input"] for r in runs]), 2),
            "tokens_output": round(media([r["tokens_output"] for r in runs]), 2),
            "total_tokens": round(media([r["total_tokens"] for r in runs]), 2),
            "tokens_promedio_por_consulta": round(
                media([r["total_tokens"] / r["count"] for r in runs]), 2
            ),
            "avg_calidad_promedio": media([r["avg_calidad_promedio"] for r in runs]),
            "avg_calidad_gpt4o": media([r["avg_calidad_gpt4o"] for r in runs]),
            "avg_calidad_claude": media([r["avg_calidad_claude"] for r in runs]),
            "dimensions": dimensions,
            "efficiency_scores": {
                "gpt4o": media([r["efficiency_scores"]["gpt4o"] for r in runs]),
                "claude": media([r["efficiency_scores"]["claude"] for r in runs]),
                "promedio": media([r["efficiency_scores"]["promedio"] for r in runs]),
            },
        }

    return stats


def calcular_variabilidad_entre_corridas(corridas: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula media y desvío estándar entre corridas, por técnica (reproducibilidad)."""
    variabilidad: dict[str, Any] = {}

    for tech in _tecnicas_presentes(corridas):
        total_tokens_runs: list[float] = []
        calidad_runs: list[float] = []
        calidad_gpt4o_runs: list[float] = []
        calidad_claude_runs: list[float] = []
        dim_runs: dict[str, list[float]] = {dim: [] for dim in QUALITY_DIMENSIONS}
        efficiency_runs: list[float] = []

        for corrida in corridas:
            s = corrida["summary"]["techniques_stats"].get(tech)
            if s is None:
                continue

            total_tokens_runs.append(s["total_tokens"])
            calidad_runs.append(s["avg_calidad_promedio"])
            calidad_gpt4o_runs.append(s["avg_calidad_gpt4o"])
            calidad_claude_runs.append(s["avg_calidad_claude"])
            for dim in QUALITY_DIMENSIONS:
                dim_runs[dim].append(s["dimensions"][dim])
            efficiency_runs.append(s["efficiency_scores"]["promedio"])

        variabilidad[tech] = {
            "n_corridas": len(total_tokens_runs),
            "total_tokens": {"media": media(total_tokens_runs), "desvio": desvio(total_tokens_runs)},
            "calidad_promedio": {"media": media(calidad_runs), "desvio": desvio(calidad_runs)},
            "calidad_gpt4o": {"media": media(calidad_gpt4o_runs), "desvio": desvio(calidad_gpt4o_runs)},
            "calidad_claude": {"media": media(calidad_claude_runs), "desvio": desvio(calidad_claude_runs)},
            "dimensions": {
                dim: {"media": media(vals), "desvio": desvio(vals)}
                for dim, vals in dim_runs.items()
            },
            "eficiencia": {"media": media(efficiency_runs), "desvio": desvio(efficiency_runs)},
        }

    return variabilidad


def combinar_resultados_individuales(corridas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatena las respuestas individuales de todas las corridas.

    Solo se llama si se pasa --incluir-detalle; por defecto se omite porque
    ese detalle ya está disponible en los archivos de origen.
    """
    combinados: list[dict[str, Any]] = []
    for corrida in corridas:
        combinados.extend(corrida["results"])
    return combinados


# =============================================================================
# Orquestación
# =============================================================================

def promediar_resultados(paths: list[Path], output_path: Path, incluir_detalle: bool) -> None:
    """Combina N corridas en un único archivo de resultados promediados.

    Args:
        paths: Rutas a los archivos JSON de cada corrida individual.
        output_path: Ruta del archivo agregado a generar.
        incluir_detalle: Si es True, agrega también la lista completa de
            respuestas individuales de cada corrida bajo la clave
            "resultados_individuales_combinados". Por defecto se omite: ese
            detalle ya está en los archivos de origen y puede hacer el
            archivo agregado muy pesado con muchas corridas.
    """
    corridas = cargar_corridas(paths)
    total_mediciones = sum(len(c["results"]) for c in corridas)

    payload: dict[str, Any] = {
        "n_corridas_combinadas": len(corridas),
        "archivos_fuente": [str(p) for p in paths],
        "total_mediciones": total_mediciones,
        "techniques_stats": promediar_techniques_stats(corridas),
        "variabilidad_entre_corridas": calcular_variabilidad_entre_corridas(corridas),
    }

    if incluir_detalle:
        payload["resultados_individuales_combinados"] = combinar_resultados_individuales(corridas)

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    logger.info("Archivo generado: %s (%.1f KB)", output_path, size_kb)
    logger.info("Corridas combinadas: %d | Mediciones totales: %d", len(corridas), total_mediciones)
    if not incluir_detalle:
        logger.info(
            "Detalle de respuestas individuales omitido (usar --incluir-detalle para incluirlo)."
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "archivos", nargs="+",
        help="Archivos JSON de corridas individuales a combinar (ej: results1.0.json ... results1.9.json)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("resultados_promediados.json"),
        help="Archivo de salida (default: resultados_promediados.json)",
    )
    parser.add_argument(
        "--incluir-detalle", action="store_true",
        help="Incluye el detalle completo de cada respuesta individual en el archivo de salida "
             "(por defecto se omite, para mantener el archivo liviano)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rutas = expandir_patrones(args.archivos)
    logger.info("Archivos a combinar (%d): %s", len(rutas), ", ".join(str(r) for r in rutas))
    promediar_resultados(rutas, args.output, args.incluir_detalle)


if __name__ == "__main__":
    main()