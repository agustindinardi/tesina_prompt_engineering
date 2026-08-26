"""Generación del set fijo de gráficos por tarea o por nivel de complejidad.

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Genera un set fijo de 5 gráficos a partir de cualquier JSON agregado que
tenga la clave ``techniques_stats`` (el mismo formato que producen
``promediar_resultados.py`` y ``promediar_tareas.py``) — sea el agregado de
una única tarea (10 corridas) o el agregado de un nivel completo de
complejidad (varias tareas promediadas entre sí):

1. Consumo de tokens (entrada/salida) por técnica.
2. Calidad promedio por técnica.
3. Eficiencia por juez (GPT-4o vs. Claude Haiku 4.5) por técnica.
4. Trade-off calidad vs. tokens, con curvas de iso-eficiencia.
5. Radar de las 4 dimensiones de calidad por técnica.

Todas las figuras usan la misma paleta de colores fija por técnica
(``TECHNIQUE_COLORS``), para que cada técnica se vea siempre con el mismo
color en cualquier gráfico de la tesina.

Uso:
    python generar_graficos.py promedio_simple.json --output-dir graficos/5.1_cierre --title-suffix "Tareas simples (5.1)"
    python generar_graficos.py resultados_promediados3.json --output-dir graficos/5.1.3 --title-suffix "Tarea de verificacion (5.1.3)" --highlight structured_prompting_json,zero_shot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generar_graficos")

# Orden canónico de técnicas
TECHNIQUE_ORDER = [
    "zero_shot",
    "structured_prompting_json",
    "role_prompting",
    "few_shot",
    "cot_limited",
    "cot",
]

TECHNIQUE_LABELS: dict[str, str] = {
    "zero_shot": "Zero-shot",
    "structured_prompting_json": "Structured Prompting (JSON)",
    "role_prompting": "Role Prompting",
    "few_shot": "Few-shot",
    "cot_limited": "Chain-of-Thought limitado",
    "cot": "Chain-of-Thought",
}

# Paleta de colores FIJA por técnica (ColorBrewer "Dark2", distinguible
# también con daltonismo). Debe coincidir siempre con la usada en
# `grafico_tradeoff_calidad_tokens.py` para mantener el mismo color por
# técnica en todas las figuras de la tesina.
TECHNIQUE_COLORS: dict[str, str] = {
    "zero_shot": "#1b9e77",
    "few_shot": "#d95f02",
    "cot": "#7570b3",
    "cot_limited": "#e7298a",
    "structured_prompting_json": "#66a61e",
    "role_prompting": "#e6ab02",
}
DEFAULT_COLOR = "#666666"

DIMENSIONS = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]
DIMENSION_LABELS = {
    "exactitud_funcional": "Exactitud\nfuncional",
    "completitud_funcional": "Completitud\nfuncional",
    "claridad_comunicacional": "Claridad\ncomunicacional",
    "profundidad_conceptual": "Profundidad\nconceptual",
}


def _comma_format(value: float, _pos: int) -> str:
    """Formatea un número con coma como separador decimal (convención de la tesina)."""
    return f"{value:g}".replace(".", ",")


def _label(tech: str) -> str:
    return TECHNIQUE_LABELS.get(tech, tech)


def _color(tech: str) -> str:
    return TECHNIQUE_COLORS.get(tech, DEFAULT_COLOR)


def load_techniques_stats(json_path: Path) -> dict[str, dict[str, Any]]:
    """Carga el JSON agregado y devuelve el diccionario `techniques_stats`."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stats = data.get("techniques_stats")
    if stats is None:
        raise ValueError(f"El archivo {json_path} no contiene la clave 'techniques_stats' esperada.")
    return stats


def _ordered_techniques(stats: dict[str, dict[str, Any]]) -> list[str]:
    ordered = [t for t in TECHNIQUE_ORDER if t in stats]
    ordered += [t for t in stats if t not in ordered]
    return ordered


# =============================================================================
# 1. Consumo de tokens (entrada/salida) por técnica
# =============================================================================

def grafico_tokens(stats: dict[str, dict[str, Any]], title: str, output_path: Path, dpi: int) -> None:
    techniques = _ordered_techniques(stats)
    labels = [_label(t) for t in techniques]
    tokens_in = [stats[t]["tokens_input"] for t in techniques]
    tokens_out = [stats[t]["tokens_output"] for t in techniques]
    colors = [_color(t) for t in techniques]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(techniques))

    ax.bar(x, tokens_in, color=colors, alpha=0.5, label="Tokens de entrada")
    ax.bar(x, tokens_out, bottom=tokens_in, color=colors, alpha=1.0, label="Tokens de salida")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Tokens promedio")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(FuncFormatter(_comma_format))
    ax.grid(True, axis="y", alpha=0.2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.5, label="Tokens de entrada"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=1.0, label="Tokens de salida"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado en %s", output_path)


# =============================================================================
# 2. Calidad promedio por técnica
# =============================================================================

def grafico_calidad(stats: dict[str, dict[str, Any]], title: str, output_path: Path, dpi: int) -> None:
    techniques = _ordered_techniques(stats)
    labels = [_label(t) for t in techniques]
    calidad = [stats[t]["avg_calidad_promedio"] for t in techniques]
    colors = [_color(t) for t in techniques]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(techniques))

    bars = ax.bar(x, calidad, color=colors)
    for bar, value in zip(bars, calidad):
        ax.annotate(
            _comma_format(value, 0),
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Calidad promedio")
    ax.set_title(title)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(_comma_format))
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado en %s", output_path)


# =============================================================================
# 3. Eficiencia por juez (GPT-4o vs. Claude Haiku 4.5) por técnica
# =============================================================================

def grafico_eficiencia_juez(stats: dict[str, dict[str, Any]], title: str, output_path: Path, dpi: int) -> None:
    techniques = _ordered_techniques(stats)
    labels = [_label(t) for t in techniques]
    eff_gpt4o = [stats[t]["efficiency_scores"]["gpt4o"] for t in techniques]
    eff_claude = [stats[t]["efficiency_scores"]["claude"] for t in techniques]
    colors = [_color(t) for t in techniques]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(techniques))
    width = 0.38

    ax.bar(x - width / 2, eff_gpt4o, width, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, eff_claude, width, color=colors, edgecolor="black", linewidth=0.5, hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Índice de eficiencia (E = calidad / tokens)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.6f}".replace(".", ",")))
    ax.grid(True, axis="y", alpha=0.2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="GPT-4o"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="///", label="Claude Haiku 4.5"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado en %s", output_path)


# =============================================================================
# 4. Trade-off calidad vs. tokens, con curvas de iso-eficiencia
# =============================================================================

def grafico_tradeoff(
    stats: dict[str, dict[str, Any]],
    highlight: tuple[str, str] | None,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    techniques = _ordered_techniques(stats)
    tokens = [stats[t]["total_tokens"] for t in techniques]
    calidad = [stats[t]["avg_calidad_promedio"] for t in techniques]

    fig, ax = plt.subplots(figsize=(9, 6))

    for tech, x, y in zip(techniques, tokens, calidad):
        ax.scatter(x, y, s=130, color=_color(tech), edgecolor="white", linewidth=0.9, zorder=3)

    max_tokens = max(tokens) * 1.05
    efficiencies = sorted({round(c / t, 6) for c, t in zip(calidad, tokens)})
    x_range = [0, max_tokens]
    for eff in efficiencies:
        y_range = [eff * x for x in x_range]
        ax.plot(x_range, y_range, linestyle="--", linewidth=0.8, color="gray", alpha=0.45, zorder=1)

    highlight_line = None
    if highlight is not None:
        tech_a, tech_b = highlight
        if tech_a in stats and tech_b in stats:
            xa, ya = stats[tech_a]["total_tokens"], stats[tech_a]["avg_calidad_promedio"]
            xb, yb = stats[tech_b]["total_tokens"], stats[tech_b]["avg_calidad_promedio"]
            ax.plot([xa, xb], [ya, yb], linestyle=":", linewidth=1.6, color="#c0392b", zorder=2)
            better = tech_a if (ya / xa) > (yb / xb) else tech_b
            ax.annotate(
                f"Mayor eficiencia:\n{_label(better)}",
                xy=((xa + xb) / 2, (ya + yb) / 2),
                xycoords="data",
                xytext=(0.03, 0.97),
                textcoords="axes fraction",
                fontsize=9, color="#c0392b", fontweight="bold",
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            )
            highlight_line = Line2D([0], [0], color="#c0392b", linestyle=":", linewidth=1.6, label="Comparación resaltada")
        else:
            logger.warning("No se encontraron ambas técnicas a resaltar (%s, %s) en los datos.", tech_a, tech_b)

    ax.set_xlabel("Tokens totales promedio")
    ax.set_ylabel("Calidad promedio")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(_comma_format))
    ax.yaxis.set_major_formatter(FuncFormatter(_comma_format))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    technique_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=_color(t), markeredgecolor="white", label=_label(t))
        for t in techniques
    ]
    iso_handle = Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, label="Iso-eficiencia (E = calidad / tokens)")
    handles = technique_handles + [iso_handle]
    if highlight_line is not None:
        handles.append(highlight_line)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title="Técnica", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado en %s", output_path)


# =============================================================================
# 5. Radar de dimensiones por técnica
# =============================================================================

def grafico_radar(stats: dict[str, dict[str, Any]], title: str, output_path: Path, dpi: int) -> None:
    techniques = _ordered_techniques(stats)
    n_dims = len(DIMENSIONS)
    angles = [n / n_dims * 2 * np.pi for n in range(n_dims)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in DIMENSIONS], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0,25", "0,50", "0,75", "1,00"], fontsize=7)
    ax.grid(True, alpha=0.3)

    for tech in techniques:
        values = [stats[tech]["dimensions"][d] for d in DIMENSIONS]
        values += values[:1]
        color = _color(tech)
        ax.plot(angles, values, color=color, linewidth=1.6, label=_label(tech))
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_title(title, pad=30)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=9, title="Técnica")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado en %s", output_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_json", type=Path,
        help="Ruta al JSON agregado (con clave 'techniques_stats'); tarea individual o nivel completo.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Carpeta donde se guardan las 5 imágenes generadas (default: carpeta actual).",
    )
    parser.add_argument(
        "--title-suffix", type=str, default="",
        help="Texto a agregar al título de cada gráfico (ej. 'Tarea de verificación (5.1.3)').",
    )
    parser.add_argument(
        "--highlight", type=str, default="none",
        help="Dos técnicas a resaltar en el gráfico de trade-off, separadas por coma. "
             "Usar 'none' (default) para no resaltar ninguna.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Resolución de las imágenes generadas (default: 150).",
    )
    parser.add_argument(
        "--diagnostico", action="store_true",
        help="Reservado para gráficos adicionales de variabilidad entre corridas y acuerdo entre "
             "jueces, sobre el JSON agregado de todo el experimento (aún no implementado).",
    )
    return parser.parse_args()


def _titled(base: str, suffix: str) -> str:
    return f"{base} — {suffix}" if suffix else base


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_techniques_stats(args.input_json)

    highlight: tuple[str, str] | None
    if args.highlight.strip().lower() == "none":
        highlight = None
    else:
        parts = [p.strip() for p in args.highlight.split(",")]
        if len(parts) != 2:
            raise ValueError("--highlight debe tener exactamente dos técnicas separadas por coma, o 'none'.")
        highlight = (parts[0], parts[1])

    grafico_tokens(
        stats, _titled("Consumo de tokens por técnica", args.title_suffix),
        args.output_dir / "grafico_tokens.png", args.dpi,
    )
    grafico_calidad(
        stats, _titled("Calidad promedio por técnica", args.title_suffix),
        args.output_dir / "grafico_calidad.png", args.dpi,
    )
    grafico_eficiencia_juez(
        stats, _titled("Eficiencia por juez (GPT-4o vs. Claude Haiku 4.5)", args.title_suffix),
        args.output_dir / "grafico_eficiencia_juez.png", args.dpi,
    )
    grafico_tradeoff(
        stats, highlight, _titled("Trade-off calidad vs. tokens", args.title_suffix),
        args.output_dir / "grafico_tradeoff.png", args.dpi,
    )
    grafico_radar(
        stats, _titled("Dimensiones de calidad", args.title_suffix),
        args.output_dir / "grafico_radar.png", args.dpi,
    )

    if args.diagnostico:
        logger.warning(
            "--diagnostico fue solicitado pero todavía no está implementado "
            "(reservado para el análisis final único sobre el JSON agregado de todo el experimento)."
        )

    logger.info("Los 5 gráficos se guardaron en %s", args.output_dir)


if __name__ == "__main__":
    main()