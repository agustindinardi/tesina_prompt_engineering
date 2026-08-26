"""Gráfico de trade-off calidad vs. tokens — anomalías puntuales de Nivel 1.

Este script forma parte de la tesina de grado "Ingeniería de Prompts
Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad
económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"
(Facultad de Informática, UNLP).

Genera un único gráfico de dispersión (calidad promedio vs. tokens totales
promedio) con curvas de iso-eficiencia (E = calidad / tokens), a partir de
un JSON agregado de resultados (el mismo formato que produce
``promediar_resultados.py``, con la clave ``techniques_stats``). Pensado
para documentar, dentro de una subtarea puntual de Nivel 1 (ej. 5.1.3), una
anomalía que rompe el patrón general observado en las demás subtareas —
como el cruce entre Structured Prompting (JSON) y Zero-shot en la tarea de
verificación, donde JSON supera a Zero-shot en eficiencia pese a tener
menor calidad.

Uso:
    python grafico_tradeoff_calidad_tokens.py results3_agregado.json
    python grafico_tradeoff_calidad_tokens.py results3_agregado.json \\
        --highlight structured_prompting_json,zero_shot \\
        --title "Trade-off calidad vs. tokens - Tarea de verificacion (5.1.3)" \\
        --output tradeoff_5_1_3.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grafico_tradeoff")

# Nombres de técnica legibles para las etiquetas del gráfico.
TECHNIQUE_LABELS: dict[str, str] = {
    "zero_shot": "Zero-shot",
    "structured_prompting_json": "Structured Prompting (JSON)",
    "role_prompting": "Role Prompting",
    "few_shot": "Few-shot",
    "cot_limited": "Chain-of-Thought limitado",
    "cot": "Chain-of-Thought",
}

# Paleta de colores FIJA por técnica (ColorBrewer "Dark2", pensada para ser
# distinguible también con daltonismo). Se define en un único lugar para
# reusar exactamente esta misma paleta en el resto de los gráficos de la
# tesina (generar_graficos.py y cualquier gráfico nuevo), de forma que cada
# técnica tenga siempre el mismo color en todas las figuras del documento.
TECHNIQUE_COLORS: dict[str, str] = {
    "zero_shot": "#1b9e77",                  # verde azulado
    "few_shot": "#d95f02",                   # naranja
    "cot": "#7570b3",                        # violeta
    "cot_limited": "#e7298a",                # magenta
    "structured_prompting_json": "#66a61e",  # verde oliva
    "role_prompting": "#e6ab02",             # mostaza
}
DEFAULT_COLOR = "#666666"  # fallback para técnicas no contempladas en el mapeo

DEFAULT_HIGHLIGHT = ("structured_prompting_json", "zero_shot")


def _comma_format(value: float, _pos: int) -> str:
    """Formatea un número con coma como separador decimal (convención de la tesina)."""
    return f"{value:g}".replace(".", ",")


def load_techniques_stats(json_path: Path) -> dict[str, dict[str, Any]]:
    """Carga el JSON agregado y devuelve el diccionario `techniques_stats`."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stats = data.get("techniques_stats")
    if stats is None:
        raise ValueError(
            f"El archivo {json_path} no contiene la clave 'techniques_stats' esperada."
        )
    return stats


def _technique_label(tech: str) -> str:
    return TECHNIQUE_LABELS.get(tech, tech)


def build_tradeoff_plot(
    stats: dict[str, dict[str, Any]],
    highlight: tuple[str, str] | None,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Arma el gráfico de dispersión calidad-vs-tokens con curvas de iso-eficiencia."""
    techniques = list(stats.keys())
    tokens = [stats[t]["total_tokens"] for t in techniques]
    calidad = [stats[t]["avg_calidad_promedio"] for t in techniques]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Un color fijo por técnica (ver TECHNIQUE_COLORS) en vez de texto pegado
    # a cada punto: con técnicas muy cercanas en tokens/calidad, las
    # etiquetas de texto se superponen y quedan ilegibles. El color +
    # leyenda identifica cada técnica sin ese problema.
    for tech, x, y in zip(techniques, tokens, calidad):
        ax.scatter(
            x, y, s=130,
            color=TECHNIQUE_COLORS.get(tech, DEFAULT_COLOR),
            edgecolor="white", linewidth=0.9, zorder=3,
        )

    # Curvas de iso-eficiencia (E = calidad / tokens -> calidad = E * tokens).
    # Se dejan sin etiqueta numérica individual (esos valores exactos ya
    # están en la tabla del texto); una única entrada en la leyenda explica
    # qué representan las líneas.
    max_tokens = max(tokens) * 1.05
    efficiencies = sorted({round(c / t, 6) for c, t in zip(calidad, tokens)})
    x_range = [0, max_tokens]
    for eff in efficiencies:
        y_range = [eff * x for x in x_range]
        ax.plot(x_range, y_range, linestyle="--", linewidth=0.8, color="gray", alpha=0.45, zorder=1)

    # Resaltar el cruce entre dos técnicas puntuales (ej. JSON vs Zero-shot).
    highlight_line = None
    if highlight is not None:
        tech_a, tech_b = highlight
        if tech_a in stats and tech_b in stats:
            xa, ya = stats[tech_a]["total_tokens"], stats[tech_a]["avg_calidad_promedio"]
            xb, yb = stats[tech_b]["total_tokens"], stats[tech_b]["avg_calidad_promedio"]
            ax.plot([xa, xb], [ya, yb], linestyle=":", linewidth=1.6, color="#c0392b", zorder=2)
            eff_a = ya / xa
            eff_b = yb / xb
            better = tech_a if eff_a > eff_b else tech_b
            # Anotación en una esquina despejada del gráfico (no encima de
            # los puntos), con una flecha que apunta al tramo resaltado.
            ax.annotate(
                f"Mayor eficiencia:\n{_technique_label(better)}",
                xy=((xa + xb) / 2, (ya + yb) / 2),
                xycoords="data",
                xytext=(0.03, 0.97),
                textcoords="axes fraction",
                fontsize=9,
                color="#c0392b",
                fontweight="bold",
                ha="left",
                va="top",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            )
            highlight_line = Line2D(
                [0], [0], color="#c0392b", linestyle=":", linewidth=1.6,
                label="Comparación resaltada",
            )
        else:
            logger.warning(
                "No se encontraron ambas técnicas a resaltar (%s, %s) en los datos.",
                tech_a, tech_b,
            )

    ax.set_xlabel("Tokens totales promedio")
    ax.set_ylabel("Calidad promedio")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(_comma_format))
    ax.yaxis.set_major_formatter(FuncFormatter(_comma_format))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    # Leyenda: un marcador de color por técnica + una línea de referencia
    # para las curvas de iso-eficiencia (y para la comparación resaltada,
    # si corresponde). Se ubica fuera del área de trazado para no tapar
    # ningún punto.
    technique_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=9,
            markerfacecolor=TECHNIQUE_COLORS.get(t, DEFAULT_COLOR),
            markeredgecolor="white", label=_technique_label(t),
        )
        for t in techniques
    ]
    iso_handle = Line2D(
        [0], [0], color="gray", linestyle="--", linewidth=0.8,
        label="Iso-eficiencia (E = calidad / tokens)",
    )
    handles = technique_handles + [iso_handle]
    if highlight_line is not None:
        handles.append(highlight_line)
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=9, title="Técnica", title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado en %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_json", type=Path,
        help="Ruta al JSON agregado de resultados (con clave 'techniques_stats').",
    )
    parser.add_argument(
        "--highlight", type=str, default=",".join(DEFAULT_HIGHLIGHT),
        help="Dos técnicas a resaltar separadas por coma "
             "(default: 'structured_prompting_json,zero_shot'). Usar 'none' para no resaltar ninguna.",
    )
    parser.add_argument(
        "--title", type=str, default="Trade-off calidad vs. tokens",
        help="Título del gráfico.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tradeoff_calidad_tokens.png"),
        help="Ruta del archivo de imagen a generar (default: 'tradeoff_calidad_tokens.png').",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Resolución de la imagen generada (default: 150).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = load_techniques_stats(args.input_json)

    highlight: tuple[str, str] | None
    if args.highlight.strip().lower() == "none":
        highlight = None
    else:
        parts = [p.strip() for p in args.highlight.split(",")]
        if len(parts) != 2:
            raise ValueError("--highlight debe tener exactamente dos técnicas separadas por coma, o 'none'.")
        highlight = (parts[0], parts[1])

    build_tradeoff_plot(stats, highlight, args.title, args.output, args.dpi)


if __name__ == "__main__":
    main()