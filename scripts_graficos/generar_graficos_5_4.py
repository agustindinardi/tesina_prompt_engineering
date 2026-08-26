"""Genera las figuras de la sección 5.5 (comparación global entre niveles)
a partir de los tres archivos JSON agregados por nivel de complejidad.

Uso:
    python generar_graficos_5_4.py

Requiere: matplotlib (pip install matplotlib --break-system-packages)

Genera tres imágenes en ./salida/:
    fig_5x_evolucion_calidad.png     -> calidad promedio por técnica, 3 niveles
    fig_evolucion_eficiencia.png     -> índice de eficiencia por técnica, 3 niveles (escala log)
    fig_5z_brechas_dimensiones.png   -> brecha (máx-mín) entre técnicas por dimensión y nivel

El script está separado en funciones reutilizables (cargar datos, armar
series, graficar) para poder adaptarlo fácilmente a otras figuras de la
tesina (por ejemplo, los gráficos de cada subtarea individual) si más
adelante hace falta.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# =============================================================================
# Configuración general
# =============================================================================

# Rutas de los tres JSON agregados por nivel (ajustar si cambian de nombre)
ARCHIVOS_NIVEL = {
    "Simple": "nivel_simple.json",
    "Intermedia": "nivel_intermedio.json",
    "Compleja": "nivel_complejo.json",
}

OUTPUT_DIR = Path("salida")

# Orden y etiquetas de las técnicas — copiados literalmente de
# generar_graficos.py para que el orden de leyenda y el color de cada
# técnica sean siempre los mismos en cualquier figura de la tesina.
TECNICAS = [
    "zero_shot",
    "structured_prompting_json",
    "role_prompting",
    "few_shot",
    "cot_limited",
    "cot",
]

ETIQUETAS_TECNICA = {
    "zero_shot": "Zero-shot",
    "structured_prompting_json": "Structured Prompting (JSON)",
    "role_prompting": "Role Prompting",
    "few_shot": "Few-shot",
    "cot_limited": "Chain-of-Thought limitado",
    "cot": "Chain-of-Thought",
}

# Paleta ColorBrewer "Dark2" — idéntica a TECHNIQUE_COLORS en
# generar_graficos.py. No cambiar acá sin cambiar también allá.
COLORES_TECNICA = {
    "zero_shot": "#1b9e77",
    "few_shot": "#d95f02",
    "cot": "#7570b3",
    "cot_limited": "#e7298a",
    "structured_prompting_json": "#66a61e",
    "role_prompting": "#e6ab02",
}

DIMENSIONES = [
    "exactitud_funcional",
    "completitud_funcional",
    "claridad_comunicacional",
    "profundidad_conceptual",
]

ETIQUETAS_DIMENSION = {
    "exactitud_funcional": "Exactitud\nfuncional",
    "completitud_funcional": "Completitud\nfuncional",
    "claridad_comunicacional": "Claridad\ncomunicacional",
    "profundidad_conceptual": "Profundidad\nconceptual",
}

# Colores por nivel (para el gráfico de brechas, donde el eje de color es
# el nivel de complejidad, no la técnica). Se usa una escala secuencial de
# azules —en vez de colores tomados de la paleta Dark2 de técnicas— para
# que no se confunda "color de nivel" con "color de técnica" en ningún
# gráfico de la tesina.
COLORES_NIVEL = {
    "Simple": "#c6dbef",
    "Intermedia": "#6baed6",
    "Compleja": "#2171b5",
}


def _comma_format(value: float, _pos: int = 0) -> str:
    """Formatea un número con coma como separador decimal (misma convención
    que _comma_format en generar_graficos.py)."""
    return f"{value:g}".replace(".", ",")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# =============================================================================
# Carga de datos
# =============================================================================

def cargar_niveles(archivos_nivel: dict[str, str]) -> dict[str, dict]:
    """Carga los JSON de cada nivel y devuelve {nivel: techniques_stats}."""
    niveles = {}
    for nivel_display, path in archivos_nivel.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        niveles[nivel_display] = data["techniques_stats"]
    return niveles


def serie_por_tecnica(niveles: dict[str, dict], campo: str, sub_campo: str | None = None) -> dict[str, list[float]]:
    """Arma, para cada técnica, la lista de valores a través de los niveles (en orden).

    campo: clave directa dentro de techniques_stats[tecnica] (ej. "avg_calidad_promedio")
    sub_campo: si el campo es un dict anidado (ej. efficiency_scores -> "promedio")
    """
    series = {tecnica: [] for tecnica in TECNICAS}
    for nivel_display in niveles:
        stats = niveles[nivel_display]
        for tecnica in TECNICAS:
            valor = stats[tecnica][campo]
            if sub_campo is not None:
                valor = valor[sub_campo]
            series[tecnica].append(valor)
    return series


# =============================================================================
# Gráfico 1: Evolución de la calidad promedio (Figura 5.X)
# =============================================================================

def graficar_evolucion_calidad(niveles: dict[str, dict], output_path: Path) -> None:
    etiquetas_niveles = list(niveles.keys())
    series = serie_por_tecnica(niveles, "avg_calidad_promedio")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tecnica in TECNICAS:
        ax.plot(
            etiquetas_niveles,
            series[tecnica],
            marker="o",
            linewidth=2,
            label=ETIQUETAS_TECNICA[tecnica],
            color=COLORES_TECNICA[tecnica],
        )

    ax.set_ylabel("Calidad promedio")
    ax.set_xlabel("Nivel de complejidad")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Evolución de la calidad promedio por técnica")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_comma_format))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =============================================================================
# Gráfico 2: Evolución del índice de eficiencia (escala log, complementario)
# =============================================================================

def graficar_evolucion_eficiencia(niveles: dict[str, dict], output_path: Path) -> None:
    etiquetas_niveles = list(niveles.keys())
    series = serie_por_tecnica(niveles, "efficiency_scores", "promedio")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tecnica in TECNICAS:
        ax.plot(
            etiquetas_niveles,
            series[tecnica],
            marker="o",
            linewidth=2,
            label=ETIQUETAS_TECNICA[tecnica],
            color=COLORES_TECNICA[tecnica],
        )

    ax.set_yscale("log")
    ax.set_ylabel("Índice de eficiencia (escala logarítmica)")
    ax.set_xlabel("Nivel de complejidad")
    ax.set_title("Evolución del índice de eficiencia por técnica")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.6f}".replace(".", ",")))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =============================================================================
# Gráfico 3: Brecha entre técnicas por dimensión y nivel (Figura 5.Z)
# =============================================================================

def calcular_brechas(niveles: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Para cada nivel y dimensión, calcula brecha = max(tecnicas) - min(tecnicas)."""
    brechas: dict[str, dict[str, float]] = {}
    for nivel_display, stats in niveles.items():
        brechas[nivel_display] = {}
        for dim in DIMENSIONES:
            valores = [stats[tecnica]["dimensions"][dim] for tecnica in TECNICAS]
            brechas[nivel_display][dim] = max(valores) - min(valores)
    return brechas


def graficar_brechas_dimensiones(niveles: dict[str, dict], output_path: Path) -> None:
    brechas = calcular_brechas(niveles)
    etiquetas_niveles = list(niveles.keys())

    n_dim = len(DIMENSIONES)
    n_niveles = len(etiquetas_niveles)
    ancho_barra = 0.8 / n_niveles
    x = range(n_dim)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, nivel_display in enumerate(etiquetas_niveles):
        valores = [brechas[nivel_display][dim] for dim in DIMENSIONES]
        posiciones = [xi + (i - (n_niveles - 1) / 2) * ancho_barra for xi in x]
        barras = ax.bar(
            posiciones,
            valores,
            width=ancho_barra,
            label=nivel_display,
            color=COLORES_NIVEL[nivel_display],
            edgecolor="white",
        )
        ax.bar_label(barras, labels=[f"{v:.3f}".replace(".", ",") for v in valores], fontsize=8, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels([ETIQUETAS_DIMENSION[dim] for dim in DIMENSIONES])
    ax.set_ylabel("Brecha entre técnicas (máx. − mín.)")
    ax.set_title("Brecha entre técnicas por dimensión y nivel de complejidad")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_comma_format))
    ax.legend(title="Nivel", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    niveles = cargar_niveles(ARCHIVOS_NIVEL)

    graficar_evolucion_calidad(niveles, OUTPUT_DIR / "fig_5x_evolucion_calidad.png")
    graficar_evolucion_eficiencia(niveles, OUTPUT_DIR / "fig_evolucion_eficiencia.png")
    graficar_brechas_dimensiones(niveles, OUTPUT_DIR / "fig_5z_brechas_dimensiones.png")

    print("Listo. Figuras guardadas en:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()