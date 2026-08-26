# Ingeniería de Prompts Eficiente

Repositorio asociado a la tesis:

**"Ingeniería de Prompts Eficiente: Optimización del uso de tokens y su impacto en la sostenibilidad económica y energética de los Modelos de Lenguaje de Gran Escala (LLMs)"**

## Descripción

Este repositorio contiene los scripts, prompts, datos procesados y resultados utilizados durante la experimentación presentada en la tesis.

El contenido se encuentra organizado de acuerdo con las diferentes etapas del proceso experimental y de análisis de resultados.

## Estructura

* `experimentos/`: experimentos realizados para evaluar las diferentes técnicas de prompting.
* `resultados_tareas_complejas/`: resultados correspondientes a las tareas de complejidad alta.
* `resultados_tareas_intermedias/`: resultados correspondientes a las tareas de complejidad intermedia.
* `resultados_tareas_simples/`: resultados correspondientes a las tareas de complejidad baja.
* `scripts_graficos/`: scripts utilizados para la generación de gráficos y visualizaciones de los resultados.
* `scripts_promediar/`: scripts utilizados para procesar, promediar y analizar los resultados obtenidos durante los experimentos.

## Técnicas evaluadas

* Zero-shot
* Few-shot
* Chain-of-Thought (CoT)
* Chain-of-Thought limitado (CoT-Limited)
* Structured Prompting JSON
* Role Prompting

## Modelos utilizados

* GPT-4o-mini
* GPT-4o
* Claude Haiku 4.5

## Ejecución

Cada script incluye en su interior una explicación de los pasos necesarios para su ejecución.

Antes de ejecutar los scripts que utilizan modelos de lenguaje, es necesario configurar las variables de entorno correspondientes a las claves de las APIs utilizadas.

## Reproducibilidad

Los scripts y resultados incluidos en este repositorio tienen como objetivo complementar la información presentada en la tesis y facilitar la comprensión y reproducción de los experimentos y análisis realizados.

Los resultados presentados fueron obtenidos a partir de las ejecuciones documentadas en el desarrollo de la investigación.
