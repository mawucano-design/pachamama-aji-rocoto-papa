# modules/ia_integration.py - Versión para Groq con prompts agroecológicos mejorados
# + función generar_frase_campesina para reporte visual campesino
import os
import time
import pandas as pd
from typing import Dict, Tuple, Optional
from groq import Groq
import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================== CLIENTE GROQ ==================
def _get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

def llamar_groq(prompt: str, system_prompt: str = None, temperature: float = 0.3, max_retries: int = 2) -> Optional[str]:
    """
    Llama a la API de Groq con reintentos. Retorna None si falla.
    """
    client = _get_groq_client()
    if client is None:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for intento in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                if intento < max_retries - 1:
                    wait = 2 ** intento
                    time.sleep(wait)
                else:
                    return None
            else:
                return None
    return None

llamar_deepseek = llamar_groq

# ================== FUNCIONES DE PREPARACIÓN DE DATOS ==================
def preparar_resumen_zonas(gdf_completo, cultivo: str, max_zonas: int = 3) -> Tuple[pd.DataFrame, Dict]:
    cols = ['id_zona', 'area_ha', 'fert_npk_actual', 'fert_ndvi', 'fert_ndre',
            'fert_materia_organica', 'fert_humedad_suelo', 'rec_N', 'rec_P', 'rec_K',
            'costo_costo_total', 'proy_rendimiento_sin_fert', 'proy_rendimiento_con_fert',
            'proy_incremento_esperado', 'textura_suelo', 'arena', 'limo', 'arcilla']
    for col in cols:
        if col not in gdf_completo.columns:
            gdf_completo[col] = 0.0

    df = gdf_completo[cols].copy()
    df.columns = ['Zona', 'Area_ha', 'NPK', 'NDVI', 'NDRE', 'MO_%', 'Humedad',
                  'N_rec', 'P_rec', 'K_rec', 'Costo_total', 'Rend_sin_fert',
                  'Rend_con_fert', 'Inc_%', 'Textura', 'Arena_%', 'Limo_%', 'Arcilla_%']

    stats = {
        'total_area': df['Area_ha'].sum(),
        'num_zonas': len(df),
        'npk_prom': df['NPK'].mean(),
        'npk_min': df['NPK'].min(),
        'npk_max': df['NPK'].max(),
        'mo_prom': df['MO_%'].mean(),
        'mo_min': df['MO_%'].min(),
        'mo_max': df['MO_%'].max(),
        'humedad_prom': df['Humedad'].mean(),
        'humedad_min': df['Humedad'].min(),
        'humedad_max': df['Humedad'].max(),
        'ndvi_prom': df['NDVI'].mean(),
        'ndvi_min': df['NDVI'].min(),
        'ndvi_max': df['NDVI'].max(),
        'ndre_prom': df['NDRE'].mean(),
        'ndre_min': df['NDRE'].min(),
        'ndre_max': df['NDRE'].max(),
        'rend_sin_prom': df['Rend_sin_fert'].mean(),
        'rend_con_prom': df['Rend_con_fert'].mean(),
        'inc_prom': df['Inc_%'].mean(),
        'costo_total': df['Costo_total'].sum(),
        'textura_dominante': df['Textura'].mode()[0] if not df['Textura'].empty else 'No determinada'
    }

    df_sorted = df.sort_values('NPK')
    n = max_zonas
    indices = [0, len(df)//2, -1] if len(df) >= 3 else list(range(len(df)))
    df_muestra = df_sorted.iloc[indices].head(n)
    return df_muestra, stats

# ================== ANÁLISIS CON PROMPTS MEJORADOS ==================
def generar_analisis_fertilidad(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = f"""Eres un ingeniero agrónomo con especialización en edafología, nutrición vegetal y transición agroecológica, experto en {cultivo}.
Redacta un análisis técnico detallado y orientado a la sostenibilidad. Incluye:
- Evaluación de la fertilidad química (NPK, MO) y su interpretación fisiológica para el cultivo.
- Relación con la textura del suelo y su impacto en la disponibilidad de nutrientes.
- Recomendaciones específicas para mejorar la fertilidad desde un enfoque agroecológico: uso de abonos verdes, compost, biofertilizantes, micorrizas, rotaciones con leguminosas, etc.
- Indicadores de salud del suelo a monitorear (respiración, agregados, carbono orgánico).
- Evita recomendaciones genéricas; sé concreto y basado en los datos proporcionados."""
    
    prompt = f"""
Lote de **{cultivo}** - {stats['num_zonas']} zonas de manejo diferenciado.

**Parámetros promedio del suelo:**
- NPK (índice de fertilidad química): {stats['npk_prom']:.2f} (rango: {stats['npk_min']:.2f} - {stats['npk_max']:.2f})
- Materia orgánica: {stats['mo_prom']:.1f}% (rango: {stats['mo_min']:.1f}% - {stats['mo_max']:.1f}%)
- Textura dominante: {stats['textura_dominante']}

**Detalle por zonas representativas:**
{df_resumen[['Zona', 'NPK', 'MO_%', 'Textura']].to_string(index=False)}

**Instrucciones para el análisis:**
1. Interpretar el nivel de NPK y MO en el contexto del cultivo y la región.
2. Explicar cómo la textura influye en la retención de nutrientes y agua.
3. Proponer **al menos 3 prácticas agroecológicas concretas** para elevar la fertilidad natural (ej. incorporación de leguminosas, elaboración de compost con residuos de cosecha, aplicación de harina de rocas, etc.).
4. Sugerir un plan de monitoreo participativo de indicadores de salud del suelo.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.3)
    if resultado is None:
        return "⚠️ El análisis de fertilidad por IA no está disponible en este momento. Por favor, use el reporte estándar."
    return resultado

def generar_analisis_ndvi_ndre(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = f"""Eres un especialista en teledetección aplicada a la agricultura de precisión y fisiología vegetal, con enfoque en transición agroecológica.
Analiza los índices espectrales NDVI y NDRE de forma técnica, relacionándolos con:
- Estado nutricional del cultivo (especialmente nitrógeno).
- Estrés hídrico o biótico.
- Heterogeneidad espacial y su relación con prácticas de manejo previas.
- Recomendaciones para la agricultura regenerativa: manejo de coberturas, siembra directa, policultivos, etc."""
    
    prompt = f"""
**Cultivo:** {cultivo}
**NDVI promedio:** {stats['ndvi_prom']:.2f} (rango: {stats['ndvi_min']:.2f} - {stats['ndvi_max']:.2f})
**NDRE promedio:** {stats['ndre_prom']:.2f} (rango: {stats['ndre_min']:.2f} - {stats['ndre_max']:.2f})

**Zonas representativas:**
{df_resumen[['Zona', 'NDVI', 'NDRE']].to_string(index=False)}

**Análisis requerido:**
1. Interpretar los valores de NDVI y NDRE: ¿Qué indican sobre biomasa, vigor y nivel de nitrógeno?
2. Identificar zonas de baja productividad potencial y sus posibles causas (compactación, deficiencias, excesos).
3. Recomendar prácticas agroecológicas específicas para homogeneizar el cultivo: aplicación diferenciada de bioinsumos, establecimiento de franjas de biodiversidad funcional, ajuste de densidades de siembra, etc.
4. Sugerir umbrales de alerta temprana basados en estos índices.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.3)
    if resultado is None:
        return "⚠️ Análisis NDVI/NDRE no disponible por error de API."
    return resultado

def generar_analisis_riesgo_hidrico(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = f"""Eres un hidrólogo de suelos y especialista en manejo del agua en agroecosistemas.
Evalúa el riesgo hídrico y propone estrategias de adaptación basadas en principios agroecológicos:
- Captación y almacenamiento de agua de lluvia.
- Mejora de la infiltración y retención (materia orgánica, coberturas, curvas de nivel).
- Selección de cultivos y variedades tolerantes a sequía o anegamiento.
- Integración de sistemas silvopastoriles o agroforestales para regular el ciclo hidrológico."""
    
    prompt = f"""
**Cultivo:** {cultivo}
**Humedad del suelo (índice o contenido):** promedio {stats['humedad_prom']:.2f}, rango {stats['humedad_min']:.2f} - {stats['humedad_max']:.2f}
**Textura dominante:** {stats['textura_dominante']}

**Zonas representativas:**
{df_resumen[['Zona', 'Humedad', 'Textura']].to_string(index=False)}

**Análisis requerido:**
1. Evaluar el riesgo de estrés hídrico (déficit o exceso) según la textura y la variabilidad espacial.
2. Estimar la capacidad de retención de agua disponible para el cultivo.
3. Proponer un plan de manejo agroecológico del agua que incluya al menos:
   - Prácticas para aumentar la infiltración (coberturas muertas/vivas, hoyos de siembra, etc.).
   - Sistemas de captación o microalmacenamiento (barreras, jagüeyes, aljibes).
   - Estrategias de riego complementario de bajo costo (riego por goteo con energía solar, etc.).
4. Indicadores de monitoreo de la eficiencia del uso del agua.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.3)
    if resultado is None:
        return "⚠️ Análisis de riesgo hídrico no disponible."
    return resultado

def generar_analisis_costos(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = f"""Eres un economista ecológico y asesor en gestión de fincas agroecológicas.
Analiza la viabilidad económica de la transición, considerando:
- Reducción de insumos externos (fertilizantes sintéticos, plaguicidas).
- Inversiones en prácticas regenerativas (compost, biofábricas, cercas vivas).
- Incrementos de rendimiento y resiliencia a largo plazo.
- Beneficios no monetarios (salud del suelo, biodiversidad, servicios ecosistémicos)."""
    
    prompt = f"""
**Cultivo:** {cultivo}
**Costo total actual (fertilizantes sintéticos + aplicaciones):** ${stats['costo_total']:,.2f}
**Rendimiento promedio sin fertilización sintética:** {stats['rend_sin_prom']:.0f} kg/ha
**Rendimiento con fertilización convencional:** {stats['rend_con_prom']:.0f} kg/ha
**Incremento porcentual por fertilización:** {stats['inc_prom']:.1f}%

**Zonas representativas (costo e incremento):**
{df_resumen[['Zona', 'Costo_total', 'Inc_%']].to_string(index=False)}

**Análisis requerido:**
1. Evaluar la rentabilidad actual y la dependencia de insumos externos.
2. Calcular el ahorro potencial al reemplazar parcialmente fertilizantes sintéticos por bioinsumos y prácticas agroecológicas.
3. Proponer un escenario de transición a 3 años con inversiones progresivas (compost, abonos verdes, etc.) y estimar el impacto en el margen neto.
4. Identificar incentivos o fuentes de financiamiento para la transición (créditos verdes, pagos por servicios ambientales, etc.).
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.3)
    if resultado is None:
        return "⚠️ Análisis de costos no disponible."
    return resultado

def generar_recomendaciones_integradas(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = f"""Eres un asesor técnico senior en agricultura de precisión y agroecología, con amplia experiencia en transición de sistemas convencionales a regenerativos.
Genera un plan de manejo integrado, priorizando acciones de bajo costo y alto impacto ecológico.
Incluye:
- Calendario agroecológico (coberturas, rotaciones, bioinsumos).
- Diseño de la biodiversidad funcional (bordes, franjas, árboles dispersos).
- Estrategias de manejo de suelo sin labranza.
- Integración de animales si es pertinente.
- Indicadores de éxito y puntos de control."""
    
    prompt = f"""
**Síntesis del diagnóstico:**
- Cultivo: {cultivo}
- NPK promedio: {stats['npk_prom']:.2f}
- Materia orgánica: {stats['mo_prom']:.1f}%
- NDVI promedio: {stats['ndvi_prom']:.2f}
- Incremento de rendimiento con fertilización convencional: {stats['inc_prom']:.1f}%
- Textura dominante: {stats['textura_dominante']}

**Plan de transición agroecológica requerido:**
1. **Fase 1 (primer año):** Acciones inmediatas de bajo costo (incorporación de rastrojos, aplicación de micorrizas, establecimiento de franjas de flores).
2. **Fase 2 (segundo año):** Rotación de cultivos, abonos verdes, reducción del 30-50% de fertilizantes sintéticos.
3. **Fase 3 (tercer año):** Consolidación de sistemas agroforestales, biofábrica en finca, certificación participativa.

Incluye indicadores de monitoreo por fase y recomendaciones para manejar la resistencia al cambio.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.3)
    if resultado is None:
        return "⚠️ Recomendaciones integradas no disponibles por error de API."
    return resultado

# ===== NUEVO: Frases ultra cortas para el reporte campesino =====
def generar_frase_campesina(cultivo, concepto, datos):
    """
    Devuelve una sola oración simple, en español de Perú, máximo 200 caracteres.
    datos puede ser una Serie (valores) o un string (textura).
    """
    system = f"Eres un agricultor mayor experto en {cultivo}. Hablas como campesino peruano (ande o costa), sin tecnicismos. Las frases son de máximo 20 palabras."
    
    if concepto == "Fertilidad":
        promedio = float(datos.mean())
        if promedio > 0.7:
            prompt = f"El suelo tiene fertilidad alta (más de {promedio:.1f}). ¿Qué le decimos al usuario? (una frase)"
        elif promedio > 0.4:
            prompt = f"Fertilidad media ({promedio:.1f}). ¿Qué consejo práctico le das?"
        else:
            prompt = f"Fertilidad baja ({promedio:.1f}). ¿Qué recomiendas para mejorarla rápido?"
    elif concepto == "Rendimiento":
        mejor = float(datos.max())
        peor = float(datos.min())
        prompt = f"El mejor rendimiento es {mejor:.0f} kg y el peor {peor:.0f} kg. ¿Qué zonas atender primero?"
    elif concepto == "Textura":
        textura = datos  # string
        if "arenoso" in textura.lower():
            prompt = f"La tierra es arenosa. ¿Cómo la mejoramos para {cultivo}? (una frase)"
        elif "arcilloso" in textura.lower():
            prompt = f"Tierra arcillosa. ¿Un consejo para que no se encharque?"
        else:
            prompt = f"Tierra franca (mixta). ¿Qué abono natural le pondrías?"
    elif concepto == "Potencial":
        alto = float(datos.max())
        bajo = float(datos.min())
        prompt = f"El área con más potencial da {alto:.0f} kg/ha, la que da menos produce {bajo:.0f} kg. ¿Qué hacer?"
    else:
        return "Observa los números y prioriza las zonas más verdes en los mapas."
    
    respuesta = llamar_groq(prompt, system_prompt=system, temperature=0.4)
    if respuesta is None:
        return "Revisa los mapas: zonas verdes = buenas, rojas = hay que mejorar el suelo."
    return respuesta.strip()[:200]  # Máximo 200 caracteres
