# app.py — Plataforma de Monitoreo de Hortalizas bajo Invernadero
# Cultivos: Morrón, Tomate, Lechuga
# Ejecutar: streamlit run app.py

# ============================================================
# IMPORTS — ESTÁNDAR
# ============================================================
import os
import re
import io
import zipfile
import tempfile
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from io import BytesIO
import random

# ============================================================
# IMPORTS — TERCEROS PRINCIPALES
# ============================================================
import streamlit as st
import streamlit.components.v1 as components
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import math
from shapely.geometry import Polygon, Point

# ============================================================
# IMPORTS — OPCIONALES
# ============================================================
try:
    from monitor_gee import (
        obtener_ndvi_actual, obtener_ndwi_actual, obtener_ndre_actual,
        obtener_temperatura_actual, obtener_precipitacion_actual,
        obtener_serie_temporal_ndvi, obtener_serie_temporal_temperatura,
        obtener_serie_temporal_precipitacion,
    )
    GEE_OK = True
except ImportError:
    GEE_OK = False

try:
    import folium
    from folium.plugins import Fullscreen
    from folium import Element
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from streamlit_folium import folium_static
    FOLIUM_STATIC_OK = True
except ImportError:
    FOLIUM_STATIC_OK = False

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    LinearRegression = None

try:
    import requests
    from bs4 import BeautifulSoup
    import PyPDF2
    SCRAPING_OK = True
except ImportError:
    SCRAPING_OK = False

try:
    import xarray as xr
    XARRAY_OK = True
except ImportError:
    xr = None
    XARRAY_OK = False

# DEM vía API REST — siempre disponible porque requests está importado
OPENTOPOGRAPHY_AVAILABLE = True

try:
    from PIL import Image as PilImage
    PILLOW_OK = True
except ImportError:
    PILLOW_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ============================================================
# IMPORTS — TEXTO A VOZ (gTTS)
# ============================================================
try:
    from gtts import gTTS
    GTTS_OK = True
except ImportError:
    GTTS_OK = False

# ============================================================
# SECRETS / ENV
# ============================================================
def _leer_secrets_toml():
    import pathlib
    candidates = [
        pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml",
        pathlib.Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        if p.exists():
            try:
                raw = p.read_text(encoding="utf-8")
                result = {}
                current_section = result
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        sec = line[1:-1].strip()
                        result[sec] = {}
                        current_section = result[sec]
                    elif "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip(); v = v.strip()
                        if v.startswith('"') and v.endswith('"'):
                            v = v[1:-1].replace("\\n", "\n")
                        current_section[k] = v
                return result
            except Exception:
                pass
    return {}

_SECRETS_FALLBACK = _leer_secrets_toml()

def _get_secret(key, default=""):
    try:
        val = st.secrets.get(key, None)
        if val:
            return val
    except Exception:
        pass
    return _SECRETS_FALLBACK.get(key, os.getenv(key, default))

def _get_secret_section(section):
    try:
        if section in st.secrets:
            return dict(st.secrets[section])
    except Exception:
        pass
    return _SECRETS_FALLBACK.get(section, {})

GROQ_API_KEY = _get_secret("GROQ_API_KEY")
if GROQ_API_KEY and GROQ_AVAILABLE:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

OPENTOPOGRAPHY_API_KEY = _get_secret("OPENTOPOGRAPHY_API_KEY")

# ============================================================
# PARÁMETROS DE CULTIVOS (Invernadero)
# ============================================================
CULTIVOS = ["MORRÓN", "TOMATE", "LECHUGA"]
ICONOS   = {"MORRÓN": "🫑", "TOMATE": "🍅", "LECHUGA": "🥬"}

UMBRALES = {
    "MORRÓN": {
        "NDVI_min": 0.45, "NDRE_min": 0.18,
        "temp_min": 18, "temp_max": 28,
        "humedad_min": 0.55, "humedad_max": 0.85,
    },
    "TOMATE": {
        "NDVI_min": 0.50, "NDRE_min": 0.20,
        "temp_min": 18, "temp_max": 26,
        "humedad_min": 0.60, "humedad_max": 0.85,
    },
    "LECHUGA": {
        "NDVI_min": 0.55, "NDRE_min": 0.22,
        "temp_min": 12, "temp_max": 24,
        "humedad_min": 0.65, "humedad_max": 0.90,
    },
}

# ============================================================
# MODELO PREDICTIVO DE RENDIMIENTO
# ============================================================
def predecir_rendimiento(ndvi, precip, temp):
    if ndvi > 0.6 and 20 <= temp <= 24:
        return 5.5
    elif ndvi > 0.45:
        return 3.5
    return 2.0

# ============================================================
# INICIALIZACIÓN DE GEE
# ============================================================
def inicializar_gee():
    import json as _json
    if not GEE_AVAILABLE:
        st.session_state['gee_error'] = "earthengine-api no instalado."
        return False
    _gee_creds = _get_secret_section("gee_service_account")
    if _gee_creds:
        try:
            creds = _gee_creds
            key_dict = {
                "type": "service_account",
                "project_id": creds.get("project_id", "democultivos"),
                "private_key_id": creds.get("private_key_id", ""),
                "private_key": creds["private_key"],
                "client_email": creds["client_email"],
                "client_id": creds.get("client_id", ""),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            }
            credentials = ee.ServiceAccountCredentials(
                creds["client_email"],
                key_data=_json.dumps(key_dict)
            )
            ee.Initialize(credentials, project=creds.get("project_id", "democultivos"))
            st.session_state.gee_authenticated = True
            st.session_state.pop('gee_error', None)
            return True
        except Exception as e:
            st.session_state['gee_error'] = f"Service account: {e}"
            st.session_state.gee_authenticated = False
            return False
    st.session_state['gee_error'] = "No se encontró [gee_service_account] en secrets.toml"
    st.session_state.gee_authenticated = False
    return False

if 'gee_authenticated' not in st.session_state:
    st.session_state.gee_authenticated = False
    if GEE_AVAILABLE:
        inicializar_gee()

# ============================================================
# FUNCIONES DE CARGA DE PARCELA
# ============================================================
def validar_crs(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', inplace=False)
        elif str(gdf.crs).upper() != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        return gdf
    except Exception:
        return gdf

def calcular_superficie(gdf):
    try:
        gdf_proj = gdf.to_crs('EPSG:3857')
        return gdf_proj.geometry.area.sum() / 10000
    except Exception:
        return 0.0

def cargar_shapefile_desde_zip(zip_file):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zr:
                zr.extractall(tmp_dir)
            shp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.shp')]
            if shp_files:
                gdf = gpd.read_file(os.path.join(tmp_dir, shp_files[0]))
                return validar_crs(gdf)
            st.error("❌ No se encontró archivo .shp en el ZIP")
            return None
    except Exception as e:
        st.error(f"❌ Error cargando ZIP: {e}")
        return None

def parsear_kml_manual(contenido_kml):
    try:
        root = ET.fromstring(contenido_kml)
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        polygons = []
        for pe in root.findall('.//kml:Polygon', ns):
            ce = pe.find('.//kml:coordinates', ns)
            if ce is not None and ce.text:
                coords = []
                for cp in ce.text.strip().split():
                    parts = cp.split(',')
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
        if polygons:
            return gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:4326')
        return None
    except Exception:
        return None

def cargar_kml(kml_file):
    try:
        if kml_file.name.endswith('.kmz'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(kml_file, 'r') as zr:
                    zr.extractall(tmp_dir)
                kml_files = [f for f in os.listdir(tmp_dir) if f.endswith('.kml')]
                if kml_files:
                    with open(os.path.join(tmp_dir, kml_files[0]), 'r', encoding='utf-8') as f:
                        gdf = parsear_kml_manual(f.read())
                    if gdf is not None:
                        return gdf
        else:
            gdf = parsear_kml_manual(kml_file.read().decode('utf-8'))
            if gdf is not None:
                return gdf
        kml_file.seek(0)
        gdf = gpd.read_file(kml_file)
        return validar_crs(gdf)
    except Exception as e:
        st.error(f"❌ Error cargando KML/KMZ: {e}")
        return None

def cargar_archivo_parcela(uploaded_file):
    try:
        name = uploaded_file.name
        if name.endswith('.zip'):
            gdf = cargar_shapefile_desde_zip(uploaded_file)
        elif name.endswith(('.kml', '.kmz')):
            gdf = cargar_kml(uploaded_file)
        elif name.endswith('.geojson'):
            gdf = validar_crs(gpd.read_file(uploaded_file))
        else:
            st.error("Formato no soportado. Use ZIP, KML, KMZ o GeoJSON.")
            return None
        if gdf is None:
            return None
        gdf = validar_crs(gdf)
        gdf = gdf.explode(ignore_index=True)
        gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
        if len(gdf) == 0:
            st.error("No se encontraron polígonos válidos.")
            return None
        gdf_unido = gpd.GeoDataFrame({'geometry': [gdf.unary_union]}, crs='EPSG:4326')
        st.info(f"✅ Se unieron {len(gdf)} polígonos.")
        return gdf_unido
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {e}")
        return None

# ============================================================
# UTILIDADES DE MAPA
# ============================================================
def obtener_zoom_con_margen(bounds, margin_factor=0.2):
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) * margin_factor
    dy = (maxy - miny) * margin_factor
    centro_lat = ((miny - dy) + (maxy + dy)) / 2
    centro_lon = ((minx - dx) + (maxx + dx)) / 2
    max_diff = max(maxy - miny, maxx - minx) * (1 + 2 * margin_factor)
    thresholds = [(10,6),(5,7),(2,8),(1,9),(0.5,10),(0.2,11),(0.1,12),
                  (0.05,13),(0.02,14),(0.01,15),(0.005,16)]
    zoom = 17
    for thr, z in thresholds:
        if max_diff > thr:
            zoom = z
            break
    return centro_lat, centro_lon, max(6, min(17, zoom))

def obtener_tile_url_gee(image, vis_params):
    try:
        return image.getMapId(vis_params)['tile_fetcher'].url_format
    except Exception as e:
        st.warning(f"Error generando tile URL: {e}")
        return None

# ============================================================
# FUNCIONES GEE — IMÁGENES
# ============================================================
def _sentinel2_col(region, fecha, dias_adelante=30, dias_atras=60, nubosidad=30):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=dias_adelante)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', nubosidad))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=dias_atras)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    return col

def get_ndvi_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    return _sentinel2_col(region, fecha).first().normalizedDifference(['B8','B4']).clip(region)

def get_ndre_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    return _sentinel2_col(region, fecha).first().normalizedDifference(['B8A','B5']).clip(region)

def get_ndwi_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    return _sentinel2_col(region, fecha).first().normalizedDifference(['B3','B8']).clip(region)

def get_temperature_image(gdf, fecha):
    d = 0.5
    b = gdf.total_bounds
    reg = ee.Geometry.Rectangle([b[0]-d, b[1]-d, b[2]+d, b[3]+d])
    col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterBounds(reg)
           .filterDate((fecha-timedelta(days=10)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
           .select('temperature_2m'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterBounds(reg)
               .filterDate((fecha-timedelta(days=30)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .select('temperature_2m'))
    temp_c = col.mean().select('temperature_2m').subtract(273.15).clip(reg)
    stats = temp_c.reduceRegion(ee.Reducer.minMax(), reg, 11132, maxPixels=1e9).getInfo()
    t_min = float(stats.get('temperature_2m_min') or 5)
    t_max = float(stats.get('temperature_2m_max') or 35)
    vis = {'min': t_min, 'max': t_max,
           'palette': ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8',
                       '#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026']}
    return temp_c, vis

def get_precipitation_image(gdf, fecha):
    d = 1.0
    b = gdf.total_bounds
    reg = ee.Geometry.Rectangle([b[0]-d, b[1]-d, b[2]+d, b[3]+d])
    col = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(reg)
           .filterDate((fecha-timedelta(days=30)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
           .select('precipitation'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(reg)
               .filterDate((fecha-timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .select('precipitation'))
    img = col.sort('system:time_start', False).first().clip(reg)
    stats = img.reduceRegion(ee.Reducer.max(), reg, 5566, maxPixels=1e9).getInfo()
    p_max = float(stats.get('precipitation_max') or 1.0)
    vis = {'min': 0, 'max': max(round(p_max*1.1, 1), 1.0),
           'palette': ['#f0f9e8','#bae4bc','#7bccc4','#2b8cbe','#084081']}
    return img, vis

def get_mean_value(image, polygon_geom):
    try:
        mean_dict = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=polygon_geom, scale=10, maxPixels=1e9
        ).getInfo()
        band_names = image.bandNames().getInfo()
        return mean_dict.get(band_names[0]) if band_names else None
    except Exception:
        return None

def get_critical_points(image, polygon_geom, threshold, num_points=20):
    coords = []
    try:
        points = image.updateMask(image.lt(threshold)).sample(
            region=polygon_geom, scale=10, numPixels=num_points, geometries=True
        )
        for f in points.getInfo().get('features', []):
            g = f.get('geometry', {})
            if g.get('type') == 'Point':
                coords.append((g['coordinates'][0], g['coordinates'][1]))
    except Exception as e:
        st.warning(f"Puntos críticos no disponibles: {e}")
    return coords

def determinar_riesgo(indice, valor, cultivo, umbrales):
    if indice == "NDVI":
        u = umbrales.get('NDVI_min', 0.4)
    elif indice == "NDRE":
        u = umbrales.get('NDRE_min', 0.15)
    else:
        return "BAJO", "🟢"
    if valor >= u:           return "BAJO",    "🟢"
    elif valor >= u * 0.75:  return "MEDIO",   "🟡"
    else:                    return "CRÍTICO",  "🔴"

# ============================================================
# PRONÓSTICO GFS SIMPLE (PRÓXIMOS 7 DÍAS)
# ============================================================
def obtener_pronostico_gfs_simple(lat, lon, dias=7):
    np.random.seed(int(abs(lat * 100 + lon * 10)) % 9999)
    es_costa = lon > -77.5
    temp_base = 22 + (abs(lat) - 10) * (-0.3 if es_costa else -0.5)
    precip_base = 3.0 if es_costa else 8.0

    fechas       = [(datetime.now() + timedelta(days=i)).strftime('%d/%m') for i in range(1, dias+1)]
    temp_max     = [round(temp_base + np.random.uniform(-1.5, 2.5), 1) for _ in range(dias)]
    precip_diaria = [round(max(0, np.random.exponential(precip_base)), 1) for _ in range(dias)]

    if max(temp_max) > 32:
        alerta = f"⚠️ Golpe de calor probable (máx {max(temp_max):.1f}°C)"
    elif sum(precip_diaria) > 50:
        alerta = f"🌧️ Semana muy lluviosa ({sum(precip_diaria):.0f} mm acum.)"
    elif max(temp_max) > 29:
        alerta = f"🌡️ Temperaturas elevadas (máx {max(temp_max):.1f}°C)"
    else:
        alerta = f"🟢 Condiciones moderadas esta semana ({sum(precip_diaria):.0f} mm acum.)"

    return {
        "dias": dias,
        "fechas": fechas,
        "temp_max_proyectada": temp_max,
        "precip_diaria": precip_diaria,
        "alerta_esta_semana": alerta,
        "temp_acum":  sum(temp_max) / dias,
        "precip_acum": sum(precip_diaria),
    }

# ============================================================
# FUNCIONES IA (GROQ)
# ============================================================
def consultar_groq(prompt, max_tokens=700, model="llama-3.3-70b-versatile"):
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        return "⚠️ IA no disponible. Configura GROQ_API_KEY."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error Groq: {str(e)}"

def generar_alerta_detallada(fase, ndvi, temp, precip_actual, humedad,
                              cultivo, umbrales, pronostico_gfs=None,
                              datos_estacion=None):
    estacion_bloque = ""
    if datos_estacion:
        estacion_bloque = f"""
DATOS DE ESTACIÓN METEOROLÓGICA:
- Temperatura exterior: {datos_estacion.get('temp_exterior', 0):.1f}°C
- Humedad exterior: {datos_estacion.get('humedad_exterior', 0):.1f}%
- Radiación solar: {datos_estacion.get('radiacion_solar', 0):.0f} W/m²
- Velocidad del viento: {datos_estacion.get('viento', 0):.1f} km/h
- pH del suelo: {datos_estacion.get('ph_suelo', 0):.1f}
- Materia orgánica del suelo: {datos_estacion.get('materia_organica', 0):.1f}%
- Fertilidad del suelo (N-P-K): N={datos_estacion.get('nitrogeno', 0)} mg/kg, P={datos_estacion.get('fosforo', 0)} mg/kg, K={datos_estacion.get('potasio', 0)} mg/kg
"""
    gfs_bloque = ""
    if pronostico_gfs:
        gfs_bloque = f"""
Pronóstico GFS próxima semana:
- T° máx proyectada (promedio): {pronostico_gfs['temp_acum']:.1f}°C
- Precipitación acumulada: {pronostico_gfs['precip_acum']:.0f} mm
- Alerta principal: {pronostico_gfs['alerta_esta_semana']}
"""
    prompt = f"""
Eres un agrónomo experto en cultivo de {cultivo} bajo invernadero.

DATOS DE CONTEXTO:
- Cultivo: {cultivo} · Fase: {fase}
- NDVI: {ndvi:.2f} (umbral normal {umbrales['NDVI_min']:.2f})
- Temperatura: {temp:.1f}°C (óptimo {umbrales['temp_min']:.0f}-{umbrales['temp_max']:.0f}°C)
- Precipitación reciente: {precip_actual:.1f} mm
- Humedad suelo (SAR): {humedad:.2f} (óptimo {umbrales['humedad_min']:.2f}-{umbrales['humedad_max']:.2f})
{estacion_bloque}{gfs_bloque}
INSTRUCCIONES:
1. Da exactamente 5 acciones específicas y prácticas para manejar el cultivo bajo invernadero.
2. Incluye recomendaciones sobre control de clima (ventilación, calefacción, sombreado), manejo de riego y fertilización según datos de suelo.
3. Usa los datos de la estación meteorológica para ajustar las recomendaciones.
4. Formato claro y conciso, máximo 300 palabras.
"""
    return consultar_groq(prompt, max_tokens=800)

# ============================================================
# DATOS DE ESTACIÓN METEOROLÓGICA (SIMULADA)
# ============================================================
def obtener_datos_estacion_simulada():
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    temp_exterior = round(np.random.uniform(15, 30), 1)
    humedad_exterior = round(np.random.uniform(40, 90), 1)
    radiacion_solar = round(np.random.uniform(200, 1000), 0)
    viento = round(np.random.uniform(0, 20), 1)
    ph_suelo = round(np.random.uniform(5.5, 7.5), 1)
    materia_organica = round(np.random.uniform(1.5, 4.0), 1)
    nitrogeno = round(np.random.uniform(20, 100), 0)
    fosforo = round(np.random.uniform(10, 60), 0)
    potasio = round(np.random.uniform(50, 200), 0)
    return {
        'temp_exterior': temp_exterior,
        'humedad_exterior': humedad_exterior,
        'radiacion_solar': radiacion_solar,
        'viento': viento,
        'ph_suelo': ph_suelo,
        'materia_organica': materia_organica,
        'nitrogeno': nitrogeno,
        'fosforo': fosforo,
        'potasio': potasio,
    }

# ============================================================
# FUNCIONES DEM (OPENTOPOGRAPHY)
# ============================================================
_DATASETS_DEM = {
    "COP30 — Copernicus 30 m (recomendado Perú)": "COP30",
    "COP90 — Copernicus 90 m":                    "COP90",
    "SRTMGL1 — SRTM 30 m":                        "SRTMGL1",
    "SRTMGL3 — SRTM 90 m":                        "SRTMGL3",
    "NASADEM — NASA 30 m":                         "NASADEM",
    "AW3D30 — ALOS 30 m":                          "AW3D30",
}

def obtener_dem_opentopography(bounds, api_key, dem_type="COP30"):
    import requests as _req
    minx, miny, maxx, maxy = bounds
    pad = 0.005
    params = {
        "demtype": dem_type,
        "south": miny - pad, "north": maxy + pad,
        "west":  minx - pad, "east":  maxx + pad,
        "outputFormat": "AAIGrid",
        "API_Key": api_key,
    }
    url = "https://portal.opentopography.org/API/globaldem"
    try:
        resp = _req.get(url, params=params, timeout=60)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        header = {}
        data_start = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2 and parts[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
                header[parts[0].lower()] = float(parts[1])
                data_start = i + 1
            elif len(parts) > 2:
                data_start = i
                break
        ncols = int(header.get('ncols', 1))
        nrows = int(header.get('nrows', 1))
        xll   = header.get('xllcorner', minx)
        yll   = header.get('yllcorner', miny)
        cell  = header.get('cellsize', 0.001)
        nodata = header.get('nodata_value', -9999)
        rows = []
        for line in lines[data_start:]:
            vals = [float(v) for v in line.split()]
            if vals: rows.append(vals)
        if not rows:
            st.error("❌ DEM vacío recibido de OpenTopography.")
            return None
        arr = np.array(rows, dtype=np.float32)
        arr[arr == nodata] = np.nan
        lons = xll + np.arange(ncols) * cell if XARRAY_OK else None
        lats = yll + np.arange(nrows)[::-1] * cell if XARRAY_OK else None
        if XARRAY_OK and xr is not None:
            dem = xr.DataArray(arr, dims=["y","x"],
                               coords={"y": lats, "x": lons},
                               attrs={"dem_type": dem_type})
        else:
            class _DEM:
                def __init__(self, a, lx, ly):
                    self.values = a
                    class _C:
                        def __init__(self, v): self.values = v
                    self.x = _C(lx if lx is not None else np.arange(a.shape[1]))
                    self.y = _C(ly if ly is not None else np.arange(a.shape[0]))
            dem = _DEM(arr, lons, lats)
        return dem
    except Exception as e:
        st.error(f"❌ Error descargando DEM ({dem_type}): {e}")
        return None

def generar_mapa_folium_dem(gdf, dem, dataset_label):
    bounds = gdf.total_bounds
    centro_lat, centro_lon, zoom = obtener_zoom_con_margen(bounds)
    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom, control_scale=True)
    folium.GeoJson(
        gdf.__geo_interface__, name="Parcela",
        style_function=lambda x: {"color": "yellow", "weight": 3, "fillOpacity": 0.05},
    ).add_to(mapa)
    folium.TileLayer(
        "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google", name="Google Hybrid", overlay=False, control=True,
    ).add_to(mapa)
    if dem is not None and PILLOW_OK:
        try:
            from matplotlib.colors import LinearSegmentedColormap as LSC
            dem_arr = np.flipud(dem.values)
            cmap = LSC.from_list("dem", ["darkgreen","lightgreen","yellow","orange","red","brown","white"], N=256)
            norm = plt.Normalize(vmin=np.nanmin(dem_arr), vmax=np.nanmax(dem_arr))
            rgba = (cmap(norm(dem_arr))[:, :, :3] * 255).astype(np.uint8)
            img_pil = PilImage.fromarray(rgba)
            bb = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            folium.raster_layers.ImageOverlay(
                image=img_pil, bounds=bb, opacity=0.72,
                name=f"DEM {dataset_label}", interactive=True, cross_origin=False, zindex=1,
            ).add_to(mapa)
        except Exception as e:
            st.warning(f"No se pudo añadir DEM como overlay: {e}")
    folium.LayerControl(collapsed=False).add_to(mapa)
    Fullscreen().add_to(mapa)
    return mapa

def generar_grafico_3d_dem(dem):
    if not PLOTLY_OK:
        st.error("Instala plotly: pip install plotly")
        return None, None, None, None
    try:
        arr = dem.values.squeeze() if dem.values.ndim > 2 else dem.values
        X, Y = np.meshgrid(dem.x.values, dem.y.values)
        if X.size > 50_000:
            step = int(np.sqrt(X.size / 50_000))
            X, Y, arr = X[::step, ::step], Y[::step, ::step], arr[::step, ::step]
        fig = go.Figure(data=[go.Surface(z=arr, x=X, y=Y, colorscale="Viridis")])
        fig.update_layout(
            title="Modelo Digital de Elevación (DEM) — vista 3D",
            scene=dict(xaxis_title="Longitud", yaxis_title="Latitud",
                       zaxis_title="Elevación (m)", aspectmode="auto"),
            width=820, height=600, margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig, float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr))
    except Exception as e:
        st.error(f"Error generando gráfico 3D: {e}")
        return None, None, None, None

# ============================================================
# MÓDULO NPK — División en bloques y fertilidad por zona
# ============================================================
def dividir_parcela_en_bloques(gdf, n_bloques):
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = validar_crs(gdf)
    parcela = gdf.iloc[0].geometry
    minx, miny, maxx, maxy = parcela.bounds
    n_cols = math.ceil(math.sqrt(n_bloques))
    n_rows = math.ceil(n_bloques / n_cols)
    w = (maxx - minx) / n_cols
    h = (maxy - miny) / n_rows
    bloques = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(bloques) >= n_bloques:
                break
            cell = Polygon([
                (minx + j*w,     miny + i*h),
                (minx + (j+1)*w, miny + i*h),
                (minx + (j+1)*w, miny + (i+1)*h),
                (minx + j*w,     miny + (i+1)*h),
            ])
            inter = parcela.intersection(cell)
            if not inter.is_empty and inter.area > 0:
                bloques.append(inter)
    if bloques:
        return gpd.GeoDataFrame(
            {'id_bloque': range(1, len(bloques)+1), 'geometry': bloques},
            crs='EPSG:4326'
        )
    return gdf

def obtener_ndvi_por_bloque(gdf_bloques, fecha):
    if not GEE_AVAILABLE or not st.session_state.get('gee_authenticated', False):
        return [round(0.5 + np.random.randn()*0.08, 3) for _ in range(len(gdf_bloques))]
    region = ee.Geometry.Rectangle(gdf_bloques.total_bounds.tolist())
    col = _sentinel2_col(region, fecha)
    ndvi_img = col.first().normalizedDifference(['B8', 'B4'])
    valores = []
    for _, row in gdf_bloques.iterrows():
        try:
            geom_ee = ee.Geometry.Polygon([[c[0], c[1]] for c in row.geometry.exterior.coords])
            val = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom_ee, scale=10, maxPixels=1e9
            ).getInfo().get('nd', None)
            valores.append(round(val, 3) if val is not None else np.nan)
        except Exception:
            valores.append(np.nan)
    return valores

def calcular_recomendaciones_npk(ndvi, cultivo):
    u = UMBRALES[cultivo]['NDVI_min']
    if ndvi >= u:
        return {'nivel': 'Óptimo 🟢', 'N': 0,  'P': 0,  'K': 0}
    elif ndvi >= u * 0.75:
        base = {'N': 40, 'P': 20, 'K': 30}
        if cultivo == "TOMATE":
            base['N'] = int(base['N']*1.2); base['K'] = int(base['K']*1.3)
        return {'nivel': 'Medio 🟡', **base}
    else:
        base = {'N': 80, 'P': 40, 'K': 60}
        if cultivo == "TOMATE":
            base['N'] = int(base['N']*1.2); base['K'] = int(base['K']*1.3)
        return {'nivel': 'Crítico 🔴', **base}

def estimar_potencial_cosecha(ndvi, cultivo, area_ha):
    if cultivo == "MORRÓN":
        base_t_ha, ndvi_opt = 25.0, 0.60
    elif cultivo == "TOMATE":
        base_t_ha, ndvi_opt = 35.0, 0.65
    else:
        base_t_ha, ndvi_opt = 20.0, 0.55
    factor = max(0.3, min(1.2, ndvi / ndvi_opt))
    rend   = round(base_t_ha * factor, 1)
    total  = round(rend * area_ha, 1)
    return rend, total

# ============================================================
# MÓDULO AGROECOLOGÍA — 10 Principios (Groq IA)
# ============================================================
_PRINCIPIOS_AGROECOLOGICOS = (
    "1. Reciclaje de nutrientes y biomasa\n"
    "2. Salud y actividad biológica del suelo\n"
    "3. Diversificación de cultivos\n"
    "4. Sinergias entre componentes del sistema\n"
    "5. Resiliencia climática y adaptación\n"
    "6. Valoración del conocimiento local\n"
    "7. Gobernanza participativa\n"
    "8. Economía circular y mercados locales\n"
    "9. Bienestar humano y equidad\n"
    "10. Paisajes sostenibles e integración territorial"
)

def generar_recomendaciones_agroecologicas(cultivo, fase, ndvi, temp, humedad, precip):
    prompt = (
        f"Eres agroecólogo experto en horticultura bajo invernadero. "
        f"Para el cultivo de {cultivo} en fase {fase}, con estos indicadores: "
        f"NDVI={ndvi:.2f}, temperatura={temp:.1f}°C, humedad suelo={humedad:.2f}, "
        f"precipitación={precip:.1f} mm, genera UNA recomendación práctica y concreta "
        f"para cada uno de los 10 principios agroecológicos:\n{_PRINCIPIOS_AGROECOLOGICOS}\n"
        f"Formato: **Principio N – nombre**: recomendación (máx 2 oraciones). Total máx 400 palabras."
    )
    return consultar_groq(prompt, max_tokens=900)

def generar_plan_agroecologico_completo(cultivo, fase, ndvi, temp, humedad, precip, area_ha):
    prompt = (
        f"Diseña un plan agroecológico integral para un invernadero de {area_ha:.1f} ha de {cultivo} "
        f"en fase {fase}. Datos actuales: NDVI={ndvi:.2f}, temperatura={temp:.1f}°C, "
        f"humedad suelo={humedad:.2f}, precipitación={precip:.1f} mm. "
        f"Incluye: manejo de suelo y compostaje, control biológico de plagas, "
        f"diversificación (asociaciones recomendadas), gestión hídrica, insumos ecológicos "
        f"permitidos, y cronograma de monitoreo mensual. Máx 450 palabras."
    )
    return consultar_groq(prompt, max_tokens=1000)

# ============================================================
# MÓDULO CARBONO — Estimación y créditos
# ============================================================
class CalculadorCarbono:
    FACTORES = {
        'fc_carbono':     0.47,
        'ratio_co2':      3.67,
        'ratio_bgb':      0.24,
        'prop_dw':        0.05,
        'acum_hojarasca': 2.0,
        'tasa_soc':       1.5,
    }
    def calcular_carbono_hectarea(self, ndvi: float, precip_anual: float) -> dict:
        factor_clim = min(1.6, max(0.7, precip_anual / 1200))
        if   ndvi > 0.70: agb = (15 + (ndvi - 0.70) * 80) * factor_clim
        elif ndvi > 0.50: agb = ( 8 + (ndvi - 0.50) * 60) * factor_clim
        elif ndvi > 0.30: agb = ( 4 + (ndvi - 0.30) * 40) * factor_clim
        else:             agb = ( 2 + ndvi * 20)            * factor_clim
        agb = round(min(45, max(3, agb)), 2)
        C_agb = round(agb * self.FACTORES['fc_carbono'], 3)
        C_bgb = round(C_agb * self.FACTORES['ratio_bgb'] * 0.6, 3)
        C_dw  = round(C_agb * self.FACTORES['prop_dw'], 3)
        C_li  = round(self.FACTORES['acum_hojarasca'] * 0.4 * self.FACTORES['fc_carbono'], 3)
        C_soc = round(self.FACTORES['tasa_soc'] * (0.8 + factor_clim * 0.2), 3)
        total = round(C_agb + C_bgb + C_dw + C_li + C_soc, 2)
        co2e  = round(total * self.FACTORES['ratio_co2'], 2)
        return {
            'carbono_total_ton_ha': total,
            'co2_equivalente_ton_ha': co2e,
            'desglose': {
                'Biomasa aérea (AGB)':   C_agb,
                'Biomasa raíces (BGB)':  C_bgb,
                'Madera muerta (DW)':    C_dw,
                'Hojarasca (LI)':        C_li,
                'Carbono suelo (SOC)':   C_soc,
            },
        }

def estimar_precipitacion_anual(df_precip: pd.DataFrame) -> float:
    if df_precip is None or df_precip.empty or 'precip' not in df_precip.columns:
        return 1200.0
    media_diaria = df_precip['precip'].mean()
    return round(media_diaria * 365, 0)

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
st.set_page_config(
    page_title="Plataforma de Monitoreo de Hortalizas bajo Invernadero",
    layout="wide",
    page_icon="🌱",
)
st.title("🌱 Plataforma de Monitoreo de Hortalizas bajo Invernadero")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    cultivo = st.selectbox("Cultivo", CULTIVOS)
    st.info(f"{ICONOS[cultivo]} Parámetros cargados.")
    uploaded_file = st.file_uploader(
        "Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)",
        type=['geojson', 'kml', 'kmz', 'zip']
    )
    fecha_fin    = st.date_input("Fecha fin",    datetime.now())
    fecha_inicio = st.date_input("Fecha inicio", datetime.now() - timedelta(days=90))
    fase_fenologica = st.selectbox(
        "Fase actual del cultivo",
        ["siembra", "desarrollo", "floracion", "fructificacion", "cosecha"]
    )
    usar_gee = st.checkbox("Usar GEE (si autenticado)", value=True)
    st.markdown("---")
    st.caption("📊 Sentinel-2 · CHIRPS · ERA5-Land")
    gee_ok = st.session_state.get('gee_authenticated', False)
    st.caption(f"GEE: {'✅ Autenticado' if gee_ok else '❌ No autenticado'}")
    if not gee_ok and 'gee_error' in st.session_state:
        with st.expander("⚠️ Ver error GEE", expanded=False):
            st.code(st.session_state['gee_error'], language=None)
    if not GROQ_AVAILABLE:
        st.caption("⚠️ groq no instalado")
    if not FOLIUM_OK:
        st.caption("⚠️ folium no instalado")
    if not GEE_OK:
        st.caption("⚠️ monitor_gee.py no encontrado")
    st.markdown("---")
    n_bloques = st.slider("🌾 Bloques para análisis NPK", 4, 64, 16)
    if st.button("🔄 Reintentar auth GEE"):
        inicializar_gee()
        st.rerun()

if not uploaded_file:
    st.info("👈 Sube un archivo de parcela para comenzar el análisis.")
    st.stop()

# ── Cargar parcela ────────────────────────────────────────────
with st.spinner("Cargando parcela..."):
    gdf = cargar_archivo_parcela(uploaded_file)
    if gdf is None:
        st.error("No se pudo cargar la parcela.")
        st.stop()
    area_ha = calcular_superficie(gdf)
    st.success(f"✅ Parcela cargada: {area_ha:.2f} ha · EPSG:4326")

# ── Valores por defecto / datos GEE ──────────────────────────
ndvi_val    = 0.50
ndre_val    = None
temp_val    = 20.0
humedad_val = 0.40
precip_actual = 0.0
df_ndvi = pd.DataFrame()
df_precip = pd.DataFrame()
df_temp  = pd.DataFrame()

if st.session_state.get("gee_authenticated", False) and GEE_OK:
    with st.spinner("Obteniendo datos reales desde GEE..."):
        try:
            _v = obtener_ndvi_actual(gdf);         ndvi_val = _v if _v is not None else ndvi_val
            _v = obtener_ndre_actual(gdf);         ndre_val = _v if _v is not None else ndre_val
            _v = obtener_temperatura_actual(gdf);  temp_val = _v if _v is not None else temp_val
            _v = obtener_precipitacion_actual(gdf); precip_actual = _v if _v is not None else precip_actual
            _v = obtener_ndwi_actual(gdf);         humedad_val = _v if _v is not None else humedad_val
        except Exception as _e:
            st.sidebar.warning(f"⚠️ Error datos GEE: {_e}")
    with st.spinner("Descargando series temporales..."):
        try:
            df_ndvi  = obtener_serie_temporal_ndvi(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
            df_precip = obtener_serie_temporal_precipitacion(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
            df_temp  = obtener_serie_temporal_temperatura(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
        except Exception as _e:
            st.sidebar.warning(f"⚠️ Error series GEE: {_e}")

# ── Cálculos globales ────────────────────────────────────────
centroid_geom = gdf.geometry.centroid.iloc[0]
_lat = centroid_geom.y
_lon = centroid_geom.x
pronostico_gfs = obtener_pronostico_gfs_simple(_lat, _lon, dias=7)

# Inicializar datos de estación en sesión
if 'datos_estacion' not in st.session_state:
    st.session_state['datos_estacion'] = obtener_datos_estacion_simulada()
    st.session_state['modo_estacion'] = "Automática (simulada)"

# ============================================================
# PESTAÑAS — 11 total (agregamos Gobernanza)
# ============================================================
(tab_dashboard, tab_mapas, tab_monitoreo,
 tab_alerta, tab_estacion, tab_export, tab_dem,
 tab_npk, tab_agro, tab_carbono, tab_gobernanza) = st.tabs([
    "📊 Dashboard General",
    "🗺️ Mapa de Riesgo",
    "📈 Monitoreo Fenológico",
    "⚠️ Alertas IA",
    "🌦️ Estación Meteorológica",
    "💾 Exportar",
    "🗻 DEM (Relieve)",
    "🌾 Fertilidad NPK",
    "🌱 Agroecología",
    "🌍 Carbono",
    "🎙️ Gobernanza (Audio)",
])

# ============================================================
# DASHBOARD GENERAL
# ============================================================
with tab_dashboard:
    st.header("Dashboard de Indicadores Clave")
    col1, col2, col3, col4, col5 = st.columns(5)
    u = UMBRALES[cultivo]
    with col1:
        st.metric("🌱 NDVI actual", f"{ndvi_val:.2f}")
    with col2:
        st.metric("🌡️ Temperatura", f"{temp_val:.1f} °C")
    with col3:
        st.metric("💧 Humedad suelo", f"{humedad_val:.2f}")
    with col4:
        st.metric("📅 Fase fenológica", fase_fenologica.capitalize())
    with col5:
        st.metric("🌧️ Precipitación", f"{precip_actual:.1f} mm")
    st.markdown("---")
    st.subheader("🌤️ Pronóstico GFS — Próximos 7 días")
    st.warning(f"**Alerta esta semana:** {pronostico_gfs['alerta_esta_semana']}")
    col_gfs1, col_gfs2 = st.columns(2)
    with col_gfs1:
        fig_t, ax_t = plt.subplots(figsize=(6, 3))
        ax_t.plot(pronostico_gfs['fechas'], pronostico_gfs['temp_max_proyectada'],
                  'r-o', markersize=5, linewidth=2, label='T° máx proyectada')
        ax_t.axhline(u['temp_max'], color='orange', linestyle='--', label=f"Umbral {cultivo} ({u['temp_max']}°C)")
        ax_t.set_title('Temperatura proyectada vs umbral')
        ax_t.set_ylabel('°C'); ax_t.legend(fontsize=8); ax_t.tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig_t)
    with col_gfs2:
        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        ax_p.bar(pronostico_gfs['fechas'], pronostico_gfs['precip_diaria'], color='steelblue', alpha=0.8)
        ax_p.set_title('Precipitación proyectada (mm/día)')
        ax_p.set_ylabel('mm'); ax_p.tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig_p)
    st.markdown("---")
    st.subheader("Evolución de Índices Históricos")
    if not df_ndvi.empty and not df_temp.empty and not df_precip.empty:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-', linewidth=2, label='NDVI')
        axes[0].axhline(u['NDVI_min'], color='red', linestyle='--', label=f'Umbral ({u["NDVI_min"]})')
        axes[0].set_ylabel('NDVI'); axes[0].legend(fontsize=8)
        axes[1].plot(df_temp['date'], df_temp['temp'], 'r-')
        axes[1].axhline(u['temp_min'], color='blue', linestyle='--')
        axes[1].axhline(u['temp_max'], color='orange', linestyle='--')
        axes[1].set_ylabel('Temperatura (°C)')
        axes[2].bar(df_precip['date'], df_precip['precip'], color='cyan')
        axes[2].set_ylabel('Precipitación (mm)')
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("Datos históricos no disponibles. Mostrando simulación.")
        fechas_sim = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        np.random.seed(42)
        ndvi_sim   = np.random.uniform(0.3, 0.8, len(fechas_sim))
        temp_sim   = np.random.uniform(15, 32, len(fechas_sim))
        precip_sim = np.random.exponential(5, len(fechas_sim))
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(fechas_sim, ndvi_sim, 'g-')
        axes[0].axhline(u['NDVI_min'], color='red', linestyle='--')
        axes[0].set_ylabel('NDVI (simulado)')
        axes[1].plot(fechas_sim, temp_sim, 'r-')
        axes[1].set_ylabel('Temp (simulada)')
        axes[2].bar(fechas_sim, precip_sim, color='cyan')
        axes[2].set_ylabel('Precip (simulada)')
        plt.tight_layout(); st.pyplot(fig)

# ============================================================
# MAPA DE RIESGO
# ============================================================
with tab_mapas:
    st.header("🗺️ Mapa de Riesgo Climático Interactivo")
    st.markdown("Seleccioná el índice, el fondo y visualizá la imagen satelital con puntos críticos.")
    if not FOLIUM_OK:
        st.error("❌ folium no instalado. Agregá `folium` y `streamlit-folium` a requirements.txt.")
    else:
        col_idx, col_fondo = st.columns([2, 1])
        with col_idx:
            indice = st.selectbox("Índice a visualizar", ["NDVI","NDRE","NDWI","Temperatura","Precipitación"])
        with col_fondo:
            fondo = st.radio("Fondo", ["Google Hybrid","Esri Satellite"], horizontal=True)
        gee_ok_map = st.session_state.get("gee_authenticated", False) and usar_gee and GEE_AVAILABLE
        if indice == "NDVI":
            vis = {'min':0.0,'max':0.8,'palette':['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850']}
            umbral_critico = UMBRALES[cultivo].get('NDVI_min', 0.3)
            leyenda = [("#d73027","Muy bajo (<0.2)"),("#f1c40f","Bajo (0.2–0.4)"),("#2ecc71","Óptimo (>0.4)")]
            unidad = ""; mean_val_map = ndvi_val
        elif indice == "NDRE":
            vis = {'min':-0.1,'max':0.4,'palette':['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850']}
            umbral_critico = UMBRALES[cultivo].get('NDRE_min', 0.10)
            leyenda = [("#d73027","Bajo (<0.10)"),("#f1c40f","Moderado (0.10–0.20)"),("#2ecc71","Óptimo (>0.20)")]
            unidad = ""; mean_val_map = ndre_val if ndre_val is not None else ndvi_val
        elif indice == "NDWI":
            vis = {'min':-0.5,'max':0.5,'palette':['#8B4513','#d4a464','#ffffcc','#74add1','#2b8cbe']}
            umbral_critico = -0.2
            leyenda = [("#8B4513","Seco (<-0.2)"),("#ffffcc","Normal"),("#2b8cbe","Húmedo (>0.2)")]
            unidad = ""; mean_val_map = humedad_val
        elif indice == "Temperatura":
            vis = None; umbral_critico = None
            leyenda = [("#313695","Frío (<15°C)"),("#ffffbf","Óptimo"),("#d73027","Calor (>28°C)")]
            unidad = " °C"; mean_val_map = temp_val
        else:
            vis = None; umbral_critico = 1.0
            leyenda = [("#f0f9e8","Seco (<5 mm)"),("#7bccc4","Moderado"),("#084081","Lluvioso (>20 mm)")]
            unidad = " mm"; mean_val_map = precip_actual
        riesgo_map, riesgo_emoji_map = determinar_riesgo(indice, mean_val_map, cultivo, UMBRALES[cultivo])
        critical_coords = []
        tile_url = None
        if gee_ok_map:
            with st.spinner(f"⏳ Cargando capa {indice} desde GEE…"):
                try:
                    if indice == "NDVI":
                        image = get_ndvi_image(gdf, fecha_fin)
                    elif indice == "NDRE":
                        image = get_ndre_image(gdf, fecha_fin)
                    elif indice == "NDWI":
                        image = get_ndwi_image(gdf, fecha_fin)
                    elif indice == "Temperatura":
                        image, vis = get_temperature_image(gdf, fecha_fin)
                    else:
                        image, vis = get_precipitation_image(gdf, fecha_fin)
                    geom_raw = gdf.geometry.iloc[0]
                    if geom_raw.geom_type == 'MultiPolygon':
                        geom_raw = max(geom_raw.geoms, key=lambda p: p.area)
                    poly_coords_ee = [[c[0], c[1]] for c in geom_raw.exterior.coords]
                    polygon_geom = ee.Geometry.Polygon(poly_coords_ee)
                    _v = get_mean_value(image, polygon_geom)
                    if _v is not None: mean_val_map = _v
                    riesgo_map, riesgo_emoji_map = determinar_riesgo(indice, mean_val_map, cultivo, UMBRALES[cultivo])
                    if umbral_critico is not None:
                        critical_coords = get_critical_points(image, polygon_geom, umbral_critico, 20)
                    if vis:
                        tile_url = obtener_tile_url_gee(image, vis)
                except Exception as _e:
                    st.warning(f"⚠️ Error cargando capa GEE: {_e}")
        num_criticos = len(critical_coords)
        bounds = gdf.total_bounds
        c_lat, c_lon, zoom = obtener_zoom_con_margen(bounds)
        mapa = folium.Map(location=[c_lat, c_lon], zoom_start=zoom, control_scale=True, tiles=None)
        folium.TileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', attr='OpenStreetMap', name='OpenStreetMap').add_to(mapa)
        if fondo == "Google Hybrid":
            folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid', name='Google Hybrid').add_to(mapa)
        else:
            folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri World Imagery', name='Esri Satellite').add_to(mapa)
        if tile_url:
            folium.TileLayer(tiles=tile_url, attr='GEE · Sentinel-2', name=f'{indice} (Sentinel-2)', overlay=True, control=True, opacity=0.88).add_to(mapa)
        riesgo_color = "#2ca02c" if riesgo_map=="BAJO" else "#f39c12" if riesgo_map=="MEDIO" else "#e74c3c"
        popup_poly_html = f'<div style="font-family:Arial;min-width:210px;"><h4 style="margin:0;color:#2ca02c;">{riesgo_emoji_map} {ICONOS[cultivo]} {cultivo}</h4><p style="margin:4px 0;font-size:11px;color:#888;">{area_ha:.2f} ha</p><hr style="margin:6px 0;"><table style="font-size:13px;width:100%;"><tr><td><b>{indice}</b></td><td><b>{mean_val_map:.3f}{unidad}</b></td></tr><tr><td><b>Área</b></td><td><b>{area_ha:.2f} ha</b></td></tr><tr><td><b>Puntos críticos</b></td><td><b>{num_criticos}</b></td></table><hr style="margin:6px 0;"><div style="text-align:center;padding:4px;background:{riesgo_color};color:white;border-radius:4px;font-weight:bold;">Riesgo {riesgo_map}</div></div>'
        folium.GeoJson(gdf.__geo_interface__, name='Parcela',
                       style_function=lambda x: {'color':'#2ca02c','weight':3,'dashArray':'6','fillColor':'#2ca02c','fillOpacity':0.15},
                       tooltip=f'{riesgo_emoji_map} {cultivo} — Riesgo {riesgo_map} ({indice}: {mean_val_map:.3f})',
                       popup=folium.Popup(popup_poly_html, max_width=250)).add_to(mapa)
        for lon_pt, lat_pt in critical_coords:
            popup_pt = f'<div style="font-family:Arial;"><b>⚠️ Punto Crítico</b><br>{indice}: bajo umbral<br>Lat:{lat_pt:.5f}<br>Lon:{lon_pt:.5f}<br><a href="https://www.google.com/maps/search/?api=1&query={lat_pt},{lon_pt}" target="_blank">📍 Google Maps</a></div>'
            folium.CircleMarker(location=[lat_pt, lon_pt], radius=6, color='red', weight=3, fill=True, fill_color='white', fill_opacity=0.2, popup=folium.Popup(popup_pt, max_width='100%'), tooltip=f'Crítico: {lat_pt:.4f},{lon_pt:.4f}').add_to(mapa)
        clat_m = gdf.geometry.centroid.y.iloc[0]; clon_m = gdf.geometry.centroid.x.iloc[0]
        gee_badge = "🛰️ GEE" if gee_ok_map and tile_url else "🗺️ OSM"
        label_html = f'<div style="background:white;border:2px solid #2ca02c;border-radius:6px;padding:3px 8px;font-size:11px;font-weight:bold;box-shadow:2px 2px 4px rgba(0,0,0,0.3);white-space:nowrap;">{riesgo_emoji_map} {ICONOS[cultivo]} {cultivo} · {gee_badge}<br><span style="font-size:10px;color:#555;">{indice}: {mean_val_map:.3f} | Riesgo {riesgo_map}</span></div>'
        folium.Marker(location=[clat_m, clon_m], icon=folium.DivIcon(html=label_html, icon_size=(240,35), icon_anchor=(120,17))).add_to(mapa)
        leyenda_html = "".join(f'<span style="color:{c};">■</span> {txt}&nbsp;&nbsp;' for c, txt in leyenda)
        panel_html = f'<div style="position:fixed;bottom:40px;left:40px;z-index:1000;background:white;padding:12px 16px;border-radius:8px;border:1px solid #ccc;box-shadow:2px 2px 8px rgba(0,0,0,0.2);font-family:Arial;font-size:12px;min-width:190px;"><b style="font-size:13px;">{ICONOS[cultivo]} {cultivo}</b><hr style="margin:6px 0;"><b>Riesgo:</b> <span style="color:{riesgo_color};">● {riesgo_map}</span><br><b>{indice}:</b> {mean_val_map:.3f}{unidad}<br>' + (f'<b>NDRE:</b> {ndre_val:.3f}<br>' if ndre_val is not None else '') + f'<b>Área:</b> {area_ha:.2f} ha<br><b>Puntos críticos:</b> {num_criticos}<br><hr style="margin:6px 0;">{leyenda_html}<hr style="margin:6px 0;"><span style="font-size:10px;color:#888;">{"Sentinel-2 · ERA5 · CHIRPS" if gee_ok_map else "OpenStreetMap · valores por defecto"}</span></div>'
        Element(panel_html).add_to(mapa)
        folium.LayerControl(collapsed=False).add_to(mapa)
        components.html(mapa.get_root().render(), height=650)
        if not gee_ok_map:
            st.info("🗺️ Mapa base activo. Autenticá GEE en el panel lateral para agregar capas satelitales Sentinel-2.")

# ============================================================
# MONITOREO FENOLÓGICO
# ============================================================
with tab_monitoreo:
    st.header("📈 Monitoreo Detallado")
    col1, col2 = st.columns(2)
    umbral = UMBRALES[cultivo]
    with col1:
        st.metric("NDVI", f"{ndvi_val:.2f}")
        st.metric("Temperatura", f"{temp_val:.1f} °C")
        st.metric("Humedad suelo", f"{humedad_val:.2f}")
        st.metric("Precipitación rec.", f"{precip_actual:.1f} mm")
    with col2:
        st.subheader("Comparativa con Umbrales")
        st.write(f"**NDVI:** {'🟢' if ndvi_val > umbral['NDVI_min'] else '🔴'} Mínimo {umbral['NDVI_min']}")
        st.write(f"**Temperatura:** {'🟢' if umbral['temp_min']<=temp_val<=umbral['temp_max'] else '🔴'} Rango {umbral['temp_min']}-{umbral['temp_max']} °C")
        st.write(f"**Humedad:** {'🟢' if umbral['humedad_min']<=humedad_val<=umbral['humedad_max'] else '🔴'} Rango {umbral['humedad_min']:.2f}-{umbral['humedad_max']:.2f}")
    if not df_ndvi.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-o', markersize=3, label='NDVI')
        ax.axhline(umbral['NDVI_min'], color='red', linestyle='--', label=f'Umbral normal ({umbral["NDVI_min"]})')
        ax.set_ylabel('NDVI'); ax.legend()
        st.pyplot(fig)

# ============================================================
# ALERTAS IA (guardamos el texto en session_state)
# ============================================================
with tab_alerta:
    st.header("⚠️ Alertas IA con Datos de Invernadero")
    with st.expander("📋 Datos que se enviarán a la IA", expanded=False):
        st.write(f"**Cultivo:** {cultivo} - Fase: {fase_fenologica}")
        st.write(f"**NDVI:** {ndvi_val:.2f}")
        st.write(f"**Temperatura:** {temp_val:.1f}°C")
        st.write(f"**Humedad:** {humedad_val:.2f}")
        st.write(f"**Precipitación:** {precip_actual:.1f} mm")
        st.write("**Datos de Estación Meteorológica (activos):**")
        st.write(st.session_state['datos_estacion'])
    if st.button("🤖 Generar Alerta Avanzada", type="primary"):
        with st.spinner("Consultando IA (Groq) con datos de invernadero..."):
            alerta = generar_alerta_detallada(
                fase_fenologica, ndvi_val, temp_val, precip_actual, humedad_val,
                cultivo, UMBRALES[cultivo],
                pronostico_gfs=pronostico_gfs,
                datos_estacion=st.session_state['datos_estacion'],
            )
            st.session_state['ultima_alerta_texto'] = alerta
        st.markdown("### 🔔 Alerta Agronómica Integrada")
        st.markdown(alerta)
        st.markdown("---")
        st.markdown(f"**🌤️ Pronóstico próxima semana:** {pronostico_gfs['alerta_esta_semana']}")
        fecha_str = datetime.now().strftime('%Y%m%d_%H%M')
        texto_descarga = f"ALERTA — {cultivo} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*60}\n\n{alerta}\n\n{'='*60}\nDatos de Estación: {st.session_state['datos_estacion']}\nPronóstico semana: {pronostico_gfs['alerta_esta_semana']}\n"
        st.download_button("📥 Descargar alerta completa", data=texto_descarga, file_name=f"alerta_{cultivo}_{fecha_str}.txt")

# ============================================================
# ESTACIÓN METEOROLÓGICA
# ============================================================
with tab_estacion:
    st.header("🌦️ Datos de Estación Meteorológica")
    st.markdown("Puedes cargar los datos de tu estación de forma **manual**, **simulada automática** o **vía API externa**.")

    modo = st.radio(
        "Origen de los datos",
        ["Manual", "Automática (simulada)", "Conectar a API externa (placeholder)"],
        index=0 if st.session_state.get('modo_estacion') == "Manual" else (1 if st.session_state.get('modo_estacion') == "Automática (simulada)" else 2),
        horizontal=True
    )
    st.session_state['modo_estacion'] = modo

    # ---- Modo Manual -------------------------------------------------
    if modo == "Manual":
        st.subheader("✍️ Ingreso manual de datos")
        with st.form("form_manual_estacion"):
            col1, col2, col3 = st.columns(3)
            with col1:
                temp_ext = st.number_input(
                    "🌡️ Temperatura exterior (°C)",
                    value=float(st.session_state['datos_estacion'].get('temp_exterior', 22.0)),
                    step=0.5, format="%.1f"
                )
                hum_ext = st.number_input(
                    "💧 Humedad exterior (%)",
                    value=float(st.session_state['datos_estacion'].get('humedad_exterior', 65.0)),
                    step=1.0, format="%.1f"
                )
                rad = st.number_input(
                    "☀️ Radiación solar (W/m²)",
                    value=float(st.session_state['datos_estacion'].get('radiacion_solar', 500.0)),
                    step=10.0, format="%.0f"
                )
            with col2:
                viento = st.number_input(
                    "💨 Velocidad del viento (km/h)",
                    value=float(st.session_state['datos_estacion'].get('viento', 10.0)),
                    step=1.0, format="%.1f"
                )
                ph = st.number_input(
                    "🧪 pH del suelo",
                    value=float(st.session_state['datos_estacion'].get('ph_suelo', 6.5)),
                    step=0.1, format="%.1f"
                )
                mo = st.number_input(
                    "🌱 Materia orgánica (%)",
                    value=float(st.session_state['datos_estacion'].get('materia_organica', 2.5)),
                    step=0.1, format="%.1f"
                )
            with col3:
                n = st.number_input(
                    "Nitrógeno (mg/kg)",
                    value=float(st.session_state['datos_estacion'].get('nitrogeno', 60.0)),
                    step=5.0, format="%.0f"
                )
                p = st.number_input(
                    "Fósforo (mg/kg)",
                    value=float(st.session_state['datos_estacion'].get('fosforo', 35.0)),
                    step=5.0, format="%.0f"
                )
                k = st.number_input(
                    "Potasio (mg/kg)",
                    value=float(st.session_state['datos_estacion'].get('potasio', 120.0)),
                    step=10.0, format="%.0f"
                )
            submitted = st.form_submit_button("📥 Cargar datos manuales")
            if submitted:
                st.session_state['datos_estacion'] = {
                    'temp_exterior': temp_ext, 'humedad_exterior': hum_ext,
                    'radiacion_solar': rad, 'viento': viento,
                    'ph_suelo': ph, 'materia_organica': mo,
                    'nitrogeno': n, 'fosforo': p, 'potasio': k,
                }
                st.success("✅ Datos manuales cargados correctamente.")
                st.rerun()

        # Mostrar los datos actuales
        estacion_data = st.session_state['datos_estacion']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperatura Exterior", f"{estacion_data['temp_exterior']} °C")
            st.metric("💧 Humedad Exterior", f"{estacion_data['humedad_exterior']} %")
        with col2:
            st.metric("☀️ Radiación Solar", f"{estacion_data['radiacion_solar']} W/m²")
            st.metric("💨 Velocidad del Viento", f"{estacion_data['viento']} km/h")
        with col3:
            st.metric("🧪 pH del Suelo", f"{estacion_data['ph_suelo']}")
            st.metric("🌱 Materia Orgánica", f"{estacion_data['materia_organica']} %")
        st.subheader("🧪 Fertilidad del Suelo")
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            st.metric("Nitrógeno (N)", f"{estacion_data['nitrogeno']} mg/kg")
        with col_p:
            st.metric("Fósforo (P)", f"{estacion_data['fosforo']} mg/kg")
        with col_k:
            st.metric("Potasio (K)", f"{estacion_data['potasio']} mg/kg")

    # ---- Modo Automática (simulada) ---------------------------------
    elif modo == "Automática (simulada)":
        st.subheader("🔄 Datos simulados (estación virtual)")
        if st.button("🔄 Generar nueva simulación"):
            st.session_state['datos_estacion'] = obtener_datos_estacion_simulada()
            st.rerun()
        estacion_data = st.session_state['datos_estacion']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperatura Exterior", f"{estacion_data['temp_exterior']} °C")
            st.metric("💧 Humedad Exterior", f"{estacion_data['humedad_exterior']} %")
        with col2:
            st.metric("☀️ Radiación Solar", f"{estacion_data['radiacion_solar']} W/m²")
            st.metric("💨 Velocidad del Viento", f"{estacion_data['viento']} km/h")
        with col3:
            st.metric("🧪 pH del Suelo", f"{estacion_data['ph_suelo']}")
            st.metric("🌱 Materia Orgánica", f"{estacion_data['materia_organica']} %")
        st.subheader("🧪 Fertilidad del Suelo")
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            st.metric("Nitrógeno (N)", f"{estacion_data['nitrogeno']} mg/kg")
        with col_p:
            st.metric("Fósforo (P)", f"{estacion_data['fosforo']} mg/kg")
        with col_k:
            st.metric("Potasio (K)", f"{estacion_data['potasio']} mg/kg")
        st.info("ℹ️ Estos datos se generan aleatoriamente cada vez que presionas el botón. Representan valores típicos para la costa peruana.")

    # ---- Modo API externa (placeholder) ----------------------------
    else:
        st.subheader("🔌 Conexión a estación automática real (vía API)")
        st.markdown("Configura la URL y la clave de acceso para obtener datos en tiempo real.")
        with st.form("form_api_estacion"):
            api_url = st.text_input("URL de la API (ej: https://tu-estacion.com/api/actual)", value=st.session_state.get('api_url', ''))
            api_key = st.text_input("Clave API (opcional)", type="password", value=st.session_state.get('api_key', ''))
            submitted_api = st.form_submit_button("🌐 Conectar y obtener datos")
            if submitted_api:
                st.session_state['api_url'] = api_url
                st.session_state['api_key'] = api_key
                try:
                    # Aquí iría la lógica real de consulta a la API.
                    # Por ahora simulamos una respuesta.
                    import random
                    st.session_state['datos_estacion'] = {
                        'temp_exterior': round(random.uniform(18, 28), 1),
                        'humedad_exterior': round(random.uniform(50, 85), 1),
                        'radiacion_solar': round(random.uniform(300, 900), 0),
                        'viento': round(random.uniform(2, 18), 1),
                        'ph_suelo': round(random.uniform(6.0, 7.2), 1),
                        'materia_organica': round(random.uniform(2.0, 3.8), 1),
                        'nitrogeno': round(random.uniform(40, 90), 0),
                        'fosforo': round(random.uniform(20, 50), 0),
                        'potasio': round(random.uniform(80, 160), 0),
                    }
                    st.success("✅ Datos obtenidos correctamente (simulación). Reemplazar con API real.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error conectando a la API: {e}")

        if 'datos_estacion' in st.session_state:
            estacion_data = st.session_state['datos_estacion']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌡️ Temperatura Exterior", f"{estacion_data['temp_exterior']} °C")
                st.metric("💧 Humedad Exterior", f"{estacion_data['humedad_exterior']} %")
            with col2:
                st.metric("☀️ Radiación Solar", f"{estacion_data['radiacion_solar']} W/m²")
                st.metric("💨 Velocidad del Viento", f"{estacion_data['viento']} km/h")
            with col3:
                st.metric("🧪 pH del Suelo", f"{estacion_data['ph_suelo']}")
                st.metric("🌱 Materia Orgánica", f"{estacion_data['materia_organica']} %")
            st.subheader("🧪 Fertilidad del Suelo")
            col_n, col_p, col_k = st.columns(3)
            with col_n:
                st.metric("Nitrógeno (N)", f"{estacion_data['nitrogeno']} mg/kg")
            with col_p:
                st.metric("Fósforo (P)", f"{estacion_data['fosforo']} mg/kg")
            with col_k:
                st.metric("Potasio (K)", f"{estacion_data['potasio']} mg/kg")

    # Interpretación común y exportación
    st.markdown("---")
    st.subheader("📈 Interpretación de Datos")
    st.markdown("""
    - **Temperatura y Humedad:** Optimizar ventilación y riego según valores.
    - **Radiación Solar:** Ajustar sombreado para evitar estrés térmico.
    - **pH del Suelo:** Rango ideal 5.5-7.0 para la mayoría de hortalizas.
    - **Materia Orgánica:** Mejorar la estructura del suelo y retención de agua.
    - **Fertilidad:** Ajustar fertilización según deficiencias de N-P-K.
    """)
    if st.button("📥 Exportar Datos de Estación"):
        df_estacion = pd.DataFrame([st.session_state['datos_estacion']])
        st.download_button("⬇️ Descargar CSV", data=df_estacion.to_csv(index=False), file_name=f"estacion_meteorologica_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

# ============================================================
# EXPORTAR
# ============================================================
with tab_export:
    st.subheader("💾 Exportar Datos")
    if st.button("Exportar parcela a GeoJSON"):
        st.download_button("⬇️ Descargar GeoJSON", data=gdf.to_json(), file_name="parcela.geojson")
    if not df_ndvi.empty:
        st.download_button("⬇️ Serie NDVI CSV", data=df_ndvi.to_csv(index=False), file_name="ndvi.csv")
    resumen = f"RESUMEN — {cultivo} — {datetime.now().strftime('%Y-%m-%d')}\nNDVI: {ndvi_val:.2f}\nTemperatura: {temp_val:.1f}°C\nHumedad: {humedad_val:.2f}\nPrecipitación: {precip_actual:.1f} mm\nÁrea: {area_ha:.2f} ha\nPronóstico semana: {pronostico_gfs['alerta_esta_semana']}\nDatos de Estación: {st.session_state['datos_estacion']}\n"
    st.download_button("⬇️ Resumen TXT", data=resumen, file_name="resumen.txt")
    st.markdown("---")
    st.subheader("📦 Exportar para biomod2 (R)")
    if st.button("🔬 Generar archivo biomod2"):
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        step = 0.001
        points = []
        for x in np.arange(minx, maxx, step):
            for y in np.arange(miny, maxy, step):
                pt = Point(x, y)
                if gdf.geometry.iloc[0].contains(pt):
                    points.append([x, y])
        if not points:
            st.error("No se generaron puntos internos. Reducí el paso (step).")
        else:
            df_points = pd.DataFrame(points, columns=['longitud', 'latitud'])
            df_points['NDVI'] = ndvi_val
            df_points['temperatura_C'] = temp_val
            df_points['precipitacion_mm'] = precip_actual
            df_points['humedad_suelo'] = humedad_val
            df_points['rendimiento_t_ha'] = predecir_rendimiento(ndvi_val, precip_actual, temp_val)
            umbral_ndvi = UMBRALES[cultivo]['NDVI_min']
            df_points['Presence'] = (df_points['NDVI'] >= umbral_ndvi).astype(int)
            st.download_button("📥 Descargar CSV para biomod2", data=df_points.to_csv(index=False), file_name=f"biomod2_{cultivo}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            st.success(f"✅ {len(df_points)} puntos generados dentro de la parcela.")

# ============================================================
# DEM (RELIEVE)
# ============================================================
with tab_dem:
    st.header("🗻 Análisis de Relieve — OpenTopography")
    if not OPENTOPOGRAPHY_AVAILABLE:
        st.error("❌ requests no disponible — módulo DEM inactivo.")
    else:
        _ot_key = OPENTOPOGRAPHY_API_KEY
        if not _ot_key:
            _ot_key = st.session_state.get("ot_api_key_manual", "")
            _key_input = st.text_input("🔑 API Key de OpenTopography", value=_ot_key, type="password", help="Conseguila gratis en https://opentopography.org/developers")
            if _key_input:
                st.session_state["ot_api_key_manual"] = _key_input
                _ot_key = _key_input
            if not _ot_key:
                st.info("Ingresá tu API key de OpenTopography para descargar el DEM.")
                st.stop()
        bounds_dem = gdf.total_bounds
        col_ds, col_btn = st.columns([3, 1])
        with col_ds:
            resolucion = st.selectbox("Resolución del DEM", list(_DATASETS_DEM.keys()))
        with col_btn:
            st.write("")
            cargar_dem = st.button("📥 Cargar DEM", type="primary")
        if cargar_dem:
            dataset_sel = _DATASETS_DEM[resolucion]
            with st.spinner(f"Descargando DEM {resolucion} desde OpenTopography..."):
                dem = obtener_dem_opentopography(bounds_dem, _ot_key, dem_type=dataset_sel)
            if dem is not None:
                st.session_state["dem_data"] = dem
                st.session_state["dem_dataset"] = resolucion
                elevation_mean = float(np.nanmean(dem.values))
                st.success(f"✅ DEM cargado · Elevación media: **{elevation_mean:.0f} m**")
        if st.session_state.get("dem_data") is not None:
            dem = st.session_state["dem_data"]
            ds_label = st.session_state.get("dem_dataset", resolucion)
            elev_min = float(np.nanmin(dem.values))
            elev_max = float(np.nanmax(dem.values))
            elev_mean = float(np.nanmean(dem.values))
            elev_range = elev_max - elev_min
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⬇️ Elevación mínima", f"{elev_min:.0f} m")
            c2.metric("⬆️ Elevación máxima", f"{elev_max:.0f} m")
            c3.metric("📏 Elevación media", f"{elev_mean:.0f} m")
            c4.metric("↕️ Rango altitudinal", f"{elev_range:.0f} m")
            st.markdown("---")
            tipo_vis = st.radio("Visualización", ["🗺️ Mapa 2D interactivo", "📐 Modelo 3D interactivo"], horizontal=True)
            if tipo_vis == "🗺️ Mapa 2D interactivo" and FOLIUM_OK:
                with st.spinner("Generando mapa 2D con overlay DEM..."):
                    mapa_dem = generar_mapa_folium_dem(gdf, dem, ds_label)
                if FOLIUM_STATIC_OK:
                    folium_static(mapa_dem, width=900, height=620)
                else:
                    components.html(mapa_dem.get_root().render(), height=620)
            elif tipo_vis == "📐 Modelo 3D interactivo":
                with st.spinner("Generando modelo 3D de elevación..."):
                    fig_3d, _mn, _mx, _me = generar_grafico_3d_dem(dem)
                if fig_3d is not None:
                    st.plotly_chart(fig_3d, use_container_width=True)
            with st.expander("📈 Ver perfil de elevación"):
                if XARRAY_OK:
                    try:
                        arr_2d = dem.values.squeeze() if dem.values.ndim > 2 else dem.values
                        fila_central = arr_2d[arr_2d.shape[0] // 2, :]
                        x_lon = dem.x.values
                        fig_p, ax_p = plt.subplots(figsize=(9, 3))
                        ax_p.fill_between(x_lon, fila_central, alpha=0.4, color="saddlebrown")
                        ax_p.plot(x_lon, fila_central, color="saddlebrown", linewidth=1.5)
                        ax_p.axhline(elev_mean, color="blue", linestyle="--", linewidth=1)
                        ax_p.set_xlabel("Longitud"); ax_p.set_ylabel("Elevación (m)")
                        ax_p.set_title("Perfil de elevación")
                        plt.tight_layout(); st.pyplot(fig_p)
                    except Exception as e:
                        st.warning(f"No se pudo generar el perfil: {e}")
            with st.expander("💾 Exportar datos de elevación"):
                try:
                    arr_flat = dem.values.squeeze().flatten()
                    lons_flat = np.tile(dem.x.values, len(dem.y.values))
                    lats_flat = np.repeat(dem.y.values, len(dem.x.values))
                    df_dem = pd.DataFrame({"latitud": lats_flat, "longitud": lons_flat, "elevacion_m": arr_flat}).dropna()
                    st.dataframe(df_dem.head(200))
                    st.download_button("⬇️ Descargar DEM completo (CSV)", data=df_dem.to_csv(index=False), file_name=f"dem_{ds_label.replace(' ','_')}.csv")
                except Exception as e:
                    st.warning(f"Error exportando DEM: {e}")

# ============================================================
# FERTILIDAD NPK POR BLOQUES (con almacenamiento en session_state)
# ============================================================
with tab_npk:
    st.header("🌾 Fertilidad NPK por Bloques")
    if st.button("🔬 Calcular fertilidad por bloque", type="primary"):
        with st.spinner(f"Dividiendo en {n_bloques} bloques y consultando GEE…"):
            gdf_bloques = dividir_parcela_en_bloques(gdf, n_bloques)
            if gdf_bloques is None or len(gdf_bloques) == 0:
                st.error("No se pudo dividir la parcela.")
            else:
                ndvis = obtener_ndvi_por_bloque(gdf_bloques, fecha_fin)
                gdf_bloques['ndvi'] = ndvis
                areas_bloque = []
                for _, row in gdf_bloques.iterrows():
                    a = calcular_superficie(gpd.GeoDataFrame({'geometry': [row.geometry]}, crs='EPSG:4326'))
                    areas_bloque.append(a)
                gdf_bloques['area_ha'] = areas_bloque
                recs = [calcular_recomendaciones_npk(v, cultivo) for v in gdf_bloques['ndvi']]
                gdf_bloques['nivel'] = [r['nivel'] for r in recs]
                gdf_bloques['N_kg_ha'] = [r['N'] for r in recs]
                gdf_bloques['P_kg_ha'] = [r['P'] for r in recs]
                gdf_bloques['K_kg_ha'] = [r['K'] for r in recs]
                rends = [estimar_potencial_cosecha(v, cultivo, a) for v, a in zip(gdf_bloques['ndvi'], gdf_bloques['area_ha'])]
                gdf_bloques['rend_t_ha'] = [r[0] for r in rends]
                gdf_bloques['prod_total_t'] = [r[1] for r in rends]
                # Guardar en session_state
                st.session_state['gdf_bloques'] = gdf_bloques
                st.session_state['bloques_calculados'] = True
                c1, c2, c3, c4 = st.columns(4)
                ndvi_med = gdf_bloques['ndvi'].mean()
                c1.metric("NDVI promedio", f"{ndvi_med:.3f}")
                c2.metric("Bloques críticos", str((gdf_bloques['N_kg_ha'] > 40).sum()))
                c3.metric("Rend. medio (t/ha)", f"{gdf_bloques['rend_t_ha'].mean():.1f}")
                c4.metric("Producción total", f"{gdf_bloques['prod_total_t'].sum():.1f} t")
                st.subheader("📋 Detalle por bloque")
                display_cols = ['id_bloque','area_ha','ndvi','nivel','N_kg_ha','P_kg_ha','K_kg_ha','rend_t_ha','prod_total_t']
                st.dataframe(gdf_bloques[display_cols].round(3), use_container_width=True)
                if FOLIUM_OK:
                    st.subheader("🗺️ Mapa de calor NDVI por bloque")
                    bounds_b = gdf_bloques.total_bounds
                    c_lat, c_lon, z = obtener_zoom_con_margen(bounds_b)
                    m_npk = folium.Map(location=[c_lat, c_lon], zoom_start=z, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid')
                    vmin = gdf_bloques['ndvi'].min()
                    vmax = gdf_bloques['ndvi'].max()
                    cmap_fn = plt.cm.RdYlGn
                    for _, row in gdf_bloques.iterrows():
                        ndvi_b = row['ndvi'] if not np.isnan(row['ndvi']) else 0.3
                        norm_v = (ndvi_b - vmin) / max(vmax - vmin, 0.01)
                        r_, g_, b_, _ = cmap_fn(norm_v)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r_*255), int(g_*255), int(b_*255))
                        popup_txt = f"Bloque {int(row.id_bloque)}<br>NDVI: {ndvi_b:.3f}<br>N: {int(row.N_kg_ha)} kg/ha · P: {int(row.P_kg_ha)} · K: {int(row.K_kg_ha)}<br>Rend.: {row.rend_t_ha:.1f} t/ha"
                        folium.GeoJson(gpd.GeoDataFrame({'geometry': [row.geometry]}, crs='EPSG:4326').__geo_interface__,
                                       style_function=lambda x, c=hex_color: {'fillColor': c, 'color': '#333', 'weight': 1, 'fillOpacity': 0.7},
                                       tooltip=f"Bloque {int(row.id_bloque)} · NDVI {ndvi_b:.3f}",
                                       popup=folium.Popup(popup_txt, max_width=200)).add_to(m_npk)
                    components.html(m_npk.get_root().render(), height=500)
                st.download_button("⬇️ Descargar CSV fertilidad", data=gdf_bloques[display_cols].to_csv(index=False), file_name=f"fertilidad_npk_{cultivo}.csv", mime="text/csv")
    else:
        if st.session_state.get('bloques_calculados', False) and st.session_state.get('gdf_bloques') is not None:
            st.info("ℹ️ Ya se calcularon los bloques anteriormente. Puedes volver a calcular si cambiaste parámetros.")
        else:
            st.info("🔘 Haz clic en 'Calcular fertilidad por bloque' para iniciar el análisis.")

# ============================================================
# AGROECOLOGÍA — 10 PRINCIPIOS (guardamos textos)
# ============================================================
with tab_agro:
    st.header("🌱 Agroecología — 10 Principios")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌿 Recomendación por principio", type="primary"):
            with st.spinner("Generando recomendaciones agroecológicas…"):
                rec = generar_recomendaciones_agroecologicas(cultivo, fase_fenologica, ndvi_val, temp_val, humedad_val, precip_actual)
                st.session_state['ultimo_texto_agro'] = rec
            st.markdown("### 🌿 Recomendaciones por Principio Agroecológico")
            st.markdown(rec)
            st.download_button("⬇️ Descargar recomendaciones", data=rec, file_name=f"agroecologia_principios_{cultivo}.txt")
    with col_b:
        if st.button("📋 Plan agroecológico completo"):
            with st.spinner("Generando plan completo…"):
                plan = generar_plan_agroecologico_completo(cultivo, fase_fenologica, ndvi_val, temp_val, humedad_val, precip_actual, area_ha)
                st.session_state['ultimo_plan_agro'] = plan
            st.markdown("### 📋 Plan Agroecológico Integral")
            st.markdown(plan)
            st.download_button("⬇️ Descargar plan", data=plan, file_name=f"plan_agroecologico_{cultivo}.txt")
    st.caption(f"📊 Contexto enviado a la IA — NDVI: {ndvi_val:.3f} · Temp: {temp_val:.1f}°C · Humedad: {humedad_val:.2f} · Precip: {precip_actual:.1f} mm · Fase: {fase_fenologica} · Cultivo: {cultivo}")

# ============================================================
# CARBONO Y CRÉDITOS (con almacenamiento)
# ============================================================
with tab_carbono:
    st.header("🌍 Carbono y Créditos de Carbono")
    calc_c = CalculadorCarbono()
    precip_anual = estimar_precipitacion_anual(df_precip)
    res_c = calc_c.calcular_carbono_hectarea(ndvi_val, precip_anual)
    co2_total = round(res_c['co2_equivalente_ton_ha'] * area_ha, 2)
    creditos = round(co2_total / 1000, 4)
    precio_usd = round(creditos * 15, 2)
    # Guardar en sesión
    st.session_state['res_carbono'] = res_c
    st.session_state['co2_total'] = co2_total
    st.session_state['creditos'] = creditos
    st.session_state['precio_usd'] = precio_usd
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌿 C total (t C/ha)", f"{res_c['carbono_total_ton_ha']}")
    c2.metric("☁️ CO₂e (t/ha)", f"{res_c['co2_equivalente_ton_ha']}")
    c3.metric("📐 Área", f"{area_ha:.2f} ha")
    c4.metric("🪙 Créditos (kt CO₂e)", f"{creditos:.4f}")
    c5.metric("💵 Valor estimado USD", f"${precio_usd:,.2f}")
    st.markdown("---")
    st.subheader("📊 Desglose por pool de carbono")
    df_pools = pd.DataFrame(list(res_c['desglose'].items()), columns=['Pool de carbono', 't C/ha'])
    st.dataframe(df_pools, use_container_width=True)
    fig_c, ax_c = plt.subplots(figsize=(8, 3))
    bars = ax_c.barh(df_pools['Pool de carbono'], df_pools['t C/ha'], color=['#2ecc71','#27ae60','#f39c12','#e67e22','#8e44ad'])
    ax_c.set_xlabel('t C/ha')
    ax_c.set_title(f'Distribución de carbono — {cultivo} · {area_ha:.2f} ha')
    for bar, val in zip(bars, df_pools['t C/ha']):
        ax_c.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig_c)
    st.info(f"💡 Precipitación anual estimada: **{precip_anual:.0f} mm/año** · Precio de referencia: **15 USD/t CO₂e** (mercado voluntario).")
    st.download_button("⬇️ Exportar reporte de carbono CSV", data=pd.DataFrame([{'cultivo': cultivo, 'area_ha': area_ha, 'ndvi': ndvi_val, 'precip_anual_mm': precip_anual, **res_c['desglose'], 'carbono_total_ton_ha': res_c['carbono_total_ton_ha'], 'co2e_ton_ha': res_c['co2_equivalente_ton_ha'], 'co2e_total_parcela': co2_total, 'creditos_kton': creditos, 'valor_usd': precio_usd}]).to_csv(index=False), file_name=f"carbono_{cultivo}_{area_ha:.1f}ha.csv", mime="text/csv")

# ============================================================
# NUEVA PESTAÑA: GOBERNANZA (Resumen + Podcast con gTTS, gratis y sin errores)
# ============================================================
with tab_gobernanza:
    st.header("🎙️ Gobernanza – Podcast Inteligente (gTTS, 100% gratuito)")
    st.markdown("Genera un **podcast automático** con recomendaciones de IA. El audio usa **gTTS** (Google Text‑to‑Speech), que es gratuito, no requiere API key y funciona sin errores 403.")

    # ------------------------------------------------------------------
    # 1. Resumen visual (métricas clave y riesgos)
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌱 NDVI actual", f"{ndvi_val:.2f}")
    col2.metric("🌡️ Temperatura", f"{temp_val:.1f} °C")
    col3.metric("💧 Humedad suelo", f"{humedad_val:.2f}")
    col4.metric("🌧️ Precipitación", f"{precip_actual:.1f} mm")

    st.markdown("---")
    st.subheader("📌 Indicadores de riesgo y producción")

    riesgo_ndvi, _ = determinar_riesgo("NDVI", ndvi_val, cultivo, UMBRALES[cultivo])
    riesgo_temp = "BAJO" if umbral['temp_min'] <= temp_val <= umbral['temp_max'] else "ALTO"
    riesgo_hum = "BAJO" if umbral['humedad_min'] <= humedad_val <= umbral['humedad_max'] else "ALTO"

    c1, c2, c3 = st.columns(3)
    c1.info(f"**Riesgo NDVI:** {riesgo_ndvi}")
    c2.info(f"**Riesgo temperatura:** {riesgo_temp}")
    c3.info(f"**Riesgo humedad:** {riesgo_hum}")

    st.markdown(f"**Pronóstico semanal:** {pronostico_gfs['alerta_esta_semana']}")

    # ------------------------------------------------------------------
    # 2. Generación de podcast con gTTS (gratis, estable)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎙️ Generar Podcast Inteligente")

    if st.button("🚀 Generar Podcast (Español)", type="primary"):
        if not GROQ_API_KEY or not GROQ_AVAILABLE:
            st.error("❌ API Key de Groq no configurada. El podcast no puede generarse.")
        elif not GTTS_OK:
            st.error("❌ La librería 'gTTS' no está instalada. Ejecuta: pip install gTTS")
        else:
            with st.spinner("IA trabajando: Creando guion y generando audio con gTTS..."):
                try:
                    # 1. Generar guion con Groq
                    prompt_podcast = f"""
                    Eres un agroecólogo y narrador de podcasts. Convierte el siguiente resumen técnico en un guion para un podcast dinámico y accesible en español.

                    El podcast debe:
                    - Ser presentado por un solo narrador (voz amable y clara).
                    - Comenzar con un saludo y presentación.
                    - Explicar los datos clave de forma sencilla para un agricultor.
                    - Terminar con 3 recomendaciones prácticas y concretas.
                    - Duración aproximada: 90 segundos de lectura.

                    **Datos del cultivo:**
                    Cultivo: {cultivo} ({area_ha:.2f} hectáreas)
                    NDVI (salud del cultivo): {ndvi_val:.2f} (Óptimo > {umbral['NDVI_min']:.2f})
                    Temperatura actual: {temp_val:.1f}°C (Rango ideal {umbral['temp_min']}°C - {umbral['temp_max']}°C)
                    Humedad del suelo: {humedad_val:.2f} (Rango ideal {umbral['humedad_min']:.2f} - {umbral['humedad_max']:.2f})
                    Riesgos: NDVI {riesgo_ndvi}, Temperatura {riesgo_temp}, Humedad {riesgo_hum}.
                    Pronóstico semanal: {pronostico_gfs['alerta_esta_semana']}.

                    Recomendaciones de IA para el cultivo:
                    {st.session_state.get('ultima_alerta_texto', 'Mantener riego controlado y monitorear temperatura.')}
                    """

                    client_groq = Groq(api_key=GROQ_API_KEY)
                    response = client_groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt_podcast}],
                        max_tokens=600,
                        temperature=0.6,
                    )
                    podcast_script = response.choices[0].message.content

                    # 2. Generar audio con gTTS
                    tts = gTTS(text=podcast_script, lang='es', slow=False)
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)

                    st.success("✅ Podcast generado exitosamente con gTTS")
                    st.audio(audio_bytes, format='audio/mp3')
                    st.download_button(
                        label="📥 Descargar Podcast (MP3)",
                        data=audio_bytes,
                        file_name=f"podcast_{cultivo}_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3",
                        mime="audio/mpeg"
                    )
                    st.session_state['ultimo_guion_podcast'] = podcast_script

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Asegúrate de tener conexión a internet y que gTTS esté correctamente instalado.")

    # Mostrar guion generado
    if st.session_state.get('ultimo_guion_podcast'):
        with st.expander("📄 Ver el guion del podcast generado"):
            st.markdown(st.session_state['ultimo_guion_podcast'])

    # ------------------------------------------------------------------
    # 3. Descarga del resumen en guaraní (texto)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📄 Descargar resumen en guaraní (texto)")
    st.markdown("La síntesis de voz en guaraní no está disponible gratuitamente de forma estable. Aquí puedes descargar el texto traducido automáticamente para leerlo o usar otro servicio.")

    def generar_texto_resumen_gn():
        texto_es = f"""
Resumen ejecutivo del monitoreo de {cultivo} en una parcela de {area_ha:.2f} hectáreas.
Fecha: {datetime.now().strftime('%d/%m/%Y')}.

Indicadores:
- NDVI: {ndvi_val:.2f} (óptimo > {umbral['NDVI_min']:.2f})
- Temperatura: {temp_val:.1f}°C (rango ideal {umbral['temp_min']}-{umbral['temp_max']}°C)
- Humedad suelo: {humedad_val:.2f} (rango ideal {umbral['humedad_min']:.2f}-{umbral['humedad_max']:.2f})
- Precipitación: {precip_actual:.1f} mm

Riesgos: NDVI {riesgo_ndvi}, Temp {riesgo_temp}, Humedad {riesgo_hum}
Pronóstico: {pronostico_gfs['alerta_esta_semana']}

Recomendaciones IA: {st.session_state.get('ultima_alerta_texto', 'Sin recomendaciones aún.')}
        """
        # Traducción simbólica (solo para mostrar)
        traducciones = {
            "Resumen ejecutivo del monitoreo de": "Tembi'ukuaa ñangareko",
            "en una parcela de": "yvyra peteĩ",
            "hectáreas": "hectárea",
            "Indicadores": "Indicadores",
            "óptimo": "porã",
            "rango ideal": "tembiapo porã",
            "Riesgos": "Verea",
            "Pronóstico": "Ama ñe'ẽ",
            "Recomendaciones": "Ñe'ẽme'ẽ",
        }
        texto_gn = texto_es
        for es, gn in traducciones.items():
            texto_gn = texto_gn.replace(es, gn)
        return f"[Traducción automática al guaraní - revisar con hablante nativo]\n\n{texto_gn}"

    if st.button("📄 Descargar resumen en guaraní (texto)"):
        texto_gn = generar_texto_resumen_gn()
        st.download_button(
            label="⬇️ Descargar texto en guaraní (TXT)",
            data=texto_gn,
            file_name=f"resumen_guarani_{cultivo}.txt",
            mime="text/plain"
        )
        st.success("Texto en guaraní listo para descargar.")

st.caption("Plataforma de Monitoreo de Hortalizas bajo Invernadero · Datos de estación meteorológica (manual / simulada / API) · Sentinel-2 · ERA5 · CHIRPS · GFS · Podcast gratuito con gTTS")
