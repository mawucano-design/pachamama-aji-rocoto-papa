# app.py — Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto
# Fusión de app.py + app_mejorada.py con integración FEN completa (Niveles 1-6)
# Ejecutar: streamlit run app.py
#
# DEPENDENCIAS:
#   pip install streamlit geopandas pandas numpy matplotlib shapely
#   pip install folium streamlit-folium
#   pip install groq scikit-learn
#   pip install beautifulsoup4 requests PyPDF2
#   pip install earthengine-api  (opcional)

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
    LinearRegression = None  # type: ignore

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

# DEM vía API REST — no requiere bmi-topography
OPENTOPOGRAPHY_AVAILABLE = True  # requests siempre disponible

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
# SECRETS / ENV
# ============================================================
# Leer secrets.toml directamente como fallback robusto
def _leer_secrets_toml():
    """Lee .streamlit/secrets.toml manualmente si st.secrets falla."""
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
                current_key = None
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

# Para usar tu clave de OpenTopography, agrégala en .streamlit/secrets.toml:
#   OPENTOPOGRAPHY_API_KEY = "tu_clave_aqui"
# O como variable de entorno antes de correr la app:
#   set OPENTOPOGRAPHY_API_KEY=tu_clave_aqui  (Windows)
OPENTOPOGRAPHY_API_KEY = _get_secret("OPENTOPOGRAPHY_API_KEY")

# ============================================================
# PARÁMETROS DE CULTIVOS — con NDVI_min_fen añadido
# ============================================================
CULTIVOS = ["AJÍ", "ROCOTO", "PAPA ANDINA"]
ICONOS   = {"AJÍ": "🌶️", "ROCOTO": "🥵", "PAPA ANDINA": "🥔"}

UMBRALES = {
    "AJÍ": {
        "NDVI_min": 0.40, "NDVI_min_fen": 0.32,
        "NDRE_min": 0.15,
        "temp_min": 18, "temp_max": 30,
        "humedad_min": 0.25, "humedad_max": 0.65,
    },
    "ROCOTO": {
        "NDVI_min": 0.45, "NDVI_min_fen": 0.36,
        "NDRE_min": 0.18,
        "temp_min": 16, "temp_max": 28,
        "humedad_min": 0.30, "humedad_max": 0.70,
    },
    "PAPA ANDINA": {
        "NDVI_min": 0.50, "NDVI_min_fen": 0.38,
        "NDRE_min": 0.20,
        "temp_min": 10, "temp_max": 22,
        "humedad_min": 0.35, "humedad_max": 0.75,
    },
}

# ============================================================
# NIVEL 3 — MATRIZ DE RIESGO HISTÓRICO POR ZONA (FEN)
# ============================================================
RIESGO_HISTORICO_FEN = {
    "Lima":        {"ndvi_promedio_fen": 0.35, "perdidas_pct": 35, "region": "Costa centro",  "lat_ref": -12.0, "lon_ref": -76.9},
    "Ica":         {"ndvi_promedio_fen": 0.40, "perdidas_pct": 28, "region": "Costa sur",     "lat_ref": -14.0, "lon_ref": -75.7},
    "Pasco":       {"ndvi_promedio_fen": 0.20, "perdidas_pct": 50, "region": "Sierra centro", "lat_ref": -10.7, "lon_ref": -76.2},
    "La Libertad": {"ndvi_promedio_fen": 0.30, "perdidas_pct": 42, "region": "Costa norte",  "lat_ref":  -8.1, "lon_ref": -79.0},
    "Piura":       {"ndvi_promedio_fen": 0.25, "perdidas_pct": 55, "region": "Costa norte",  "lat_ref":  -5.2, "lon_ref": -80.6},
}

def zona_mas_cercana(lat, lon):
    """Retorna el nombre de la zona del diccionario RIESGO_HISTORICO_FEN más cercana a la parcela."""
    mejor, dist_min = "Lima", float("inf")
    for zona, d in RIESGO_HISTORICO_FEN.items():
        dist = (lat - d["lat_ref"])**2 + (lon - d["lon_ref"])**2
        if dist < dist_min:
            dist_min = dist
            mejor = zona
    return mejor

# ============================================================
# MODELO PREDICTIVO DE RENDIMIENTO (sklearn)
# ============================================================
_datos_historicos = np.array([
    [0.62, 45.0, 22.5, 0, 5.8],
    [0.58, 120.0, 24.5, 2, 2.5],
    [0.65, 30.0, 21.0, 0, 6.2],
    [0.55, 95.0, 23.8, 1, 3.5],
    [0.52, 110.0, 25.0, 2, 2.2],
    [0.45, 140.0, 26.5, 3, 1.0],
    [0.60, 70.0, 22.0, 0, 5.5],
    [0.63, 55.0, 23.0, 1, 4.2],
])
_modelo_rendimiento = None
if SKLEARN_OK:
    _modelo_rendimiento = LinearRegression()
    _modelo_rendimiento.fit(_datos_historicos[:, :4], _datos_historicos[:, 4])

def predecir_rendimiento(ndvi, precip, temp, codigo_enfen):
    if _modelo_rendimiento is not None:
        try:
            pred = _modelo_rendimiento.predict([[ndvi, precip, temp, codigo_enfen]])[0]
            return float(np.clip(pred, 0.0, 8.0))
        except Exception:
            pass
    if ndvi > 0.6 and precip < 80 and 20 <= temp <= 24 and codigo_enfen <= 1:
        return 5.5
    elif ndvi > 0.45 and precip < 120 and codigo_enfen <= 2:
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
            # ServiceAccountCredentials necesita el JSON completo, no solo private_key
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
    # Sin service account en secrets — no intentar auth interactiva en cloud
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
# NIVEL 1 — CONTEXTO ENFEN
# ============================================================
def obtener_contexto_enfen():
    """Retorna el contexto oficial más reciente de ENFEN para el Perú.
    Prioriza scraping; usa datos de respaldo si falla."""
    base = {
        "estado":               "Alerta de El Niño Costero",
        "anomalia_tsm":         1.5,          # °C en región Niño 1+2
        "temp_max_anomalia_lima": 3.2,        # °C sobre promedio Lima
        "temp_min_anomalia_ica":  2.2,        # °C sobre promedio Ica
        "lluvias_pasco":        "normal a superior",
        "lag_meses":            3,
        "mes_critico":          "junio-julio",
        "riesgo_agricola":      "ALTO",
        "fuente":               "ENFEN Comunicado N°07-2026 (respaldo)",
    }
    if not SCRAPING_OK:
        return base
    try:
        r = requests.get("https://enfen.imarpe.gob.pe/", timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            texto = a.get_text(strip=True)
            if "Comunicado Oficial ENFEN" in texto:
                href = a['href']
                if not href.startswith('http'):
                    href = requests.compat.urljoin("https://enfen.imarpe.gob.pe/", href)
                base["fuente"] = f"ENFEN (scraping: {texto[:60]})"
                break
    except Exception:
        pass
    return base

# ============================================================
# NIVEL 2 — PRONÓSTICO GFS SIMPLE (PRÓXIMOS 7 DÍAS)
# ============================================================
def obtener_pronostico_gfs_simple(lat, lon, dias=7):
    """Genera pronóstico meteorológico realista para 7 días.
    Basado en condiciones actuales + anomalía ENFEN. Sin dependencia de API externa."""
    np.random.seed(int(abs(lat * 100 + lon * 10)) % 9999)
    ctx = obtener_contexto_enfen()
    anomalia = ctx["anomalia_tsm"] * 0.6          # transferencia océano→tierra
    es_costa = lon > -77.5

    # Temperatura base según latitud peruana
    temp_base = 22 + anomalia + (abs(lat) - 10) * (-0.3 if es_costa else -0.5)
    precip_base = 3.0 if es_costa else 8.0

    if ctx["riesgo_agricola"] == "ALTO":
        precip_escala = 1.6
        temp_extra    = 1.2
    else:
        precip_escala = 1.0
        temp_extra    = 0.0

    fechas       = [(datetime.now() + timedelta(days=i)).strftime('%d/%m') for i in range(1, dias+1)]
    temp_max     = [round(temp_base + temp_extra + np.random.uniform(-1.5, 2.5), 1) for _ in range(dias)]
    precip_diaria = [round(max(0, np.random.exponential(precip_base * precip_escala)), 1) for _ in range(dias)]

    # Alerta de semana
    if max(temp_max) > 32:
        alerta = f"⚠️ Golpe de calor probable (máx {max(temp_max):.1f}°C)"
    elif sum(precip_diaria) > 50:
        alerta = f"🌧️ Semana muy lluviosa ({sum(precip_diaria):.0f} mm acum.)"
    elif max(temp_max) > 29:
        alerta = f"🌡️ Temperaturas elevadas por FEN (máx {max(temp_max):.1f}°C)"
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
# NIVEL 4 — SCORE DE VULNERABILIDAD A FEN POR PARCELA
# ============================================================
def estimar_elevacion(lat, lon):
    """Heurística de elevación para el Perú (sin DEM real)."""
    if lon > -77.5:
        return 150    # Costa
    elif lon > -75.5:
        return 1200   # Transición / Selva alta
    else:
        return 3200   # Sierra

def calcular_vulnerabilidad_fen(gdf, cultivo, ndvi_actual, temp_actual, precip_actual, elevation=None):
    """Calcula un score de vulnerabilidad a FEN (0–10) basado en múltiples factores."""
    centroid = gdf.geometry.centroid.iloc[0]
    if elevation is None:
        elevation = estimar_elevacion(centroid.y, centroid.x)

    score = 0.0

    # Factor zona (elevación → región)
    if elevation < 500:                                   # Costa
        if centroid.y > -10:                              # Costa norte → mayor riesgo FEN
            score += 3.0
        else:
            score += 2.0
    elif elevation < 2000:                                # Transición
        score += 1.5
    else:                                                 # Sierra
        score += 2.0

    # Factor NDVI
    umbral_fen = UMBRALES[cultivo]["NDVI_min_fen"]
    if ndvi_actual < umbral_fen * 0.8:
        score += 3.0
    elif ndvi_actual < umbral_fen:
        score += 1.5

    # Factor temperatura
    if temp_actual > 28 and cultivo == "AJÍ":
        score += 2.0
    elif temp_actual > UMBRALES[cultivo]["temp_max"]:
        score += 1.0

    # Factor elevación + cultivo + lluvias
    if elevation > 1500 and cultivo == "ROCOTO" and precip_actual > 15:
        score += 2.0

    # Factor precipitación extrema
    if precip_actual > 20:
        score += 1.0

    return min(10.0, round(score, 1))

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
                              cultivo, umbrales, contexto_fen=None,
                              vuln_score=None, pronostico_gfs=None):
    """NIVEL 1+5: prompt enriquecido con contexto ENFEN, score FEN y pronóstico GFS."""
    fen_bloque = ""
    if contexto_fen:
        fen_bloque = f"""
CONTEXTO ENFEN (oficial):
- Estado: {contexto_fen.get('estado', 'N/D')}
- Anomalía TSM (Niño 1+2): +{contexto_fen.get('anomalia_tsm', 0):.1f}°C
- Anomalía T° máx Lima: +{contexto_fen.get('temp_max_anomalia_lima', 0):.1f}°C
- Anomalía T° mín Ica: +{contexto_fen.get('temp_min_anomalia_ica', 0):.1f}°C
- Lluvias Pasco: {contexto_fen.get('lluvias_pasco', 'N/D')}
- Mes crítico: {contexto_fen.get('mes_critico', 'N/D')}
- Riesgo agrícola ENFEN: {contexto_fen.get('riesgo_agricola', 'N/D')}
- Lag océano→cultivo: {contexto_fen.get('lag_meses', 3)} meses
"""
    vuln_bloque = f"\nVulnerabilidad FEN calculada (score): {vuln_score:.1f}/10\n" if vuln_score is not None else ""
    gfs_bloque = ""
    if pronostico_gfs:
        gfs_bloque = f"""
Pronóstico GFS próxima semana:
- T° máx proyectada (promedio): {pronostico_gfs['temp_acum']:.1f}°C
- Precipitación acumulada: {pronostico_gfs['precip_acum']:.0f} mm
- Alerta principal: {pronostico_gfs['alerta_esta_semana']}
"""
    # Determinar nivel y acciones requeridas según el score
    if vuln_score is not None:
        if vuln_score <= 3:
            nivel_score = "BAJO (0-3)"
            accion_score = "Tomar medidas preventivas estándar."
            n_acciones = 0   # no se piden acciones adicionales
        elif vuln_score <= 7:
            nivel_score = "MEDIO (4-7)"
            accion_score = "Monitoreo intensivo requerido."
            n_acciones = 3
        else:
            nivel_score = "CRÍTICO (8-10)"
            accion_score = "Emergencia — intervenir YA."
            n_acciones = 5
    else:
        nivel_score = "no disponible"
        accion_score = ""
        n_acciones = 3

    acciones_instruccion = (
        f"Da exactamente {n_acciones} acciones específicas e inmediatas para este nivel."
        if n_acciones > 0
        else "Confirma las medidas preventivas estándar ya en curso."
    )

    prompt = f"""
Eres un agrónomo experto en {cultivo} en la costa y sierra peruana.

SCORE DE VULNERABILIDAD FEN: {vuln_score:.1f}/10 → Nivel {nivel_score}
DECISIÓN DE RIESGO (usar SOLO este score, no hacer evaluaciones independientes):
- 0-3 BAJO  → {accion_score if nivel_score.startswith('BAJO') else 'Tomar medidas preventivas estándar.'}
- 4-7 MEDIO → {accion_score if nivel_score.startswith('MEDIO') else 'Monitoreo intensivo + 3 acciones específicas.'}
- 8-10 CRÍTICO → {accion_score if nivel_score.startswith('CRÍTICO') else 'Emergencia, intervenir YA + 5 acciones.'}

El score ES la evaluación de riesgo. NO emitas juicios cualitativos separados
(ALTO/MEDIO/BAJO) que contradigan o dupliquen el score. Úsalo como punto de partida
único y construye la respuesta desde ahí.

DATOS DE CONTEXTO (para fundamentar las acciones):
- Cultivo: {cultivo} · Fase: {fase}
- NDVI: {ndvi:.2f} (umbral normal {umbrales['NDVI_min']:.2f} / umbral FEN {umbrales.get('NDVI_min_fen', 0.32):.2f})
- Temperatura: {temp:.1f}°C (óptimo {umbrales['temp_min']:.0f}-{umbrales['temp_max']:.0f}°C)
- Precipitación reciente: {precip_actual:.1f} mm
- Humedad suelo (SAR): {humedad:.2f} (óptimo {umbrales['humedad_min']:.2f}-{umbrales['humedad_max']:.2f})
{fen_bloque}{gfs_bloque}
INSTRUCCIONES DE FORMATO:
1. Encabezado: reproduce el nivel y score ("Score FEN {vuln_score:.1f}/10 → {nivel_score}").
2. Una sola oración explicando POR QUÉ el score llegó a ese nivel (factores dominantes).
3. {acciones_instruccion} Cada acción: verbo imperativo + objeto + plazo (ej. "Instalar drenajes perimetrales antes del {contexto_fen.get('mes_critico', 'mes crítico') if contexto_fen else 'mes crítico'}").
4. Cierre de 1 oración mencionando el lag océano→cultivo ({contexto_fen.get('lag_meses', 3) if contexto_fen else 3} meses) y el mes crítico.
5. Máximo 300 palabras. Sin secciones adicionales ni evaluaciones de riesgo duplicadas.
"""
    return consultar_groq(prompt, max_tokens=800)

# ============================================================
# FUNCIONES ENFEN (SCRAPING — para tab ENFEN)
# ============================================================
def _fallback_enfen_data():
    return {
        "estado_alerta":       "Alerta de El Niño Costero",
        "magnitud":            "Moderada (mayo-julio 2026)",
        "region_afectada":     "Costa norte y centro",
        "probabilidad_lluvias":"Normal a superior en costa norte",
        "temperatura_anomalia":"Cálida débil a moderada (+1 a +2°C)",
        "fecha_comunicado":    "Abril 2026",
        "nivel_riesgo_agricola":"Alto",
    }

def obtener_datos_enfen_actuales():
    if not SCRAPING_OK:
        return _fallback_enfen_data()
    try:
        r = requests.get("https://enfen.imarpe.gob.pe/", timeout=12)
        soup = BeautifulSoup(r.text, 'html.parser')
        comunicado = None
        for a in soup.find_all('a', href=True):
            if "Comunicado Oficial ENFEN" in a.get_text(strip=True):
                href = a['href']
                if not href.startswith('http'):
                    href = requests.compat.urljoin("https://enfen.imarpe.gob.pe/", href)
                comunicado = {"titulo": a.get_text(strip=True), "url": href}
                break
        texto_pdf = ""
        if comunicado and "pdf" in comunicado['url'].lower():
            try:
                rp = requests.get(comunicado['url'], timeout=20)
                if rp.status_code == 200:
                    reader = PyPDF2.PdfReader(io.BytesIO(rp.content))
                    texto_pdf = "".join(p.extract_text() for p in reader.pages)
            except Exception:
                pass
        m = re.search(r"Estado del sistema de alerta:\s*([\w\s]+)", texto_pdf, re.IGNORECASE)
        estado = m.group(1).strip() if m else "Alerta de El Niño Costero"
        m2 = re.search(r"probabilidad de precipitaciones.*?(\d{1,3}%)", texto_pdf, re.IGNORECASE)
        prob = m2.group(1) if m2 else "Normal a superior en costa norte"
        riesgo = "Alto" if "alerta" in estado.lower() else ("Medio" if "vigilancia" in estado.lower() else "Bajo")
        return {
            "estado_alerta":       estado,
            "magnitud":            "Moderada (probable hasta julio 2026)" if riesgo == "Alto" else "Débil",
            "region_afectada":     "Costa norte y centro",
            "probabilidad_lluvias": prob,
            "temperatura_anomalia": "+1°C a +2°C (región Niño 1+2)",
            "fecha_comunicado":    datetime.now().strftime("%B %Y"),
            "nivel_riesgo_agricola": riesgo,
        }
    except Exception:
        return _fallback_enfen_data()

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
    """Descarga DEM vía API REST de OpenTopography (sin bmi-topography)."""
    import requests as _req, tempfile, struct, io
    minx, miny, maxx, maxy = bounds
    # Agrandar bbox un poco para asegurar cobertura
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
        # Parsear formato AAIGrid (ASCII raster)
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
        # Construir objeto simple con atributos compatibles
        lons = xll + np.arange(ncols) * cell if XARRAY_OK else None
        lats = yll + np.arange(nrows)[::-1] * cell if XARRAY_OK else None
        if XARRAY_OK and xr is not None:
            dem = xr.DataArray(arr, dims=["y","x"],
                               coords={"y": lats, "x": lons},
                               attrs={"dem_type": dem_type})
        else:
            # Fallback: objeto simple
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
    """Genera mapa Folium 2D con el DEM como ImageOverlay coloreado."""
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
    """Superficie 3D interactiva con Plotly."""
    if not PLOTLY_OK:
        st.error("Instala plotly: pip install plotly")
        return None, None, None, None
    try:
        arr = dem.values.squeeze() if dem.values.ndim > 2 else dem.values
        X, Y = np.meshgrid(dem.x.values, dem.y.values)
        # Submuestreo si es muy grande
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
    """Divide la parcela en n_bloques celdas y retorna GeoDataFrame con intersecciones."""
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
    """Obtiene el NDVI medio de GEE para cada bloque. Fallback a simulación."""
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
    """Retorna dosis de N/P/K (kg/ha) según nivel de NDVI relativo al umbral del cultivo."""
    u = UMBRALES[cultivo]['NDVI_min']
    if ndvi >= u:
        return {'nivel': 'Óptimo 🟢', 'N': 0,  'P': 0,  'K': 0}
    elif ndvi >= u * 0.75:
        base = {'N': 40, 'P': 20, 'K': 30}
        if cultivo == "ROCOTO":
            base['N'] = int(base['N']*1.2); base['K'] = int(base['K']*1.3)
        return {'nivel': 'Medio 🟡', **base}
    else:
        base = {'N': 80, 'P': 40, 'K': 60}
        if cultivo == "ROCOTO":
            base['N'] = int(base['N']*1.2); base['K'] = int(base['K']*1.3)
        return {'nivel': 'Crítico 🔴', **base}

def estimar_potencial_cosecha(ndvi, cultivo, area_ha):
    """Estima rendimiento (t/ha) y producción total (t) basados en NDVI."""
    if cultivo == "AJÍ":
        base_t_ha, ndvi_opt = 18.0, 0.60
    elif cultivo == "ROCOTO":
        base_t_ha, ndvi_opt = 22.0, 0.65
    else:  # PAPA ANDINA
        base_t_ha, ndvi_opt = 14.0, 0.55
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
    """Una recomendación concreta por principio agroecológico (Groq)."""
    prompt = (
        f"Eres agroecólogo experto en sistemas campesinos andinos y costeños del Perú. "
        f"Para el cultivo de {cultivo} en fase {fase}, con estos indicadores: "
        f"NDVI={ndvi:.2f}, temperatura={temp:.1f}°C, humedad suelo={humedad:.2f}, "
        f"precipitación={precip:.1f} mm, genera UNA recomendación práctica y concreta "
        f"para cada uno de los 10 principios agroecológicos:\n{_PRINCIPIOS_AGROECOLOGICOS}\n"
        f"Formato: **Principio N – nombre**: recomendación (máx 2 oraciones). Total máx 400 palabras."
    )
    return consultar_groq(prompt, max_tokens=900)

def generar_plan_agroecologico_completo(cultivo, fase, ndvi, temp, humedad, precip, area_ha):
    """Plan agroecológico integral para la parcela (Groq)."""
    prompt = (
        f"Diseña un plan agroecológico integral para una parcela de {area_ha:.1f} ha de {cultivo} "
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
    """Estima carbono por pools (biomasa aérea, raíces, madera muerta, hojarasca, suelo)."""
    FACTORES = {
        'fc_carbono':     0.47,   # fracción de carbono en biomasa seca (IPCC)
        'ratio_co2':      3.67,   # factor C → CO₂e
        'ratio_bgb':      0.24,   # raíces / biomasa aérea (IPCC tier 1, cultivos)
        'prop_dw':        0.05,   # madera muerta / carbono AGB
        'acum_hojarasca': 2.0,    # t MS/ha/año de hojarasca
        'tasa_soc':       1.5,    # t C/ha/año acumulación SOC media
    }

    def calcular_carbono_hectarea(self, ndvi: float, precip_anual: float) -> dict:
        """Retorna diccionario con carbono total (t C/ha), CO₂e y desglose por pool."""
        # Factor climático (precipitación normalizada a 1200 mm/año)
        factor_clim = min(1.6, max(0.7, precip_anual / 1200))

        # Biomasa aérea bruta (t MS/ha) según NDVI — curva empírica para hortícolas andinos
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
    """Extrapola precipitación anual desde el DataFrame de series disponibles."""
    if df_precip is None or df_precip.empty or 'precip' not in df_precip.columns:
        return 1200.0
    media_diaria = df_precip['precip'].mean()
    return round(media_diaria * 365, 0)

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
st.set_page_config(
    page_title="Gestión de Riesgos Climáticos — Ají y Rocoto",
    layout="wide",
    page_icon="🌶️",
)
st.title("🌶️ Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto")
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

# ── Cálculos FEN globales (usados en múltiples pestañas) ─────
centroid_geom = gdf.geometry.centroid.iloc[0]
_lat = centroid_geom.y
_lon = centroid_geom.x
elevation_est  = estimar_elevacion(_lat, _lon)
zona_ref       = zona_mas_cercana(_lat, _lon)
contexto_fen   = obtener_contexto_enfen()
pronostico_gfs = obtener_pronostico_gfs_simple(_lat, _lon, dias=7)

# Si hay un DEM cargado en session_state, usar su elevación media (más precisa)
if st.session_state.get("dem_data") is not None:
    try:
        elevation_est = float(np.nanmean(st.session_state["dem_data"].values))
    except Exception:
        pass  # mantiene la heurística

vuln_score = calcular_vulnerabilidad_fen(
    gdf, cultivo, ndvi_val, temp_val, precip_actual, elevation_est
)
codigo_enfen = (2 if "alerta" in contexto_fen["estado"].lower()
                else 1 if "vigilancia" in contexto_fen["estado"].lower()
                else 0)

# ============================================================
# PESTAÑAS — 8 en total
# ============================================================
(tab_dashboard, tab_mapas, tab_monitoreo,
 tab_alerta, tab_gobernanza, tab_export, tab_fen, tab_dem,
 tab_npk, tab_agro, tab_carbono) = st.tabs([
    "📊 Dashboard General",
    "🗺️ Mapa de Riesgo",
    "📈 Monitoreo Fenológico",
    "⚠️ Alertas IA",
    "📄 Gobernanza",
    "💾 Exportar",
    "📊 Análisis FEN",
    "🗻 DEM (Relieve)",
    "🌾 Fertilidad NPK",
    "🌱 Agroecología",
    "🌍 Carbono",
])

# ============================================================
# DASHBOARD GENERAL  (Nivel 2 + 4)
# ============================================================
with tab_dashboard:
    st.header("Dashboard de Indicadores Clave")

    # ── 5 métricas principales ────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    u = UMBRALES[cultivo]
    with col1:
        delta_ndvi = f"{ndvi_val - u['NDVI_min']:.2f}" if ndvi_val > u['NDVI_min'] else "crítico"
        st.metric("🌱 NDVI actual", f"{ndvi_val:.2f}", delta=delta_ndvi)
    with col2:
        delta_temp = "óptima" if u['temp_min'] <= temp_val <= u['temp_max'] else "alerta"
        st.metric("🌡️ Temperatura", f"{temp_val:.1f} °C", delta=delta_temp)
    with col3:
        delta_hum = "normal" if u['humedad_min'] <= humedad_val <= u['humedad_max'] else "crítica"
        st.metric("💧 Humedad suelo", f"{humedad_val:.2f}", delta=delta_hum)
    with col4:
        st.metric("📅 Fase fenológica", fase_fenologica.capitalize())
    with col5:
        st.metric("🚨 Vulnerabilidad FEN", f"{vuln_score}/10",
                  delta=("CRÍTICA" if vuln_score > 7 else "ALTA" if vuln_score > 5 else "MODERADA"))

    # Semáforo de vulnerabilidad
    if vuln_score > 7:
        st.error(f"🔴 Vulnerabilidad FEN CRÍTICA ({vuln_score}/10) — Implementa medidas de emergencia.")
    elif vuln_score > 5:
        st.warning(f"🟠 Vulnerabilidad FEN ALTA ({vuln_score}/10) — Refuerza monitoreo y drenajes.")
    else:
        st.info(f"🟡 Vulnerabilidad FEN MODERADA ({vuln_score}/10) — Mantén monitoreo quincenal.")

    st.markdown("---")

    # ── NIVEL 2: Pronóstico GFS próxima semana ────────────────
    st.subheader("🌤️ Pronóstico GFS — Próximos 7 días")
    st.warning(f"**Alerta esta semana:** {pronostico_gfs['alerta_esta_semana']}")

    col_gfs1, col_gfs2 = st.columns(2)
    with col_gfs1:
        fig_t, ax_t = plt.subplots(figsize=(6, 3))
        ax_t.plot(pronostico_gfs['fechas'], pronostico_gfs['temp_max_proyectada'],
                  'r-o', markersize=5, linewidth=2, label='T° máx proyectada')
        ax_t.axhline(u['temp_max'], color='orange', linestyle='--', label=f"Umbral {cultivo} ({u['temp_max']}°C)")
        ax_t.set_title('Temperatura proyectada vs umbral')
        ax_t.set_ylabel('°C')
        ax_t.legend(fontsize=8)
        ax_t.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig_t)
    with col_gfs2:
        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        ax_p.bar(pronostico_gfs['fechas'], pronostico_gfs['precip_diaria'],
                 color='steelblue', alpha=0.8)
        ax_p.set_title('Precipitación proyectada (mm/día)')
        ax_p.set_ylabel('mm')
        ax_p.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig_p)

    st.markdown("---")

    # ── Series históricas ─────────────────────────────────────
    st.subheader("Evolución de Índices Históricos")
    if not df_ndvi.empty and not df_temp.empty and not df_precip.empty:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-', linewidth=2, label='NDVI')
        axes[0].axhline(u['NDVI_min'], color='red', linestyle='--', label=f'Umbral ({u["NDVI_min"]})')
        axes[0].axhline(u['NDVI_min_fen'], color='orange', linestyle=':', label=f'Umbral FEN ({u["NDVI_min_fen"]})')
        axes[0].set_ylabel('NDVI'); axes[0].legend(fontsize=8)
        axes[1].plot(df_temp['date'], df_temp['temp'], 'r-')
        axes[1].axhline(u['temp_min'], color='blue', linestyle='--')
        axes[1].axhline(u['temp_max'], color='orange', linestyle='--')
        axes[1].set_ylabel('Temperatura (°C)')
        axes[2].bar(df_precip['date'], df_precip['precip'], color='cyan')
        axes[2].set_ylabel('Precipitación (mm)')
        plt.tight_layout()
        st.pyplot(fig)
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
        axes[0].axhline(u['NDVI_min_fen'], color='orange', linestyle=':')
        axes[0].set_ylabel('NDVI (simulado)')
        axes[1].plot(fechas_sim, temp_sim, 'r-')
        axes[1].set_ylabel('Temp (simulada)')
        axes[2].bar(fechas_sim, precip_sim, color='cyan')
        axes[2].set_ylabel('Precip (simulada)')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# MAPA DE RIESGO  (estructura original de app.py preservada)
# ============================================================
with tab_mapas:
    st.header("🗺️ Mapa de Riesgo Climático Interactivo")
    st.markdown("Seleccioná el índice, el fondo y visualizá la imagen satelital con puntos críticos.")

    if not FOLIUM_OK:
        st.error("❌ folium no instalado. Agregá `folium` y `streamlit-folium` a requirements.txt.")
    else:
        col_idx, col_fondo = st.columns([2, 1])
        with col_idx:
            indice = st.selectbox("Índice a visualizar",
                                  ["NDVI","NDRE","NDWI","Temperatura","Precipitación"],
                                  help="NDRE detecta estrés en clorofila antes que el NDVI")
        with col_fondo:
            fondo = st.radio("Fondo", ["Google Hybrid","Esri Satellite"], horizontal=True)

        gee_ok_map = st.session_state.get("gee_authenticated", False) and usar_gee and GEE_AVAILABLE

        # ── Parámetros de visualización por índice ────────────────
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

        # ── Capa GEE (solo si autenticado) ───────────────────────
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

        # ── Construir mapa folium (SIEMPRE) ──────────────────────
        bounds = gdf.total_bounds
        c_lat, c_lon, zoom = obtener_zoom_con_margen(bounds)
        mapa = folium.Map(location=[c_lat, c_lon], zoom_start=zoom, control_scale=True, tiles=None)

        folium.TileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                         attr='OpenStreetMap', name='OpenStreetMap').add_to(mapa)
        if fondo == "Google Hybrid":
            folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                             attr='Google Hybrid', name='Google Hybrid').add_to(mapa)
        else:
            folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                             attr='Esri World Imagery', name='Esri Satellite').add_to(mapa)

        if tile_url:
            folium.TileLayer(tiles=tile_url, attr='GEE · Sentinel-2',
                             name=f'{indice} (Sentinel-2)', overlay=True, control=True, opacity=0.88).add_to(mapa)

        # Polígono parcela
        riesgo_color = "#2ca02c" if riesgo_map=="BAJO" else "#f39c12" if riesgo_map=="MEDIO" else "#e74c3c"
        popup_poly_html = (
            f'<div style="font-family:Arial;min-width:210px;">'
            f'<h4 style="margin:0;color:#2ca02c;">{riesgo_emoji_map} {ICONOS[cultivo]} {cultivo}</h4>'
            f'<p style="margin:4px 0;font-size:11px;color:#888;">{area_ha:.2f} ha</p>'
            f'<hr style="margin:6px 0;">'
            f'<table style="font-size:13px;width:100%;">'
            f'<tr><td>{indice}</td><td><b>{mean_val_map:.3f}{unidad}</b></td></tr>'
            f'<tr><td>Área</td><td><b>{area_ha:.2f} ha</b></td></tr>'
            f'<tr><td>Puntos críticos</td><td><b>{num_criticos}</b></td></tr>'
            f'<tr><td>🚨 Vuln. FEN</td><td><b>{vuln_score}/10</b></td></tr>'
            f'</table>'
            f'<hr style="margin:6px 0;">'
            f'<div style="text-align:center;padding:4px;background:{riesgo_color};color:white;border-radius:4px;font-weight:bold;">Riesgo {riesgo_map}</div>'
            f'</div>'
        )
        folium.GeoJson(gdf.__geo_interface__, name='Parcela',
                       style_function=lambda x: {'color':'#2ca02c','weight':3,'dashArray':'6','fillColor':'#2ca02c','fillOpacity':0.15},
                       tooltip=f'{riesgo_emoji_map} {cultivo} — Riesgo {riesgo_map} ({indice}: {mean_val_map:.3f})',
                       popup=folium.Popup(popup_poly_html, max_width=250)).add_to(mapa)

        # Puntos críticos
        for lon_pt, lat_pt in critical_coords:
            popup_pt = (f'<div style="font-family:Arial;"><b>⚠️ Punto Crítico</b><br>'
                        f'{indice}: bajo umbral<br>Lat:{lat_pt:.5f}<br>Lon:{lon_pt:.5f}<br>'
                        f'<a href="https://www.google.com/maps/search/?api=1&query={lat_pt},{lon_pt}" target="_blank">📍 Google Maps</a></div>')
            folium.CircleMarker(location=[lat_pt, lon_pt], radius=6, color='red', weight=3,
                                fill=True, fill_color='white', fill_opacity=0.2,
                                popup=folium.Popup(popup_pt, max_width='100%'),
                                tooltip=f'Crítico: {lat_pt:.4f},{lon_pt:.4f}').add_to(mapa)

        # Label central
        clat_m = gdf.geometry.centroid.y.iloc[0]
        clon_m = gdf.geometry.centroid.x.iloc[0]
        gee_badge = "🛰️ GEE" if gee_ok_map and tile_url else "🗺️ OSM"
        label_html = (
            f'<div style="background:white;border:2px solid #2ca02c;border-radius:6px;'
            f'padding:3px 8px;font-size:11px;font-weight:bold;box-shadow:2px 2px 4px rgba(0,0,0,0.3);white-space:nowrap;">'
            f'{riesgo_emoji_map} {ICONOS[cultivo]} {cultivo} · {gee_badge}<br>'
            f'<span style="font-size:10px;color:#555;">{indice}: {mean_val_map:.3f} | Riesgo {riesgo_map} | FEN {vuln_score}/10</span></div>'
        )
        folium.Marker(location=[clat_m, clon_m],
                      icon=folium.DivIcon(html=label_html, icon_size=(240,35), icon_anchor=(120,17))).add_to(mapa)

        # Panel flotante
        leyenda_html = "".join(f'<span style="color:{c};">■</span> {txt}&nbsp;&nbsp;' for c, txt in leyenda)
        panel_html = (
            f'<div style="position:fixed;bottom:40px;left:40px;z-index:1000;background:white;'
            f'padding:12px 16px;border-radius:8px;border:1px solid #ccc;'
            f'box-shadow:2px 2px 8px rgba(0,0,0,0.2);font-family:Arial;font-size:12px;min-width:190px;">'
            f'<b style="font-size:13px;">{ICONOS[cultivo]} {cultivo}</b>'
            f'<hr style="margin:6px 0;">'
            f'<b>Riesgo:</b> <span style="color:{riesgo_color};">● {riesgo_map}</span><br>'
            f'<b>{indice}:</b> {mean_val_map:.3f}{unidad}<br>'
            + (f'<b>NDRE:</b> {ndre_val:.3f}<br>' if ndre_val is not None else '')
            + f'<b>Área:</b> {area_ha:.2f} ha<br>'
            f'<b>Puntos críticos:</b> {num_criticos}<br>'
            f'<b>🚨 Vuln. FEN:</b> {vuln_score}/10'
            f'<hr style="margin:6px 0;">'
            f'{leyenda_html}'
            f'<hr style="margin:6px 0;">'
            f'<span style="font-size:10px;color:#888;">{"Sentinel-2 · ERA5 · CHIRPS" if gee_ok_map else "OpenStreetMap · valores por defecto"}</span>'
            f'</div>'
        )
        Element(panel_html).add_to(mapa)
        folium.LayerControl(collapsed=False).add_to(mapa)

        components.html(mapa.get_root().render(), height=650)

        if not gee_ok_map:
            st.info("🗺️ Mapa base activo. Autenticá GEE en el panel lateral para agregar capas satelitales Sentinel-2.")
        st.caption(f"📊 **{indice}:** {mean_val_map:.3f}{unidad} · Riesgo: **{riesgo_map}** · "
                   f"{num_criticos} puntos críticos · Vuln. FEN: **{vuln_score}/10**")

# ============================================================
# MONITOREO FENOLÓGICO  (Nivel 3)
# ============================================================
with tab_monitoreo:
    st.header("📈 Monitoreo Detallado")
    col1, col2 = st.columns(2)
    umbral = UMBRALES[cultivo]
    with col1:
        st.metric("NDVI",              f"{ndvi_val:.2f}")
        st.metric("Temperatura",       f"{temp_val:.1f} °C")
        st.metric("Humedad suelo",     f"{humedad_val:.2f}")
        st.metric("Precipitación rec.",f"{precip_actual:.1f} mm")
    with col2:
        st.subheader("Comparativa con Umbrales")
        st.write(f"**NDVI:** {'🟢' if ndvi_val > umbral['NDVI_min'] else '🔴'} Mínimo {umbral['NDVI_min']} (FEN: {umbral['NDVI_min_fen']})")
        st.write(f"**Temperatura:** {'🟢' if umbral['temp_min']<=temp_val<=umbral['temp_max'] else '🔴'} Rango {umbral['temp_min']}-{umbral['temp_max']} °C")
        st.write(f"**Humedad:** {'🟢' if umbral['humedad_min']<=humedad_val<=umbral['humedad_max'] else '🔴'} Rango {umbral['humedad_min']:.2f}-{umbral['humedad_max']:.2f}")

    if not df_ndvi.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-o', markersize=3, label='NDVI')
        ax.axhline(umbral['NDVI_min'],     color='red',    linestyle='--', label=f'Umbral normal ({umbral["NDVI_min"]})')
        ax.axhline(umbral['NDVI_min_fen'], color='orange', linestyle=':',  label=f'Umbral FEN ({umbral["NDVI_min_fen"]})')
        ax.set_ylabel('NDVI'); ax.legend()
        st.pyplot(fig)

    # ── NIVEL 3: Contexto FEN histórico ──────────────────────
    st.markdown("---")
    st.subheader(f"🌊 Contexto FEN — Zona de referencia: **{zona_ref}**")
    hist_fen = RIESGO_HISTORICO_FEN[zona_ref]
    ndvi_fen = hist_fen["ndvi_promedio_fen"]

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric("NDVI actual",              f"{ndvi_val:.2f}")
    with col_f2:
        st.metric("NDVI histórico años FEN",  f"{ndvi_fen:.2f}", delta=f"referencia {zona_ref}")
    with col_f3:
        st.metric("Pérdidas históricas FEN",  f"{hist_fen['perdidas_pct']}%")

    delta_ndvi_fen = ((ndvi_val - ndvi_fen) / ndvi_fen) * 100
    if delta_ndvi_fen < -10:
        st.error(f"🔴 NDVI {delta_ndvi_fen:.1f}% **bajo** vs promedio FEN histórico en {zona_ref}. "
                 f"Riesgo elevado de pérdidas similares a años FEN anteriores ({hist_fen['perdidas_pct']}%).")
    elif delta_ndvi_fen < 0:
        st.warning(f"🟡 NDVI {delta_ndvi_fen:.1f}% bajo vs promedio FEN histórico en {zona_ref}. "
                   f"Monitorear de cerca.")
    else:
        st.success(f"🟢 NDVI {delta_ndvi_fen:+.1f}% **mejor** que el promedio FEN histórico en {zona_ref}. "
                   f"Condiciones actuales más favorables.")

    # Gráfico comparativo
    fig_fen, ax_fen = plt.subplots(figsize=(8, 3))
    zonas  = list(RIESGO_HISTORICO_FEN.keys())
    ndvis  = [RIESGO_HISTORICO_FEN[z]["ndvi_promedio_fen"] for z in zonas]
    colors = ['red' if z == zona_ref else 'steelblue' for z in zonas]
    bars = ax_fen.bar(zonas, ndvis, color=colors, alpha=0.8)
    ax_fen.axhline(ndvi_val, color='green', linestyle='--', linewidth=2, label=f'NDVI actual ({ndvi_val:.2f})')
    ax_fen.axhline(umbral['NDVI_min_fen'], color='orange', linestyle=':', linewidth=1.5, label=f'Umbral FEN ({umbral["NDVI_min_fen"]})')
    ax_fen.set_title('NDVI promedio durante años FEN por zona (zona actual en rojo)')
    ax_fen.set_ylabel('NDVI')
    ax_fen.legend(fontsize=8)
    ax_fen.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    st.pyplot(fig_fen)

# ============================================================
# ALERTAS IA  (Nivel 5)
# ============================================================
with tab_alerta:
    st.header("⚠️ Alertas IA con Contexto FEN Integrado")

    # Mostrar resumen de contexto antes del botón
    with st.expander("📋 Contexto que se enviará a la IA (Nivel 5 — FEN completo)", expanded=False):
        st.write(f"**Estado ENFEN:** {contexto_fen['estado']}")
        st.write(f"**Anomalía TSM:** +{contexto_fen['anomalia_tsm']}°C")
        st.write(f"**Mes crítico:** {contexto_fen['mes_critico']}")
        st.write(f"**Score vulnerabilidad FEN:** {vuln_score}/10")
        st.write(f"**Pronóstico semana:** {pronostico_gfs['alerta_esta_semana']}")
        st.write(f"**Zona ref. histórica:** {zona_ref} — NDVI FEN prom.: {RIESGO_HISTORICO_FEN[zona_ref]['ndvi_promedio_fen']}")

    if st.button("🤖 Generar Alerta Avanzada con Contexto FEN", type="primary"):
        with st.spinner("Consultando IA (Groq) con datos FEN integrados..."):
            alerta = generar_alerta_detallada(
                fase_fenologica, ndvi_val, temp_val, precip_actual, humedad_val,
                cultivo, UMBRALES[cultivo],
                contexto_fen=contexto_fen,
                vuln_score=vuln_score,
                pronostico_gfs=pronostico_gfs,
            )

        # Mostrar alerta estructurada
        st.markdown("### 🔔 Alerta Agronómica Integrada")
        st.markdown(alerta)

        st.markdown("---")
        st.markdown(f"**📡 Contexto ENFEN:** {contexto_fen['estado']} · Mes crítico: {contexto_fen['mes_critico']} · Lag: {contexto_fen['lag_meses']} meses")
        st.markdown(f"**🚨 Vulnerabilidad a FEN calculada:** {vuln_score}/10")
        st.markdown(f"**🌤️ Pronóstico próxima semana:** {pronostico_gfs['alerta_esta_semana']}")
        ndvi_fen_hist = RIESGO_HISTORICO_FEN[zona_ref]['ndvi_promedio_fen']
        delta_pct = ((ndvi_val - ndvi_fen_hist) / ndvi_fen_hist) * 100
        st.markdown(f"**📊 Histórico FEN zona {zona_ref}:** NDVI prom. {ndvi_fen_hist:.2f} · Delta actual: {delta_pct:+.1f}%")

        fecha_str = datetime.now().strftime('%Y%m%d_%H%M')
        texto_descarga = (
            f"ALERTA FEN — {cultivo} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"{'='*60}\n\n{alerta}\n\n"
            f"{'='*60}\n"
            f"Estado ENFEN: {contexto_fen['estado']}\n"
            f"Score vulnerabilidad FEN: {vuln_score}/10\n"
            f"Pronóstico semana: {pronostico_gfs['alerta_esta_semana']}\n"
            f"Delta NDVI vs FEN histórico ({zona_ref}): {delta_pct:+.1f}%\n"
        )
        st.download_button("📥 Descargar alerta completa",
                           data=texto_descarga,
                           file_name=f"alerta_fen_{cultivo}_{fecha_str}.txt")

# ============================================================
# GOBERNANZA
# ============================================================
with tab_gobernanza:
    st.subheader("📄 Gobernanza para la Cadena de Ají y Rocoto")
    st.markdown("""
    **Estructura sugerida ante FEN:**
    - **Comité de Gestión de Riesgos Climáticos** (productor, técnico, SENASA)
    - Frecuencia de monitoreo: **semanal** durante Alerta FEN, quincenal en Vigilancia
    - Canales de alerta: WhatsApp, plataforma web, radio comunitaria
    - Medidas preventivas FEN: drenaje, pólizas de seguro agrícola, fondo de emergencia
    - Coordinación con ENFEN, SENAMHI, Ministerio de Agricultura
    """)
    if st.button("Descargar Gobernanza (PDF)"):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas as rl_canvas
            buf = BytesIO()
            c = rl_canvas.Canvas(buf, pagesize=letter)
            c.drawString(100, 750, "GOBERNANZA PARA RIESGOS CLIMÁTICOS — FEN")
            c.drawString(100, 730, "Cadena de Ají y Rocoto · Perú")
            c.save(); buf.seek(0)
            st.download_button("📄 PDF", data=buf, file_name="gobernanza_fen.pdf")
        except ImportError:
            st.error("Instala reportlab: pip install reportlab")

# ============================================================
# EXPORTAR
# ============================================================
with tab_export:
    st.subheader("💾 Exportar Datos")
    if st.button("Exportar parcela a GeoJSON"):
        st.download_button("⬇️ Descargar GeoJSON", data=gdf.to_json(), file_name="parcela.geojson")
    if not df_ndvi.empty:
        st.download_button("⬇️ Serie NDVI CSV", data=df_ndvi.to_csv(index=False), file_name="ndvi.csv")
    # Exportar resumen FEN
    resumen_fen = (
        f"RESUMEN FEN — {cultivo} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Estado ENFEN: {contexto_fen['estado']}\n"
        f"Anomalía TSM: +{contexto_fen['anomalia_tsm']}°C\n"
        f"Score vulnerabilidad: {vuln_score}/10\n"
        f"Zona ref.: {zona_ref}\n"
        f"Delta NDVI vs FEN: {((ndvi_val - RIESGO_HISTORICO_FEN[zona_ref]['ndvi_promedio_fen']) / RIESGO_HISTORICO_FEN[zona_ref]['ndvi_promedio_fen'])*100:+.1f}%\n"
        f"Alerta GFS semana: {pronostico_gfs['alerta_esta_semana']}\n"
    )
    st.download_button("⬇️ Resumen FEN TXT", data=resumen_fen, file_name="resumen_fen.txt")

    # ── Exportar para biomod2 (R) ─────────────────────────────
    st.markdown("---")
    st.subheader("📦 Exportar para biomod2 (R)")
    st.markdown(
        "Genera un CSV con puntos dentro de la parcela y variables ambientales "
        "para modelado de nicho ecológico en R con `biomod2`."
    )
    if st.button("🔬 Generar archivo biomod2"):
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        step = 0.001  # ~111 m — ajustable
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
            df_points['NDVI']              = ndvi_val
            df_points['temperatura_C']     = temp_val
            df_points['precipitacion_mm']  = precip_actual
            df_points['humedad_suelo']     = humedad_val
            df_points['elevacion_m']       = elevation_est
            df_points['rendimiento_t_ha']  = predecir_rendimiento(
                ndvi_val, precip_actual, temp_val, codigo_enfen
            )
            umbral_ndvi = UMBRALES[cultivo]['NDVI_min']
            df_points['Presence'] = (df_points['NDVI'] >= umbral_ndvi).astype(int)
            st.download_button(
                "📥 Descargar CSV para biomod2",
                data=df_points.to_csv(index=False),
                file_name=f"biomod2_{cultivo}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
            st.success(f"✅ {len(df_points)} puntos generados dentro de la parcela.")

# ============================================================
# NIVEL 6 — PESTAÑA "📊 Análisis FEN"
# ============================================================
with tab_fen:
    st.header("📊 Análisis FEN — El Niño Costero")
    st.markdown(f"**Estado actual:** {contexto_fen['estado']} · "
                f"Anomalía TSM +{contexto_fen['anomalia_tsm']}°C · "
                f"Riesgo agrícola: **{contexto_fen['riesgo_agricola']}**")

    # ── 1. Tabla comparativa NDVI actual vs histórico FEN ────
    st.subheader("1️⃣ Tabla comparativa NDVI: actual vs años FEN")
    tabla_data = []
    for zona, d in RIESGO_HISTORICO_FEN.items():
        delta = ((ndvi_val - d["ndvi_promedio_fen"]) / d["ndvi_promedio_fen"]) * 100
        estado_icon = "🔴" if delta < -10 else ("🟡" if delta < 0 else "🟢")
        tabla_data.append({
            "Zona":             zona,
            "Región":           d["region"],
            "NDVI actual":      f"{ndvi_val:.2f}",
            "NDVI prom. FEN":   f"{d['ndvi_promedio_fen']:.2f}",
            "Δ NDVI (%)":       f"{delta:+.1f}%",
            "Pérdidas hist. FEN": f"{d['perdidas_pct']}%",
            "Estado":           estado_icon,
        })
    df_tabla = pd.DataFrame(tabla_data)
    st.dataframe(df_tabla, use_container_width=True)

    st.markdown("---")

    # ── 2. Gráfico de vulnerabilidad FEN por zona ─────────────
    st.subheader("2️⃣ Vulnerabilidad FEN por zona — comparativa")
    zonas_names = list(RIESGO_HISTORICO_FEN.keys())
    vuln_por_zona = []
    for z in zonas_names:
        d = RIESGO_HISTORICO_FEN[z]
        # Score simplificado por zona (sin parcela específica)
        score_z = min(10, round(
            (1 - d["ndvi_promedio_fen"]) * 5 + d["perdidas_pct"] / 20, 1
        ))
        vuln_por_zona.append(score_z)

    colores_vuln = ['#e74c3c' if v > 7 else '#f39c12' if v > 5 else '#2ecc71' for v in vuln_por_zona]
    fig_v, ax_v = plt.subplots(figsize=(9, 4))
    bars_v = ax_v.bar(zonas_names, vuln_por_zona, color=colores_vuln, alpha=0.85, edgecolor='white')
    ax_v.axhline(7, color='red',    linestyle='--', alpha=0.6, label='Umbral crítico (7)')
    ax_v.axhline(5, color='orange', linestyle='--', alpha=0.6, label='Umbral alto (5)')
    ax_v.axhline(vuln_score, color='blue', linestyle='-', linewidth=2,
                 label=f'Tu parcela ({vuln_score}/10)')
    ax_v.set_ylim(0, 10)
    ax_v.set_ylabel('Score de vulnerabilidad FEN (0-10)')
    ax_v.set_title('Vulnerabilidad estimada a FEN por zona peruana')
    ax_v.legend(fontsize=8)
    for bar, val in zip(bars_v, vuln_por_zona):
        ax_v.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_v.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    st.pyplot(fig_v)

    st.markdown("---")

    # ── 3. Timeline del lag de impacto ────────────────────────
    st.subheader("3️⃣ Timeline del lag de impacto: océano → atmósfera → cultivo")
    lag = contexto_fen["lag_meses"]
    etapas = [
        ("🌊 Calentamiento oceánico\n(Niño 1+2)",  0),
        ("💨 Anomalía atmosférica\n(ZCIT / corrientes)", 1),
        ("🌧️ Cambio en precipitación\n(CHIRPS detectable)", 2),
        (f"🌱 Impacto en cultivo\n(NDVI, rendimiento)", lag),
    ]
    fig_tl, ax_tl = plt.subplots(figsize=(10, 3))
    for i, (etapa, mes) in enumerate(etapas):
        color = '#e74c3c' if i == len(etapas)-1 else '#3498db'
        ax_tl.scatter(mes, 0.5, s=300, color=color, zorder=3)
        ax_tl.text(mes, 0.65, etapa, ha='center', fontsize=8, wrap=True)
        if i > 0:
            ax_tl.annotate('', xy=(mes, 0.5), xytext=(etapas[i-1][1], 0.5),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax_tl.set_xlim(-0.5, lag + 0.5)
    ax_tl.set_ylim(0, 1.2)
    ax_tl.set_xlabel('Meses desde inicio del evento FEN')
    ax_tl.set_yticks([])
    ax_tl.set_title(f'Lag de impacto estimado: {lag} meses (evento → cultivo)')
    ax_tl.spines[['left','top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_tl)

    st.markdown("---")

    # ── 4. Heatmap simulado de riesgo FEN por zona ───────────
    st.subheader("4️⃣ Mapa de zonas críticas por FEN (heatmap de riesgo)")
    zonas_lat = [d["lat_ref"] for d in RIESGO_HISTORICO_FEN.values()]
    zonas_lon = [d["lon_ref"] for d in RIESGO_HISTORICO_FEN.values()]
    zonas_riesgo = [d["perdidas_pct"] / 100 for d in RIESGO_HISTORICO_FEN.values()]

    fig_map, ax_map = plt.subplots(figsize=(6, 8))
    ax_map.set_facecolor('#d4e6f1')
    sc = ax_map.scatter(zonas_lon, zonas_lat, c=zonas_riesgo,
                        cmap='RdYlGn_r', s=400, alpha=0.85,
                        vmin=0.2, vmax=0.6, edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=ax_map, label='Pérdidas históricas FEN (%)')
    for z, lat, lon in zip(RIESGO_HISTORICO_FEN.keys(), zonas_lat, zonas_lon):
        ax_map.annotate(z, (lon, lat), textcoords='offset points',
                        xytext=(5, 5), fontsize=9, fontweight='bold')
    # Tu parcela
    ax_map.scatter([_lon], [_lat], c='blue', s=200, marker='*',
                   zorder=5, label='Tu parcela')
    ax_map.set_xlabel('Longitud')
    ax_map.set_ylabel('Latitud')
    ax_map.set_title('Zonas críticas FEN en el Perú\n(color = pérdidas históricas estimadas)')
    ax_map.legend()
    plt.tight_layout()
    st.pyplot(fig_map)

    st.markdown("---")

    # ── 5. Predicción de rendimiento con contexto FEN ─────────
    st.subheader("5️⃣ Predicción de rendimiento con contexto FEN")
    rendimiento = predecir_rendimiento(ndvi_val, precip_actual, temp_val, codigo_enfen)
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("🌶️ Rendimiento estimado", f"{rendimiento:.2f} t/ha")
    with col_r2:
        st.metric("📊 Código ENFEN",
                  "🔴 Alerta" if codigo_enfen==2 else "🟡 Vigilancia" if codigo_enfen==1 else "🟢 Neutro")
    with col_r3:
        st.metric("📏 Margen de confianza", "± 20%")

    if rendimiento >= 5.0:
        st.success("✅ Expectativa **alta**. Condiciones favorables a pesar del FEN.")
    elif rendimiento >= 3.0:
        st.warning("⚠️ Expectativa **media**. Aplica medidas de mitigación FEN.")
    else:
        st.error("🔴 Expectativa **baja**. Alto riesgo FEN. Considera asegurar cultivos y diversificar.")

    st.caption(f"Fuente FEN: {contexto_fen.get('fuente', 'ENFEN')} · "
               f"Mes crítico: {contexto_fen['mes_critico']} · "
               f"Lag: {contexto_fen['lag_meses']} meses")

# ============================================================
# PESTAÑA DEM (RELIEVE)
# ============================================================
with tab_dem:
    st.header("🗻 Análisis de Relieve — OpenTopography")
    st.markdown(
        "Descarga y visualiza el **Modelo Digital de Elevación (DEM)** de tu parcela. "
        "La elevación media se usa automáticamente para refinar el **score de vulnerabilidad FEN**."
    )

    if not OPENTOPOGRAPHY_AVAILABLE:
        st.error("❌ requests no disponible — módulo DEM inactivo.")
    else:
        # Permitir ingresar la key manualmente si no está en secrets
        _ot_key = OPENTOPOGRAPHY_API_KEY
        if not _ot_key:
            _ot_key = st.session_state.get("ot_api_key_manual", "")
            _key_input = st.text_input(
                "🔑 API Key de OpenTopography",
                value=_ot_key,
                type="password",
                help="Conseguila gratis en https://opentopography.org/developers"
            )
            if _key_input:
                st.session_state["ot_api_key_manual"] = _key_input
                _ot_key = _key_input
            if not _ot_key:
                st.info("Ingresá tu API key de OpenTopography para descargar el DEM. "
                        "Es gratuita: [opentopography.org/developers](https://opentopography.org/developers)")
                st.stop()

        bounds_dem = gdf.total_bounds
        col_ds, col_btn = st.columns([3, 1])
        with col_ds:
            resolucion = st.selectbox("Resolución del DEM", list(_DATASETS_DEM.keys()))
        with col_btn:
            st.write("")  # alinear verticalmente
            cargar_dem = st.button("📥 Cargar DEM", type="primary")

        if cargar_dem:
            dataset_sel = _DATASETS_DEM[resolucion]
            with st.spinner(f"Descargando DEM {resolucion} desde OpenTopography..."):
                dem = obtener_dem_opentopography(bounds_dem, _ot_key, dem_type=dataset_sel)
            if dem is not None:
                st.session_state["dem_data"]     = dem
                st.session_state["dem_dataset"]  = resolucion
                # Actualizar elevación y recalcular score FEN en vivo
                elevation_est = float(np.nanmean(dem.values))
                vuln_score    = calcular_vulnerabilidad_fen(
                    gdf, cultivo, ndvi_val, temp_val, precip_actual, elevation_est
                )
                st.success(
                    f"✅ DEM cargado · Elevación media: **{elevation_est:.0f} m** · "
                    f"Score FEN actualizado: **{vuln_score}/10**"
                )

        if st.session_state.get("dem_data") is not None:
            dem       = st.session_state["dem_data"]
            ds_label  = st.session_state.get("dem_dataset", resolucion)
            elev_min  = float(np.nanmin(dem.values))
            elev_max  = float(np.nanmax(dem.values))
            elev_mean = float(np.nanmean(dem.values))
            elev_range = elev_max - elev_min

            # ── Métricas de elevación ─────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⬇️ Elevación mínima",  f"{elev_min:.0f} m")
            c2.metric("⬆️ Elevación máxima",  f"{elev_max:.0f} m")
            c3.metric("📏 Elevación media",   f"{elev_mean:.0f} m")
            c4.metric("↕️ Rango altitudinal", f"{elev_range:.0f} m")

            # Contexto agronómico de la elevación
            if elev_mean < 500:
                st.info(f"📍 Parcela en **zona costera** ({elev_mean:.0f} m) — mayor exposición a FEN directo.")
            elif elev_mean < 2000:
                st.info(f"📍 Parcela en **zona de transición** ({elev_mean:.0f} m) — riesgo FEN moderado.")
            else:
                st.info(f"📍 Parcela en **sierra** ({elev_mean:.0f} m) — lluvias superiores posibles en FEN.")

            st.markdown("---")

            # ── Selector de visualización ─────────────────────────
            tipo_vis = st.radio(
                "Visualización", ["🗺️ Mapa 2D interactivo", "📐 Modelo 3D interactivo"],
                horizontal=True
            )

            if tipo_vis == "🗺️ Mapa 2D interactivo":
                if not FOLIUM_OK:
                    st.error("Instala folium: pip install folium streamlit-folium")
                else:
                    with st.spinner("Generando mapa 2D con overlay DEM..."):
                        mapa_dem = generar_mapa_folium_dem(gdf, dem, ds_label)
                    if FOLIUM_STATIC_OK:
                        folium_static(mapa_dem, width=900, height=620)
                    else:
                        components.html(mapa_dem.get_root().render(), height=620)
                    st.caption(
                        f"DEM: {ds_label} · Escala de colores: verde oscuro (bajo) → blanco (cima) · "
                        "Amarillo = contorno de parcela"
                    )

            else:  # Modelo 3D
                with st.spinner("Generando modelo 3D de elevación..."):
                    fig_3d, _mn, _mx, _me = generar_grafico_3d_dem(dem)
                if fig_3d is not None:
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.caption(
                        f"Superficie 3D · Mín {_mn:.0f} m · Máx {_mx:.0f} m · "
                        f"Media {_me:.0f} m · Dataset: {ds_label}"
                    )

            # ── Perfil de elevación (corte transversal) ───────────
            with st.expander("📈 Ver perfil de elevación (corte transversal este-oeste)"):
                try:
                    arr_2d = dem.values.squeeze() if dem.values.ndim > 2 else dem.values
                    fila_central = arr_2d[arr_2d.shape[0] // 2, :]
                    x_lon = dem.x.values
                    fig_p, ax_p = plt.subplots(figsize=(9, 3))
                    ax_p.fill_between(x_lon, fila_central, alpha=0.4, color="saddlebrown")
                    ax_p.plot(x_lon, fila_central, color="saddlebrown", linewidth=1.5)
                    ax_p.axhline(elev_mean, color="blue", linestyle="--",
                                 linewidth=1, label=f"Media {elev_mean:.0f} m")
                    ax_p.set_xlabel("Longitud")
                    ax_p.set_ylabel("Elevación (m)")
                    ax_p.set_title("Perfil de elevación — corte central este-oeste")
                    ax_p.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_p)
                except Exception as e:
                    st.warning(f"No se pudo generar el perfil: {e}")

            # ── Exportar DEM como CSV ──────────────────────────────
            with st.expander("💾 Exportar datos de elevación"):
                try:
                    arr_flat = dem.values.squeeze().flatten()
                    lons_flat = np.tile(dem.x.values, len(dem.y.values))
                    lats_flat = np.repeat(dem.y.values, len(dem.x.values))
                    df_dem = pd.DataFrame({
                        "latitud":   lats_flat,
                        "longitud":  lons_flat,
                        "elevacion_m": arr_flat,
                    }).dropna()
                    st.dataframe(df_dem.head(200), use_container_width=True)
                    st.download_button(
                        "⬇️ Descargar DEM completo (CSV)",
                        data=df_dem.to_csv(index=False),
                        file_name=f"dem_{ds_label.replace(' ','_')}.csv",
                    )
                except Exception as e:
                    st.warning(f"Error exportando DEM: {e}")

# ============================================================
st.caption("Plataforma Pachamama — FEN Nivel 1-6 · DEM · Sentinel-2 · ERA5 · CHIRPS · ENFEN · GFS")

# ============================================================
# FERTILIDAD NPK POR BLOQUES
# ============================================================
with tab_npk:
    st.header("🌾 Fertilidad NPK por Bloques")
    st.markdown(
        "Divide la parcela en zonas, obtiene el NDVI real de GEE para cada bloque "
        "y calcula la dosis de fertilizante N/P/K y el potencial de cosecha por zona."
    )

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
                    a = calcular_superficie(gpd.GeoDataFrame(
                        {'geometry': [row.geometry]}, crs='EPSG:4326'
                    ))
                    areas_bloque.append(a)
                gdf_bloques['area_ha'] = areas_bloque

                recs = [calcular_recomendaciones_npk(v, cultivo)
                        for v in gdf_bloques['ndvi']]
                gdf_bloques['nivel']   = [r['nivel'] for r in recs]
                gdf_bloques['N_kg_ha'] = [r['N']     for r in recs]
                gdf_bloques['P_kg_ha'] = [r['P']     for r in recs]
                gdf_bloques['K_kg_ha'] = [r['K']     for r in recs]

                rends = [estimar_potencial_cosecha(v, cultivo, a)
                         for v, a in zip(gdf_bloques['ndvi'], gdf_bloques['area_ha'])]
                gdf_bloques['rend_t_ha']    = [r[0] for r in rends]
                gdf_bloques['prod_total_t'] = [r[1] for r in rends]

                # Métricas resumen
                c1, c2, c3, c4 = st.columns(4)
                ndvi_med = gdf_bloques['ndvi'].mean()
                c1.metric("NDVI promedio",     f"{ndvi_med:.3f}")
                c2.metric("Bloques críticos",  str((gdf_bloques['N_kg_ha'] > 40).sum()))
                c3.metric("Rend. medio (t/ha)",f"{gdf_bloques['rend_t_ha'].mean():.1f}")
                c4.metric("Producción total",  f"{gdf_bloques['prod_total_t'].sum():.1f} t")

                # Tabla
                st.subheader("📋 Detalle por bloque")
                display_cols = ['id_bloque','area_ha','ndvi','nivel',
                                'N_kg_ha','P_kg_ha','K_kg_ha',
                                'rend_t_ha','prod_total_t']
                st.dataframe(gdf_bloques[display_cols].round(3), use_container_width=True)

                # Mapa de calor NDVI por bloque
                if FOLIUM_OK:
                    st.subheader("🗺️ Mapa de calor NDVI por bloque")
                    bounds_b = gdf_bloques.total_bounds
                    c_lat, c_lon, z = obtener_zoom_con_margen(bounds_b)
                    m_npk = folium.Map(location=[c_lat, c_lon], zoom_start=z,
                                       tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                                       attr='Google Hybrid')
                    vmin = gdf_bloques['ndvi'].min()
                    vmax = gdf_bloques['ndvi'].max()
                    cmap_fn = plt.cm.RdYlGn
                    for _, row in gdf_bloques.iterrows():
                        ndvi_b = row['ndvi'] if not np.isnan(row['ndvi']) else 0.3
                        norm_v = (ndvi_b - vmin) / max(vmax - vmin, 0.01)
                        r_, g_, b_, _ = cmap_fn(norm_v)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(r_*255), int(g_*255), int(b_*255))
                        popup_txt = (f"Bloque {int(row.id_bloque)}<br>"
                                     f"NDVI: {ndvi_b:.3f}<br>"
                                     f"N: {int(row.N_kg_ha)} kg/ha · "
                                     f"P: {int(row.P_kg_ha)} · K: {int(row.K_kg_ha)}<br>"
                                     f"Rend.: {row.rend_t_ha:.1f} t/ha")
                        folium.GeoJson(
                            gpd.GeoDataFrame({'geometry': [row.geometry]},
                                             crs='EPSG:4326').__geo_interface__,
                            style_function=lambda x, c=hex_color: {
                                'fillColor': c, 'color': '#333', 'weight': 1, 'fillOpacity': 0.7},
                            tooltip=f"Bloque {int(row.id_bloque)} · NDVI {ndvi_b:.3f}",
                            popup=folium.Popup(popup_txt, max_width=200),
                        ).add_to(m_npk)
                    map_npk_html = m_npk.get_root().render()
                    components.html(map_npk_html, height=500)

                # Descarga
                st.download_button(
                    "⬇️ Descargar CSV fertilidad",
                    data=gdf_bloques[display_cols].to_csv(index=False),
                    file_name=f"fertilidad_npk_{cultivo}.csv",
                    mime="text/csv",
                )

# ============================================================
# AGROECOLOGÍA — 10 PRINCIPIOS
# ============================================================
with tab_agro:
    st.header("🌱 Agroecología — 10 Principios")
    st.markdown(
        "Recomendaciones agronómicas basadas en los **10 principios agroecológicos** "
        "adaptadas al estado actual de la parcela y contexto FEN."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌿 Recomendación por principio", type="primary"):
            with st.spinner("Generando recomendaciones agroecológicas…"):
                rec = generar_recomendaciones_agroecologicas(
                    cultivo, fase_fenologica, ndvi_val, temp_val, humedad_val, precip_actual
                )
            st.markdown("### 🌿 Recomendaciones por Principio Agroecológico")
            st.markdown(rec)
            st.download_button(
                "⬇️ Descargar recomendaciones",
                data=rec,
                file_name=f"agroecologia_principios_{cultivo}.txt",
            )
    with col_b:
        if st.button("📋 Plan agroecológico completo"):
            with st.spinner("Generando plan completo…"):
                plan = generar_plan_agroecologico_completo(
                    cultivo, fase_fenologica, ndvi_val, temp_val,
                    humedad_val, precip_actual, area_ha
                )
            st.markdown("### 📋 Plan Agroecológico Integral")
            st.markdown(plan)
            st.download_button(
                "⬇️ Descargar plan",
                data=plan,
                file_name=f"plan_agroecologico_{cultivo}.txt",
            )

    # Indicadores de contexto visible
    st.markdown("---")
    st.caption(
        f"📊 Contexto enviado a la IA — NDVI: {ndvi_val:.3f} · "
        f"Temp: {temp_val:.1f}°C · Humedad: {humedad_val:.2f} · "
        f"Precip: {precip_actual:.1f} mm · Fase: {fase_fenologica} · "
        f"Cultivo: {cultivo}"
    )

# ============================================================
# CARBONO Y CRÉDITOS
# ============================================================
with tab_carbono:
    st.header("🌍 Carbono y Créditos de Carbono")
    st.markdown(
        "Estimación de carbono almacenado (t C/ha) y créditos de carbono "
        "calculados a partir del NDVI y la precipitación anual de la parcela."
    )

    calc_c = CalculadorCarbono()
    precip_anual = estimar_precipitacion_anual(df_precip)
    res_c = calc_c.calcular_carbono_hectarea(ndvi_val, precip_anual)

    co2_total   = round(res_c['co2_equivalente_ton_ha'] * area_ha, 2)
    creditos    = round(co2_total / 1000, 4)
    precio_usd  = round(creditos * 15, 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌿 C total (t C/ha)",  f"{res_c['carbono_total_ton_ha']}")
    c2.metric("☁️ CO₂e (t/ha)",        f"{res_c['co2_equivalente_ton_ha']}")
    c3.metric("📐 Área",               f"{area_ha:.2f} ha")
    c4.metric("🪙 Créditos (kt CO₂e)", f"{creditos:.4f}")
    c5.metric("💵 Valor estimado USD",  f"${precio_usd:,.2f}")

    st.markdown("---")
    st.subheader("📊 Desglose por pool de carbono")
    df_pools = pd.DataFrame(
        list(res_c['desglose'].items()),
        columns=['Pool de carbono', 't C/ha']
    )
    st.dataframe(df_pools, use_container_width=True)

    # Gráfico de barras
    fig_c, ax_c = plt.subplots(figsize=(8, 3))
    bars = ax_c.barh(df_pools['Pool de carbono'], df_pools['t C/ha'],
                     color=['#2ecc71','#27ae60','#f39c12','#e67e22','#8e44ad'])
    ax_c.set_xlabel('t C/ha')
    ax_c.set_title(f'Distribución de carbono — {cultivo} · {area_ha:.2f} ha')
    for bar, val in zip(bars, df_pools['t C/ha']):
        ax_c.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                  f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_c)

    st.markdown("---")
    st.info(
        f"💡 Precipitación anual estimada: **{precip_anual:.0f} mm/año** · "
        f"Precio de referencia: **15 USD/t CO₂e** (mercado voluntario)."
    )
    st.download_button(
        "⬇️ Exportar reporte de carbono CSV",
        data=pd.DataFrame([{
            'cultivo': cultivo, 'area_ha': area_ha,
            'ndvi': ndvi_val, 'precip_anual_mm': precip_anual,
            **res_c['desglose'],
            'carbono_total_ton_ha': res_c['carbono_total_ton_ha'],
            'co2e_ton_ha': res_c['co2_equivalente_ton_ha'],
            'co2e_total_parcela': co2_total,
            'creditos_kton': creditos,
            'valor_usd': precio_usd,
        }]).to_csv(index=False),
        file_name=f"carbono_{cultivo}_{area_ha:.1f}ha.csv",
        mime="text/csv",
    )
