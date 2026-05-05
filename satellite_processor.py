import numpy as np
import rasterio
from sentinelhub import (
    SHConfig, 
    BBox, 
    CRS, 
    DataCollection, 
    MimeType, 
    MosaickingOrder,
    SentinelHubRequest, 
    bbox_to_dimensions
)
from datetime import datetime, timedelta
import logging
import streamlit as st

class SatelliteProcessor:
    def __init__(self, config):
        self.config = config
        self.sh_config = SHConfig()
        self._setup_sentinelhub_config()
    
    def _setup_sentinelhub_config(self):
        """Configurar credenciales de Sentinel Hub desde secrets.toml"""
        try:
            if self.config and self.config['instance_id']:
                self.sh_config.instance_id = self.config['instance_id']
                self.sh_config.sh_client_id = self.config['client_id']
                self.sh_config.sh_client_secret = self.config['client_secret']
                
                # Verificar que la configuración sea válida
                if (self.sh_config.instance_id and 
                    self.sh_config.sh_client_id and 
                    self.sh_config.sh_client_secret):
                    st.success("🔑 Configuración de Sentinel Hub inicializada")
                    return True
                else:
                    st.error("❌ Configuración de Sentinel Hub incompleta")
                    return False
            else:
                st.error("❌ No se encontró configuración de Sentinel Hub")
                return False
                
        except Exception as e:
            st.error(f"❌ Error configurando Sentinel Hub: {str(e)}")
            return False
    
    def check_credentials(self):
        """Verificar que las credenciales sean válidas"""
        return (hasattr(self.sh_config, 'instance_id') and 
                self.sh_config.instance_id and
                hasattr(self.sh_config, 'sh_client_id') and 
                self.sh_config.sh_client_id and
                hasattr(self.sh_config, 'sh_client_secret') and 
                self.sh_config.sh_client_secret)
    
    def get_field_bbox(self, gdf):
        """Obtener bounding box de la parcela"""
        try:
            # Asegurarse de que esté en WGS84
            if gdf.crs != CRS.WGS84:
                gdf = gdf.to_crs(CRS.WGS84)
                
            bounds = gdf.total_bounds
            return BBox(bbox=bounds, crs=CRS.WGS84)
        except Exception as e:
            st.error(f"❌ Error obteniendo BBox: {str(e)}")
            return None
    
    def download_sentinel2_data(self, gdf, start_date, end_date, indices=['ndvi']):
        """Descargar datos de Sentinel-2 para la parcela"""
        try:
            # Verificar credenciales primero
            if not self.check_credentials():
                st.error("🔑 Credenciales de Sentinel Hub no configuradas")
                return None
            
            # Obtener bounding box
            bbox = self.get_field_bbox(gdf)
            if bbox is None:
                return None
                
            # Configurar resolución y tamaño
            resolution = 10  # metros
            size = bbox_to_dimensions(bbox, resolution=resolution)
            
            st.info(f"📍 Área de descarga: {size} píxeles")
            
            # Evalscript simplificado para NDVI
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B04", "B08"],
                        units: "REFLECTANCE"
                    }],
                    output: {
                        id: "ndvi",
                        bands: 1,
                        sampleType: "FLOAT32"
                    }
                };
            }
            
            function evaluatePixel(sample) {
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                return [ndvi];
            }
            """
            
            # Crear request
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(start_date, end_date),
                        mosaicking_order=MosaickingOrder.LEAST_CC
                    )
                ],
                responses=[SentinelHubRequest.output_response('ndvi', MimeType.TIFF)],
                bbox=bbox,
                size=size,
                config=self.sh_config
            )
            
            # Descargar datos
            with st.spinner("📡 Descargando datos de Sentinel-2..."):
                data = request.get_data()
                
            if data and len(data) > 0:
                st.success(f"✅ Datos descargados: {data[0].shape if hasattr(data[0], 'shape') else 'N/A'}")
                return data[0]
            else:
                st.error("❌ No se recibieron datos de Sentinel Hub")
                return None
                
        except Exception as e:
            st.error(f"❌ Error descargando datos Sentinel-2: {str(e)}")
            return None
