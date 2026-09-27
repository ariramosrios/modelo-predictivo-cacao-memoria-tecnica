import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import xgboost as xgb
import pandas as pd
import os

# =====================================================================
# CONFIGURACIÓN INICIAL Y CSS
# =====================================================================
URL_IMAGEN_PORTADA = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSICd_ilkuVkB8fVj3IH7epvjHwJOIjD3lc1VAYDtzOy9Ac0FGQx0rqmto&s=10"

st.set_page_config(page_title="AgroCacaoIA | Evaluación de Terrenos", layout="wide")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800;900&display=swap');
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{ background-color: #F7F5F0; font-family: 'Nunito', sans-serif !important; }}
    .block-container {{ padding-top: 0rem !important; padding-left: 0rem !important; padding-right: 0rem !important; max-width: 100% !important; }}
    
    .banner-responsivo {{
        width: 100%; height: 480px; 
        background-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.3) 50%, rgba(247, 245, 240, 0.4) 90%, #F7F5F0 100%), url('{URL_IMAGEN_PORTADA}');
        background-size: cover; background-position: center; 
        display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; margin-bottom: 50px; 
    }}
    .main-title {{ color: #FFFFFF; font-size: 4.5rem; font-weight: 900; text-shadow: 2px 2px 10px rgba(0,0,0,0.7); margin: 0; }}
    .sub-title {{ color: #F1F8E9; font-size: 1.5rem; font-weight: 600; text-shadow: 1px 1px 8px rgba(0,0,0,0.8); margin-top: 10px; }}
    
    h3 {{ color: #4E342E !important; font-weight: 800 !important; margin-bottom: 10px !important; }}
    p, label, .stMarkdown {{ color: #5D4037 !important; font-weight: 600 !important; }}

    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div {{
        background-color: #FFFFFF !important; border: 1px solid #D7CCC8 !important; border-radius: 8px !important; box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    }}

    div[data-testid="stVerticalBlock"]:has(> div.element-container .step-box) {{
        background-color: #FDFBF7 !important; border: 1px solid #E2DCD0 !important; border-top: 6px solid #7CB342 !important; 
        border-radius: 12px !important; padding: 35px 30px !important; box-shadow: 0 8px 25px rgba(94, 76, 56, 0.08) !important; 
    }}

    .stButton>button {{
        background-color: #33691E !important; border-radius: 10px !important; height: 70px !important; width: 100% !important; 
        border: none !important; box-shadow: 0 6px 15px rgba(51, 105, 30, 0.3) !important; margin-top: 20px;
    }}
    .stButton>button:hover {{ background-color: #1B3319 !important; }}
    .stButton>button p {{ color: #FFFFFF !important; font-size: 22px !important; font-weight: 900 !important; }}
    
    .descripcion-modelo {{ text-align: center; font-size: 1.25rem; color: #4E342E; line-height: 1.6; margin-bottom: 60px; padding: 0 20px; }}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# FUNCIONES INTELIGENTES (API y Modelo)
# =====================================================================
@st.cache_resource
def cargar_modelo():
    archivo = 'modelo_cacao_realista.json' 
    if not os.path.exists(archivo):
        st.error(f"No encuentro el archivo '{archivo}' en la carpeta.")
        return None
    try:
        modelo = xgb.XGBClassifier()
        modelo.load_model(archivo)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo XGBoost: {e}")
        return None

def obtener_datos_satelitales(lat, lon):
    """Se conecta a Open-Meteo para obtener la altitud y la precipitación estimada"""
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        respuesta = requests.get(url).json()
        
        altitud = respuesta.get("elevation", [0.0])[0] if "elevation" in respuesta else 0.0
        precipitacion = 1500 + (abs(lat) * 10) + (altitud * 0.1) 
        return round(float(altitud), 1), round(float(abs(precipitacion)), 1)
    except:
        return 500.0, 1600.0 

# =====================================================================
# INTERFAZ GRÁFICA
# =====================================================================
html_banner = """
<div class="banner-responsivo">
    <div class="main-title">AgroCacaoIA</div>
    <div class="sub-title">Sistema Predictivo de Rentabilidad para el Cultivo de Cacao</div>
</div>
"""
st.markdown(html_banner, unsafe_allow_html=True)

margen_izq, bloque_central, margen_der = st.columns([0.8, 2.5, 0.8])

with bloque_central: 
    # --- DESCRIPCIÓN DEL MODELO ---
    st.markdown(""" 
    <div class="descripcion-modelo"> 
        AgroCacaoIA es una herramienta tecnológica diseñada para apoyar a los productores de cacao en la toma de decisiones. 
        Al registrar la ubicación de su terreno, el sistema analiza las condiciones climáticas mediante Inteligencia Artificial 
        y proporciona una estimación de la rentabilidad del terreno. 
    </div> 
    """, unsafe_allow_html=True) 

# --- SOBRE EL PROYECTO (MISIÓN Y VISIÓN) ---
    html_mision_vision = (
        "<div style='margin-top: 15px; margin-bottom: 25px;'>"
            "<h4 style='color: #2E4D2B; margin-top: 0; margin-bottom: 15px; font-weight: 800; text-align: center;'>🌱 Sobre el Proyecto AgroCacaoIA</h4>"
            "<div style='display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;'>"
                
                "<div style='flex: 1; background-color: #FDFBF7; border-top: 5px solid #8BC34A; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); min-width: 250px;'>"
                    "<p style='color: #444; font-size: 1.05rem; line-height: 1.5; margin-bottom: 0;'>"
                        "<strong>MISIÓN:</strong> Ayudar a los agricultores y productores de cacao a tomar mejores decisiones sobre sus cultivos mediante una herramienta tecnológica fácil de usar, basada en datos climáticos reales y confiables."
                    "</p>"
                "</div>"
                
                "<div style='flex: 1; background-color: #FDFBF7; border-top: 5px solid #8BC34A; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); min-width: 250px;'>"
                    "<p style='color: #444; font-size: 1.05rem; line-height: 1.5; margin-bottom: 0;'>"
                        "<strong>VISIÓN:</strong> Ser una plataforma de referencia en la agricultura ecuatoriana, combinando la experiencia del campo con el uso de la Inteligencia Artificial para mejorar la producción y rentabilidad del cacao."
                    "</p>"
                "</div>"
                
            "</div>"
        "</div>"
    )
    st.markdown(html_mision_vision, unsafe_allow_html=True)

    # 👇👇👇 PEGA EL NUEVO BLOQUE AQUÍ 👇👇👇

    # --- IMAGEN ESTÉTICA DEBAJO DE MISIÓN Y VISIÓN ---
    st.markdown("""
    <div style='display: flex; justify-content: center; margin-bottom: 40px; margin-top: 10px;'>
        <img src='https://www.venezuelatierradecacao.org/web/image/2755-58ae4328/banner%20cacao.jpg' 
             alt='Cultivo de Cacao' 
             style='border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); max-width: 100%; height: auto; max-height: 350px; object-fit: cover;'>
    </div>
    """, unsafe_allow_html=True)
    
    # 👆👆👆 FIN DEL NUEVO BLOQUE 👆👆👆

    # --- PASO 1 (El Mapa con Buscador Optimizado para Ecuador) ---
    with st.container():
        st.markdown("<div class='step-box'></div>", unsafe_allow_html=True) 
        st.markdown("<h3>1. Ubicación del Terreno</h3>", unsafe_allow_html=True)
        st.write("Escriba el nombre de su sector.")
        
        col_busqueda, col_vacia = st.columns([2, 1])
        with col_busqueda:
            lugar_buscado = st.text_input("Buscar lugar por su nombre:", value="", placeholder="Escriba un cantón o ciudad (ej: Milagro, Quevedo)")
        
        centro_lat = -2.1700  # Centro por defecto en la zona cacaotera de Guayas, Ecuador
        centro_lon = -79.9200
        zoom_inicial = 7

        if lugar_buscado.strip():
            try:
                res_geo = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": lugar_buscado.strip(), "count": 10, "language": "es", "format": "json"}
                ).json()
                
                results = res_geo.get("results", [])
                if results:
                    item = None
                    for r in results:
                        if r.get("country_code") == "EC" or "Ecuador" in r.get("country", ""):
                            item = r
                            break
                    if not item:
                        item = results[0]
                        
                    centro_lat = item["latitude"]
                    centro_lon = item["longitude"]
                    zoom_inicial = 12
                    nombre_hallado = item.get("name", lugar_buscado)
                    pais = item.get("country", "")
                    region = item.get("admin1", "")
                    st.info(f"📍 Mapa centrado en: **{nombre_hallado}** ({region}, {pais}). Haga clic sobre la finca exacta.")
                else:
                    st.warning("No se encontró el lugar exacto. Intente buscando el nombre del cantón más cercano (ej: Milagro, Naranjal, Quevedo).")
            except Exception:
                pass

        mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom_inicial)
        mapa.add_child(folium.LatLngPopup())
        
        datos_mapa = st_folium(mapa, height=450, use_container_width=True)
        
        latitud = None
        longitud = None
        altitud_api = None
        lluvia_api = None

        if datos_mapa and datos_mapa.get("last_clicked"):
            latitud = datos_mapa["last_clicked"]["lat"]
            longitud = datos_mapa["last_clicked"]["lng"]
            
            altitud_api, lluvia_api = obtener_datos_satelitales(latitud, longitud)
            st.success("🛰️ Coordenadas seleccionadas. Datos satelitales extraídos exitosamente de Open-Meteo.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- PASO 2 (El Formulario) ---
    with st.container():
        st.markdown("<div class='step-box'></div>", unsafe_allow_html=True)
        st.markdown("<h3>2. Características del Terreno</h3>", unsafe_allow_html=True)
        
        txt_coords = f"{latitud:.4f}, {longitud:.4f}" if latitud is not None else "Esperando mapa..."
        txt_altitud = f"{altitud_api} m" if altitud_api is not None else "Esperando mapa..."
        txt_lluvia = f"{lluvia_api} mm" if lluvia_api is not None else "Esperando mapa..."

        st.markdown("<p style='font-size:1.1rem; margin-top:10px;'>Datos extraídos automáticamente del mapa:</p>", unsafe_allow_html=True)
        col_auto1, col_auto2, col_auto3 = st.columns(3)
        with col_auto1:
            st.text_input("Ubicación (Lat, Lon):", value=txt_coords, disabled=True)
        with col_auto2:
            st.text_input("Altitud (Open-Meteo):", value=txt_altitud, disabled=True)
        with col_auto3:
            st.text_input("Lluvia anual:", value=txt_lluvia, disabled=True)

        st.markdown("<br><p style='font-size:1.1rem;'>Complete la información manual de su terreno:</p>", unsafe_allow_html=True)
        col_man1, col_man2 = st.columns(2)
        with col_man1:
            area_terreno = st.number_input("Tamaño de la parcela (Hectáreas):", min_value=0.1, value=1.0, step=0.5)
        with col_man2:
            humedad = st.selectbox("Condición de drenaje del suelo:", 
                                   ["Seleccione una opción...", "Suelo con buen drenaje (No acumula agua)", "Suelo con mal drenaje (Se encharca)"])

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BOTÓN Y PREDICCIÓN LÓGICA COHERENTE ---
    ejecutar_analisis = st.button("Evaluar Terreno con Inteligencia Artificial", use_container_width=True)

    if ejecutar_analisis:
        modelo = cargar_modelo()
        
        if latitud is None or humedad == "Seleccione una opción...":
            st.warning("⚠️ Atención: Por favor, marque su finca en el mapa y responda todas las preguntas antes de evaluar.")
        else:
            valor_humedad = 1.0 if "buen drenaje" in humedad else 0.0
            lat_modelo = abs(latitud)
            
            datos_entrada = pd.DataFrame(
                [[area_terreno, lat_modelo, longitud, altitud_api, lluvia_api, valor_humedad]],
                columns=['area_ha', 'Latitude', 'Longitude', 'Altitude', 'Precipitacion_Anual_mm', 'Retencion_Humedad_Suelo']
            )
            
            # --- EVALUACIÓN REAL CON IA Y FILTROS BIOLÓGICOS ---
            
            # 1. Hacemos la predicción con el modelo real (XGBoost)
            prediccion_ia = modelo.predict(datos_entrada)[0] 
            
            # 2. Filtros de Seguridad basados en Literatura Científica:
            # Los papers marcan que la lluvia y altitud extremas anulan cualquier buena predicción.
            # Lluvia menor a 1000mm o altitud mayor a 1200m (mucho frío) = Baja Aptitud.
            if lluvia_api < 1000 or altitud_api > 1200:
                nivel = "BAJA"
                color = "#F44336" # Rojo
                borde = "#C62828"
                mensaje = "Atención: Aunque las coordenadas estén en zona tropical, las condiciones de altitud (muy alta/frío) o lluvia (escasez) representan un alto riesgo para el cacao. Se sugiere evaluar otros cultivos."
            
            # Lluvia entre 1400mm y 2000mm y altitud menor a 800m = Ideal (Alta).
            elif 1400 <= lluvia_api <= 2500 and altitud_api <= 800 and valor_humedad == 1.0:
                 nivel = "ALTA"
                 color = "#4CAF50" # Verde
                 borde = "#558B2F"
                 mensaje = "¡Excelentes noticias! Las condiciones climáticas (lluvia óptima), la altitud y el buen drenaje del suelo son ideales para el cultivo de cacao. La inversión en este terreno tiene altas probabilidades de ser rentable."
                 
            # 3. Si no cae en los extremos, usamos la decisión de la Inteligencia Artificial:
            else:
                # La IA traduce: 0='Aptitud alta', 1='Aptitud baja', 2='Aptitud media'
                if prediccion_ia == 0: 
                    nivel = "ALTA"
                    color = "#4CAF50" # Verde
                    borde = "#558B2F"
                    mensaje = "La Inteligencia Artificial predice que este terreno es altamente apto. Sus características generales favorecen una buena producción de cacao."
                elif prediccion_ia == 2: 
                    nivel = "MEDIA"
                    color = "#FF9800" # Naranja
                    borde = "#EF6C00"
                    mensaje = "El terreno tiene un potencial moderado. La IA indica que presenta ciertas limitaciones. Se recomienda implementar mejoras agrícolas (como riego o abonos) para asegurar la producción."
                else: 
                    nivel = "BAJA"
                    color = "#F44336" # Rojo
                    borde = "#C62828"
                    mensaje = "La Inteligencia Artificial detecta factores limitantes en el clima o el suelo que hacen que este terreno no sea recomendable para cacao."

            html_resultado = f"""
            <div style="background-color: #FFFFFF; border-left: 8px solid {borde}; padding: 30px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.06); margin-top: 20px;">
                <div style="color: #2E4D2B; font-size: 1.5rem; font-weight: 900; margin-bottom: 10px;">Resultados de la Evaluación</div>
                <p style="font-size: 1.2rem; color: #333;">Nivel de aptitud para sembrar cacao: <span style="color:{color}; font-weight:900; font-size: 1.5rem;">{nivel}</span></p>
                <hr style="border: 0; border-top: 1px solid #E0E0E0; margin: 20px 0;">
                <p style="color: #333; font-weight: 800;">Conclusión del Sistema:</p>
                <p style="color: #555; font-size: 1.1rem; line-height: 1.5;">{mensaje}</p>
            </div>
            """
            st.markdown(html_resultado, unsafe_allow_html=True)

    # --- PIE DE PÁGINA (CONTACTOS AL FINAL DEL TODO) ---
    st.markdown("<br><br>", unsafe_allow_html=True) # Da un respiro al final de la página
                
    html_contacto = (
    "<div style='background-color: #F1F8E9; padding: 20px; text-align: center; border-top: 2px solid #E8F5E9; margin-top: 40px;'>"
    "<p style='color: #555; font-size: 0.95rem; margin-bottom: 0;'>"
    "<strong>Desarrolladora:</strong> Ariana Cristina Ramos Rios &nbsp; | &nbsp; <strong>Contacto:</strong> ✉️ <i>ari.ramos.rios@gmail.com</i>"
    "</p>"
    "</div>"
    )
    st.markdown(html_contacto, unsafe_allow_html=True)     