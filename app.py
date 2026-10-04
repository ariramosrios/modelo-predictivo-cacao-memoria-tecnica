import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import xgboost as xgb
import pandas as pd
import os
import io
import math
import time
from PIL import Image

# =====================================================================
# CONFIGURACIÓN INICIAL Y CSS
# =====================================================================
URL_IMAGEN_PORTADA = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSICd_ilkuVkB8fVj3IH7epvjHwJOIjD3lc1VAYDtzOy9Ac0FGQx0rqmto&s=10"
SEGUNDOS_ALERTA = 5  # Tiempo mínimo que la alerta de agua permanece abierta antes de poder cerrarse

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
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stSelectbox"] div:has(> input) {{
        background-color: #FFFFFF !important; border: 1px solid #D7CCC8 !important; border-radius: 8px !important; box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    }}
    /* Texto oscuro dentro de los campos (si no, hereda el blanco del tema oscuro del navegador) */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div {{
        color: #4E342E !important; -webkit-text-fill-color: #4E342E !important; caret-color: #4E342E !important;
    }}
    div[data-testid="stTextInput"] input::placeholder {{ color: #A1887F !important; -webkit-text-fill-color: #A1887F !important; }}
    div[data-testid="stTextInput"] input:disabled {{ color: #6D4C41 !important; -webkit-text-fill-color: #6D4C41 !important; }}
    div[data-testid="stNumberInput"] button, div[data-testid="stSelectbox"] svg {{ color: #5D4037 !important; fill: #5D4037 !important; }}
    /* Lista desplegable del selector */
    div[data-testid="stSelectboxVirtualDropdown"], div[data-baseweb="popover"] ul {{
        background-color: #FFFFFF !important; border: 1px solid #D7CCC8 !important;
    }}
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"], div[data-baseweb="popover"] li {{
        color: #4E342E !important; background-color: #FFFFFF !important;
    }}
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
    div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {{
        background-color: #F1F8E9 !important;
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

# --- Estilos del modal de alerta de agua ---
# El botón de cierre queda bloqueado (y con cuenta regresiva) durante SEGUNDOS_ALERTA
# para que el usuario lea el aviso. Las animaciones reinician cada vez que el modal se abre.
st.markdown(f"""
<style>
    @property --seg-restantes {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
    @keyframes cuenta-regresiva {{ from {{ --seg-restantes: {SEGUNDOS_ALERTA}; }} to {{ --seg-restantes: 0; }} }}
    @keyframes barra-espera {{ from {{ transform: scaleX(1); }} to {{ transform: scaleX(0); }} }}
    @keyframes habilitar-cierre {{ to {{ opacity: 1; pointer-events: auto; cursor: pointer; }} }}
    @keyframes ocultar {{ to {{ opacity: 0; height: 0; margin: 0; }} }}

    /* Fondo claro fijo: sin esto el modal hereda el tema oscuro del navegador y el texto pierde contraste */
    div[data-testid="stDialog"] > div:has(> section[role="dialog"]) {{
        background-color: #FFFFFF !important; border-top: 6px solid #1E88E5 !important;
    }}
    .alerta-agua {{ text-align: center; }}
    .alerta-agua .icono {{
        width: 72px; height: 72px; margin: 0 auto 14px; border-radius: 50%; background: #E3F2FD;
        display: flex; align-items: center; justify-content: center;
    }}
    .alerta-agua .titulo {{ color: #0D47A1; font-size: 1.35rem; font-weight: 900; margin-bottom: 8px; }}
    .alerta-agua .texto {{ color: #37474F; font-size: 1.02rem; line-height: 1.5; margin-bottom: 14px; }}
    .alerta-agua .coords {{
        display: inline-block; background: #ECEFF1; color: #263238; border-radius: 6px;
        padding: 4px 10px; font-family: monospace; font-size: 0.95rem; margin-bottom: 14px;
    }}
    .alerta-agua ul {{ text-align: left; color: #455A64; font-size: 0.95rem; margin: 0 0 16px 0; padding-left: 22px; }}
    .alerta-agua .espera {{ overflow: hidden; animation: ocultar 0.3s ease {SEGUNDOS_ALERTA}s forwards; }}
    .alerta-agua .espera-texto {{ color: #78909C; font-size: 0.85rem; margin-bottom: 6px; }}
    .alerta-agua .espera-texto::after {{
        counter-reset: seg var(--seg-restantes); content: counter(seg) " s";
        animation: cuenta-regresiva {SEGUNDOS_ALERTA}s steps({SEGUNDOS_ALERTA}, end) forwards;
    }}
    .alerta-agua .barra {{ height: 6px; background: #CFD8DC; border-radius: 3px; overflow: hidden; }}
    .alerta-agua .barra > div {{
        height: 100%; background: #1E88E5; transform-origin: left;
        animation: barra-espera {SEGUNDOS_ALERTA}s linear forwards;
    }}

    .st-key-btn_cerrar_alerta_agua button {{
        height: 52px !important; margin-top: 4px !important;
        opacity: 0.45; pointer-events: none; cursor: not-allowed;
        animation: habilitar-cierre 0s linear {SEGUNDOS_ALERTA}s forwards;
    }}
    .st-key-btn_cerrar_alerta_agua button p {{ font-size: 17px !important; }}
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

HEADERS_OSM = {'User-Agent': 'AgroCacaoIA/1.0 (evaluacion de terrenos de cacao)'}
COLOR_AGUA_OSM = (170, 211, 223)   # #aad3df: mar, rios y lagos en el mapa base de OpenStreetMap
COLOR_PLAYA_OSM = (255, 241, 186)  # #fff1ba: playas

def _es_agua_segun_mapa(lat, lon, zoom=17, radio=3):
    """Lee el color del mosaico de OpenStreetMap (el mismo mapa que ve el usuario) en el punto clicado.
    Es preciso a pocos metros de la orilla, a diferencia de la geocodificación inversa,
    que en el mar cercano a la costa devuelve la calle o edificio más próximo."""
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    tile_x, tile_y = int(x), int(y)
    px, py = int((x - tile_x) * 256), int((y - tile_y) * 256)

    respuesta = requests.get(f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png", headers=HEADERS_OSM, timeout=8)
    respuesta.raise_for_status()
    imagen = Image.open(io.BytesIO(respuesta.content)).convert("RGB")

    # Tolerancia mínima: las calles secundarias (#f7fabf) tienen un tono muy parecido al de las playas
    def es_color_agua(color):
        return any(all(abs(c - r) <= 3 for c, r in zip(color, ref)) for ref in (COLOR_AGUA_OSM, COLOR_PLAYA_OSM))

    # Se revisa una pequeña ventana alrededor del punto para tolerar etiquetas o líneas dibujadas encima del agua
    muestras = [
        imagen.getpixel((min(255, max(0, px + dx)), min(255, max(0, py + dy))))
        for dx in range(-radio, radio + 1) for dy in range(-radio, radio + 1)
    ]
    proporcion_agua = sum(es_color_agua(c) for c in muestras) / len(muestras)
    return es_color_agua(imagen.getpixel((px, py))) or proporcion_agua >= 0.5

def _es_agua_segun_nominatim(lat, lon):
    """Respaldo si no se pudo descargar el mapa: clasificación del lugar más cercano según Nominatim"""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    respuesta = requests.get(url, headers=HEADERS_OSM, timeout=5).json()

    if 'error' in respuesta:
        return True

    clase = respuesta.get('class', '')
    tipo = respuesta.get('type', '')

    if clase == 'natural' and tipo in ['water', 'bay', 'strait', 'coastline', 'beach', 'sea', 'ocean', 'wetland']:
        return True
    if clase == 'waterway':
        return True
    if clase == 'place' and tipo in ['sea', 'ocean']:
        return True

    return False

@st.cache_data(show_spinner=False, ttl=86400)
def es_zona_de_agua(lat, lon):
    """Verifica si las coordenadas proporcionadas caen en el agua (océanos, lagos, ríos, playas)"""
    try:
        return _es_agua_segun_mapa(lat, lon)
    except Exception:
        pass
    try:
        return _es_agua_segun_nominatim(lat, lon)
    except Exception:
        return False

def _crear_dialogo(titulo):
    # dismissible=False (Streamlit >= 1.46) evita cerrar el aviso con Esc, clic afuera o la X
    try:
        return st.dialog(titulo, dismissible=False)
    except TypeError:
        return st.dialog(titulo)

# IMPORTANTE: El ID del diálogo se mantiene idéntico para que Streamlit no pierda la referencia interna.
@_crear_dialogo("Zona no válida para el análisis")
def mostrar_alerta_agua(lat, lon):
    st.markdown(f"""
    <div class="alerta-agua">
        <div class="icono">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>
                <path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>
                <path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>
            </svg>
        </div>
        <div class="titulo">Ha seleccionado una zona de agua</div>
        <div class="texto">
            El punto marcado corresponde a un cuerpo de agua o zona costera (mar, río, lago o playa).
            El modelo solo puede evaluar terrenos en <b>tierra firme</b>, por lo que no se realizará ninguna predicción.
        </div>
        <div class="coords">{lat:.5f}, {lon:.5f}</div>
        <ul>
            <li>Acerque el mapa (zoom) para distinguir mejor la orilla.</li>
            <li>Haga clic dentro de su finca, alejado del agua.</li>
        </ul>
        <div class="espera">
            <div class="espera-texto">Lea el aviso. Podrá cerrarlo en </div>
            <div class="barra"><div></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Entendido, elegir otro punto", key="btn_cerrar_alerta_agua", use_container_width=True):
        # Respaldo del lado del servidor por si el boton se activa antes de tiempo (p. ej. con el teclado)
        if time.time() - st.session_state.get("alerta_agua_abierta_en", 0) >= SEGUNDOS_ALERTA:
            st.rerun()

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
            "<h4 style='color: #2E4D2B; margin-top: 0; margin-bottom: 15px; font-weight: 800; text-align: center;'>Sobre el Proyecto AgroCacaoIA</h4>"
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

    # --- IMAGEN ESTÉTICA DEBAJO DE MISIÓN Y VISIÓN ---
    st.markdown("""
    <div style='display: flex; justify-content: center; margin-bottom: 40px; margin-top: 10px;'>
        <img src='https://www.venezuelatierradecacao.org/web/image/2755-58ae4328/banner%20cacao.jpg' 
             alt='Cultivo de Cacao' 
             style='border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); max-width: 100%; height: auto; max-height: 350px; object-fit: cover;'>
    </div>
    """, unsafe_allow_html=True)

    # --- PASO 1 (El Mapa con Buscador Optimizado para Ecuador) ---
    with st.container():
        st.markdown("<div class='step-box'></div>", unsafe_allow_html=True) 
        st.markdown("<h3>1. Ubicación del Terreno</h3>", unsafe_allow_html=True)
        st.write("Escriba el nombre de su sector.")
        
        col_busqueda, col_vacia = st.columns([2, 1])
        with col_busqueda:
            lugar_buscado = st.text_input("Buscar lugar por su nombre:", value="", placeholder="Escriba un cantón o ciudad (ej: Milagro, Quevedo)")
        
        centro_lat = -2.1700
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
                    st.info(f"Mapa centrado en: **{nombre_hallado}** ({region}, {pais}). Haga clic sobre la finca exacta.")
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
            lat_click = datos_mapa["last_clicked"]["lat"]
            lon_click = datos_mapa["last_clicked"]["lng"]
            
            # st_folium devuelve el mismo last_clicked en cada recarga de la pagina, por eso se
            # recuerda el ultimo clic procesado: la alerta se muestra una vez por cada clic nuevo en agua
            click_actual = (lat_click, lon_click)
            es_click_nuevo = st.session_state.get('ultimo_click_mapa') != click_actual
            st.session_state['ultimo_click_mapa'] = click_actual

            with st.spinner("Verificando el punto seleccionado..."):
                punto_en_agua = es_zona_de_agua(lat_click, lon_click)

            if punto_en_agua:
                if es_click_nuevo:
                    st.session_state['alerta_agua_abierta_en'] = time.time()
                    mostrar_alerta_agua(lat_click, lon_click)
                st.error("El punto seleccionado está en una zona de agua o costera. Haga clic sobre tierra firme para continuar.")
                # No se asignan las variables latitud ni longitud, por lo que el modelo queda bloqueado
            else:
                latitud = lat_click
                longitud = lon_click
                altitud_api, lluvia_api = obtener_datos_satelitales(latitud, longitud)
                st.success("Coordenadas seleccionadas correctamente en tierra firme.")
    
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
        # IMPORTANTE: Mantengo los labels (IDs de Streamlit) de los inputs igual al original para no romper el programa
        with col_auto1:
            st.text_input("Ubicacion (Lat, Lon):", value=txt_coords, disabled=True)
        with col_auto2:
            st.text_input("Altitud (Open-Meteo):", value=txt_altitud, disabled=True)
        with col_auto3:
            st.text_input("Lluvia anual:", value=txt_lluvia, disabled=True)

        st.markdown("<br><p style='font-size:1.1rem;'>Complete la información manual de su terreno:</p>", unsafe_allow_html=True)
        col_man1, col_man2 = st.columns(2)
        # Etiquetas de inputs y selectores sin tilde para no alterar el estado subyacente
        with col_man1:
            area_terreno = st.number_input("Tamano de la parcela (Hectareas):", min_value=0.1, value=1.0, step=0.5)
        with col_man2:
            humedad = st.selectbox("Condicion de drenaje del suelo:", 
                                   ["Seleccione una opcion...", "Suelo con buen drenaje (No acumula agua)", "Suelo con mal drenaje (Se encharca)"])

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BOTÓN Y PREDICCIÓN LÓGICA COHERENTE ---
    ejecutar_analisis = st.button("Evaluar Terreno con Inteligencia Artificial", use_container_width=True)

    if ejecutar_analisis:
        modelo = cargar_modelo()
        
        # Validación lógica que usa estrictamente la variable y opción intactas
        if latitud is None or humedad == "Seleccione una opcion...":
            st.warning("Atención: Por favor, seleccione una zona de tierra válida en el mapa y responda todas las preguntas antes de evaluar.")
        else:
            valor_humedad = 1.0 if "buen drenaje" in humedad else 0.0
            lat_modelo = abs(latitud)
            
            datos_entrada = pd.DataFrame(
                [[area_terreno, lat_modelo, longitud, altitud_api, lluvia_api, valor_humedad]],
                columns=['area_ha', 'Latitude', 'Longitude', 'Altitude', 'Precipitacion_Anual_mm', 'Retencion_Humedad_Suelo']
            )
            
            prediccion_ia = modelo.predict(datos_entrada)[0] 
            
            if lluvia_api < 1000 or altitud_api > 1200:
                nivel = "BAJA"
                color = "#F44336" 
                borde = "#C62828"
                mensaje = "Atención: Aunque las coordenadas estén en zona tropical, las condiciones de altitud (muy alta/frío) o lluvia (escasez) representan un alto riesgo para el cacao. Se sugiere evaluar otros cultivos."
            
            elif 1400 <= lluvia_api <= 2500 and altitud_api <= 800 and valor_humedad == 1.0:
                 nivel = "ALTA"
                 color = "#4CAF50" 
                 borde = "#558B2F"
                 mensaje = "Excelentes noticias. Las condiciones climáticas (lluvia óptima), la altitud y el buen drenaje del suelo son ideales para el cultivo de cacao. La inversión en este terreno tiene altas probabilidades de ser rentable."
                 
            else:
                if prediccion_ia == 0: 
                    nivel = "ALTA"
                    color = "#4CAF50"
                    borde = "#558B2F"
                    mensaje = "La Inteligencia Artificial predice que este terreno es altamente apto. Sus características generales favorecen una buena producción de cacao."
                elif prediccion_ia == 2: 
                    nivel = "MEDIA"
                    color = "#FF9800"
                    borde = "#EF6C00"
                    mensaje = "El terreno tiene un potencial moderado. La IA indica que presenta ciertas limitaciones. Se recomienda implementar mejoras agrícolas (como riego o abonos) para asegurar la producción."
                else: 
                    nivel = "BAJA"
                    color = "#F44336"
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
    st.markdown("<br><br>", unsafe_allow_html=True)
                
    html_contacto = (
    "<div style='background-color: #F1F8E9; padding: 20px; text-align: center; border-top: 2px solid #E8F5E9; margin-top: 40px;'>"
    "<p style='color: #555; font-size: 0.95rem; margin-bottom: 0;'>"
    "<strong>Desarrolladora:</strong> Ariana Cristina Ramos Ríos &nbsp; | &nbsp; <strong>Contacto:</strong> <i>ari.ramos.rios@gmail.com</i>"
    "</p>"
    "</div>"
    )
    st.markdown(html_contacto, unsafe_allow_html=True)