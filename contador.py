import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Conteo de Frutos - YOLO", layout="centered")

# Estilos CSS para replicar la tabla de tu captura
st.markdown("""
<style>
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 15px;
    border-bottom: 1px solid #e6e6e6;
    font-size: 16px;
    align-items: center;
}
.metric-row:last-child {
    border-bottom: none;
}
.metric-row.header {
    font-weight: bold;
    background-color: #eaf2ea;
    border-radius: 10px 10px 0 0;
}
.metric-container {
    background-color: #f4fdf4;
    border-radius: 10px;
    padding: 5px;
    margin-top: 15px;
    margin-bottom: 20px;
}
.color-box {
    display: inline-block;
    width: 25px;
    height: 8px;
    border-radius: 4px;
    margin-right: 10px;
    vertical-align: middle;
}
/* Colores de las etiquetas */
.purple { background-color: #9C27B0; }
.green { background-color: #00FF00; }
.blue { background-color: #0000FF; }
.yellow { background-color: #FFEB3B; }
.pct-text { color: #2E7D32; } 
</style>
""", unsafe_allow_html=True)

# 2. Cargar el modelo YOLO
@st.cache_resource
def load_yolo_model():
    # Aquí está la modificación apuntando a tu modelo exacto
    return YOLO("arandano.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"No se pudo cargar el modelo. Asegúrate de que el archivo 'arandano.pt' esté en el repositorio. Error: {e}")
    st.stop()

# 3. Interfaz de usuario - Título
st.title("Resultados de Conteo")

# 4. Cargador de archivos
uploaded_file = st.file_uploader("Sube la imagen del cultivo de arándanos", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir la imagen subida con Pillow
    input_image = Image.open(uploaded_file)
    
    # Ejecutar la inferencia de YOLO
    # 'conf=0.25' es el umbral de confianza, puedes ajustarlo si detecta de más o de menos
    results = model(input_image, conf=0.25)
    
    # Obtener la imagen con las cajas de predicción dibujadas por YOLO
    res_plotted = results[0].plot()
    predicted_image = Image.fromarray(res_plotted[:, :, ::-1])
    
    # Mostrar la imagen con las detecciones
    st.image(predicted_image, caption="Resultado del análisis YOLO", use_container_width=True)
    
    # 5. Extracción y conteo de clases
    # Asegúrate de que estos nombres coincidan con las clases de tu dataset
    conteo_clases = {
        "Light purple": 0,
        "Green": 0,
        "Blue": 0,
        "Flower": 0
    }
    
    yolo_names = model.names 
    
    # Iterar sobre cada caja detectada
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = yolo_names[class_id]
            
            if class_name in conteo_clases:
                conteo_clases[class_name] += 1

    # 6. Cálculos de Totales y Porcentajes
    total_detectado = sum(conteo_clases.values())
    
    def calcular_porcentaje(valor, total):
        if total == 0: 
            return 0.0
        return round((valor / total) * 100, 1)

    # 7. Construcción de la tabla HTML
    html_content = f"""
    <div class="metric-container">
        <div class="metric-row header">
            <span style="flex:2;">Total</span>
            <span style="flex:1; text-align: left;">{total_detectado}</span>
            <span style="flex:1; text-align: right; color: #2E7D32;">%</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box purple"></span>Light purple</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Light purple']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Light purple'], total_detectado)}%)</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box green"></span>Green</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Green']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Green'], total_detectado)}%)</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box blue"></span>Blue</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Blue']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Blue'], total_detectado)}%)</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box yellow"></span>Flower</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Flower']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Flower'], total_detectado)}%)</span>
        </div>
    </div>
    """
    
    # Mostrar la tabla
    st.markdown(html_content, unsafe_allow_html=True)
    
    # 8. Botones inferiores
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start again", use_container_width=True, type="primary"):
            st.rerun()
    with col2:
        if st.button("Exit", use_container_width=True):
            st.info("Aplicación en espera de nueva actividad.")
