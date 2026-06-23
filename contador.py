import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Conteo de Arándanos", layout="centered")

# Estilos CSS
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
/* Colores asignados a tus nuevas categorías */
.purple { background-color: #9C27B0; } /* Para Pintón */
.green { background-color: #00FF00; }  /* Para Verde */
.blue { background-color: #0000FF; }   /* Para Maduro */
.yellow { background-color: #FFEB3B; } /* Para Flor */
.pct-text { color: #2E7D32; } 
</style>
""", unsafe_allow_html=True)

# 2. Cargar el modelo YOLO
@st.cache_resource
def load_yolo_model():
    return YOLO("arandano.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 3. Encabezado
st.title("Resultados de Conteo")

# 4. Cargador de archivos
uploaded_file = st.file_uploader("Sube la imagen del cultivo de arándanos", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    # Inferencia
    results = model(input_image, conf=0.25)
    
    # Mostrar imagen con detecciones
    res_plotted = results[0].plot()
    predicted_image = Image.fromarray(res_plotted[:, :, ::-1])
    st.image(predicted_image, caption="Resultado del análisis YOLO", use_container_width=True)
    
    # 5. CONTEO CON TUS CATEGORÍAS EXACTAS
    # Nota: Las mayúsculas y minúsculas deben ser idénticas a como las entrenaste.
    # Por lo que veo en tu foto, "Verde" y "Maduro" inician con mayúscula.
    conteo_clases = {
        "Pinton": 0,
        "Verde": 0,
        "Maduro": 0,
        "Flor": 0
    }
    
    yolo_names = model.names 
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = yolo_names[class_id]
            
            # Si la clase detectada está en nuestro diccionario, sumamos 1
            if class_name in conteo_clases:
                conteo_clases[class_name] += 1

    # 6. Cálculos
    total_detectado = sum(conteo_clases.values())
    
    def calcular_porcentaje(valor, total):
        if total == 0: 
            return 0.0
        return round((valor / total) * 100, 1)

    # 7. Tabla HTML actualizada con tus nombres y los valores del diccionario
    html_content = f"""
    <div class="metric-container">
        <div class="metric-row header">
            <span style="flex:2;">Total</span>
            <span style="flex:1; text-align: left;">{total_detectado}</span>
            <span style="flex:1; text-align: right; color: #2E7D32;">%</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box purple"></span>Pintón</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Pinton']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Pinton'], total_detectado)}%)</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box green"></span>Verde</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Verde']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Verde'], total_detectado)}%)</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box blue"></span>Maduro</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Maduro']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Maduro'], total_detectado)}%)</span>
        </div>
        <div class="metric-row">
            <span style="flex:2;"><span class="color-box yellow"></span>Flor</span>
            <span style="flex:1; text-align: left;">{conteo_clases['Flor']}</span>
            <span class="pct-text" style="flex:1; text-align: right;">({calcular_porcentaje(conteo_clases['Flor'], total_detectado)}%)</span>
        </div>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)
    
    # 8. Botones
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start again", use_container_width=True, type="primary"):
            st.rerun()
    with col2:
        if st.button("Exit", use_container_width=True):
            st.info("Aplicación en espera de nueva actividad.")
