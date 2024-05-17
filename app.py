import streamlit as st 
import pickle 
import pandas as pd
import numpy as np  
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


def get_clean_data():
  data = 'https://raw.githubusercontent.com/isaacperomero/Bioinformatics/main/breast_cancer_data.csv'
  
  data = pd.read_csv(data)
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def add_sidebar():
  st.sidebar.header("Medidas de los núcleos celulares")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radio (promedio)", "radius_mean"),
        ("Textura (promedio)", "texture_mean"),
        ("Perímetro (promedio)", "perimeter_mean"),
        ("Área (promedio)", "area_mean"),
        ("Suavidad (promedio)", "smoothness_mean"),
        ("Compacidad (promedio)", "compactness_mean"),
        ("Concavidad (promedio)", "concavity_mean"),
        ("Puntos cóncavos (promedio)", "concave points_mean"),
        ("Simetría (promedio)", "symmetry_mean"),
        ("Dimensión fractal (promedio)", "fractal_dimension_mean"),
        ("Radio (error estándar)", "radius_se"),
        ("Textura (error estándar)", "texture_se"),
        ("Perímetro (error estándar)", "perimeter_se"),
        ("Área (error estándar)", "area_se"),
        ("Suavidad (error estándar)", "smoothness_se"),
        ("Compacidad (error estándar)", "compactness_se"),
        ("Concavidad (error estándar)", "concavity_se"),
        ("Puntos cóncavos (error estándar)", "concave points_se"),
        ("Simetría (error estándar)", "symmetry_se"),
        ("Dimensión fractal (error estándar)", "fractal_dimension_se"),
        ("Radio (máximo)", "radius_worst"),
        ("Textura (máximo)", "texture_worst"),
        ("Perímetro (máximo)", "perimeter_worst"),
        ("Área (máximo)", "area_worst"),
        ("Suavidad (máximo)", "smoothness_worst"),
        ("Compacidad (máximo)", "compactness_worst"),
        ("Concavidad (máximo)", "concavity_worst"),
        ("Puntos cóncavos (máximo)", "concave points_worst"),
        ("Simetría (máximo)", "symmetry_worst"),
        ("Dimensión fractal (máximo)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
  return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'],axis = 1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value)/(max_value - min_value)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ["Radio","Textura","Perímetro","Área",
    "Suavidad","Compacidad","Concavidad", "Puntos cóncavos",
    "Simetría","Dimensión fractal"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
       r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Promedio'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Error Estándar'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Máximo'
    ))
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1] 
        )),
    showlegend=True
    )

    return fig

def add_prediction(input_data):
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))

    input_np = np.array(list(input_data.values())).reshape(1,-1)
    input_scaled = scaler.transform(input_np)

    prediction = model.predict(input_scaled)
    st.subheader("Estado de la agrupación celular")
    st.write("La agrupación celular es:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benigna</span>",unsafe_allow_html = True)
    else:
        st.write("<span class='diagnosis malicious'>Maligna</span>",unsafe_allow_html = True)
    
    st.write("Probabilidad Benigna: ", round(model.predict_proba(input_scaled)[0][0],3))
    st.write("Probabilidad Maligna: ", round(model.predict_proba(input_scaled)[0][1],3))

    st.write('El análisis tiene como único objetivo mejorar la calidad del diagnóstico y no pretende sustituir al diagnóstico profesional.')


def main():
    st.set_page_config(
        page_title = "Predicción de Cáncer de Mama",
        page_icon = ":female-doctor",
        layout = "wide",
        initial_sidebar_state="expanded"
    )

   with open("style.css") as f:
       st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)
     

    input_data = add_sidebar()
 

    with st.container():
        st.title("Predictor de Cáncer de Mama")
        st.write("El diagnóstico del cáncer de mama suele implicar el examen de muestras celulares obtenidas mediante procedimientos citológicos. Al integrar nuestra aplicación ML con un laboratorio de citología, se puede crear un flujo de trabajo completo y eficiente que maximiza la precisión y la rápidez en la detección del cáncer de mama.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_prediction(input_data)

if __name__ == '__main__':
    main() 
