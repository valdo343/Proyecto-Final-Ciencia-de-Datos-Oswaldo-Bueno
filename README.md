# Breast Cancer Bone Metastasis Prediction

## 📋 Descripción del Proyecto

Este proyecto presenta un pipeline de aprendizaje automático para predecir la aparición de metástasis ósea en pacientes con cáncer de mama a partir de datos de expresión genética obtenidos del repositorio GEO. El flujo de trabajo incluye preprocesamiento de datos, reducción de dimensionalidad mediante PCA, entrenamiento y comparación de distintos modelos de clasificación, y evaluación de su desempeño mediante validación cruzada y métricas de clasificación.

## 🎯 Objetivos

- Analizar datos de expresión genética de pacientes con cáncer de mama.
- Desarrollar modelos de clasificación para predecir metástasis ósea.
- Comparar el desempeño de distintos algoritmos de aprendizaje automático mediante métricas de clasificación y validación cruzada.

## 📁 Estructura del Proyecto

```
Breast-Cancer-Bone-Metastasis-Prediction/
│
├── data/                          # Directorio de datos
│   └── [archivos de datos .txt.gz]
│
├── Proyecto_Oswaldo.py           # Script principal del análisis
└── Reporte_Proyecto_Oswaldo.pdf  # Reporte final del proyecto 
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.13.0**
- Bibliotecas de ciencia de datos:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib/seaborn
  - plotly.express

## 📊 Datos

El proyecto utiliza datos de perfiles de expresión genética con información sobre metástasis óseas en pacientes, codificadas como:
- `0`: Sin metástasis ósea en un lapso de 5 años.
- `1`: Con metástasis ósea en un lapso de 5 años.


## 📈 Resultados Clave

- Reducción de más de 22,000 variables mediante Análisis de Componentes Principales (PCA).
- Comparación de seis algoritmos de clasificación supervisada.
- El modelo SVM con kernel RBF obtuvo el mejor desempeño en términos de F1-score durante la validación cruzada.
- Evaluación mediante Accuracy, Precision, Recall, F1-score, curvas ROC y matrices de confusión.

## 👨‍💻 Autor

**Oswaldo Bueno Rivera**  
- E-mail: oswaldo.bueno@cimat.mx

