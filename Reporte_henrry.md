
## Clasificación Automática de Neumonía en Radiografías de Tórax: basada en descriptores tradicionales y modelos de Deep Learning

---

# 1. Introducción

La identificación temprana y precisa de patologías pulmonares mediante radiografías de tórax es un componente fundamental del diagnóstico clínico. Sin embargo, la interpretación manual depende fuertemente de la experiencia del especialista, y puede verse limitada en instituciones con alta demanda, escasez de radiólogos o variabilidad interobservador. En este contexto, los sistemas de apoyo al diagnóstico basados en visión por computador ofrecen una alternativa valiosa para aumentar la precisión y estandarizar interpretaciones.

Este proyecto aborda el problema de **clasificación binaria entre radiografías normales y radiografías con neumonía**, utilizando dos enfoques complementarios:  
1. **Descriptores tradicionales** (HOG, LBP, Haralick, Zernike) combinados con modelos clásicos de machine learning.  
2. **Modelos profundos preentrenados** (ResNet mediante Transfer Learning).

El objetivo principal es comparar rigurosamente ambos enfoques dentro de un pipeline reproducible y bien estructurado, evaluando su rendimiento, ventajas, limitaciones y potencial de generalización en imágenes médicas.

---

# 2. Marco Teórico

## 2.1. Preprocesamiento de Imágenes Médicas

El preprocesamiento constituye una etapa crítica en tareas de visión por computador, especialmente en imágenes médicas, donde la calidad puede variar según condiciones de adquisición, equipos utilizados y características del paciente. En este proyecto se emplearon:

- **Redimensionamiento:** garantiza dimensiones uniformes para permitir procesamiento vectorial.  
- **Conversión a escala de grises:** reduce la dimensionalidad sin perder información clínica relevante.  
- **Normalización:** homogeniza intensidades entre imágenes para estabilizar modelos.  
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** técnica propuesta por Pizer et al. (1987) que mejora el contraste de manera local y evita la sobre-amplificación del ruido, lo cual es particularmente útil para resaltar estructuras pulmonares sutiles.

Estas técnicas son fundamentales para asegurar que los descriptores tradicionales y las redes neuronales operen sobre insumos consistentes y comparables.

---

## 2.2. Descriptores Tradicionales de Visión por Computador

Se utilizaron cuatro familias de descriptores, cada una capturando propiedades distintas de las imágenes:

### ✔ HOG — *Histogram of Oriented Gradients*  
Propuesto por Dalal & Triggs (2005), captura gradientes locales y orientaciones, siendo ideal para bordes y estructuras anatómicas.

### ✔ LBP — *Local Binary Patterns*  
Introducido por Ojala et al. (2002), describe microtexturas robustas a cambios de iluminación, esenciales para caracterizar patrones pulmonares.

### ✔ Haralick (GLCM) — *Gray Level Co-occurrence Matrix*  
Haralick (1973) define atributos como homogeneidad, contraste y correlación, los cuales son relevantes para detectar alteraciones en la textura del parénquima pulmonar.

### ✔ Momentos de Zernike  
Descriptores invariantes a rotación que capturan información global de forma, útiles como complemento para distinguir configuraciones anatómicas anómalas.

---

## 2.3. Métodos de Clasificación

Los descriptores tradicionales se utilizaron como entrada para múltiples clasificadores clásicos:

- **Máquinas de Vectores de Soporte (SVM)**  
- **Random Forest**  
- **XGBoost**  
- **k-Nearest Neighbors (KNN)**  
- **Regresión Logística**

Todos estos métodos requieren vectores de características previamente construidos, motivo por el cual la calidad del feature engineering resulta determinante.

---

## 2.4. Deep Learning — Transfer Learning

Se integró una **ResNet** preentrenada en ImageNet, tomando solo las capas convolucionales como extractor de características y ajustando las capas finales para la clasificación binaria.  

Las redes residuales (He et al., 2016) resuelven problemas de degradación en arquitecturas profundas mediante conexiones de salto (*skip connections*), lo cual facilita el aprendizaje de representaciones complejas en imágenes médicas.

---

# 3. Metodología

Este proyecto se desarrolla mediante un pipeline estructurado en tres notebooks principales.

---

## 3.1. Notebook 01 — Exploratory Analysis & Cleaning

### Actividades realizadas
1. **Carga del dataset** (5.856 imágenes).  
2. Evaluación de **distribución de tamaños**, evidenciando alta heterogeneidad.  
3. Visualización de muestras para identificar diferencias de contraste y nitidez.  
4. Implementación del pipeline de preprocesamiento:  
   - Escala de grises  
   - Redimensionamiento  
   - Normalización  
   - Aplicación de CLAHE  
5. Documentación de conclusiones preliminares sobre calidad de las imágenes.

### Justificación
La inconsistencia en tamaños y niveles de contraste compromete tanto la extracción de características como el aprendizaje de modelos, por lo que se diseñó un preprocesamiento estandarizado que garantice homogeneidad estructural.

---

## 1. Diferencias de contraste y textura
Las imágenes con neumonía tienden a mostrar zonas más homogéneas y densas, debido a la consolidación pulmonar, mientras que las imágenes normales presentan un patrón más uniforme y con mayor detalle de estructuras pulmonares.

Esto se evidencia en los resultados de la ecualización CLAHE, los bordes detectados (Sobel y Laplaciano) y los colormaps, donde los detalles de los pulmones normales son más visibles.

---

## 2. Distribución de intensidades
Las gráficas de densidad (KDE) muestran que las imágenes de neumonía presentan una concentración mayor de píxeles en rangos intermedios de intensidad, reflejando áreas opacas, mientras que las imágenes normales tienen una distribución más dispersa, representando aire y tejido sano.

La CDF confirma que la probabilidad acumulada en los valores más bajos y medios es mayor en las imágenes con neumonía, mostrando diferencias en la iluminación global y opacidad pulmonar.

---

## 3. Comparación agregada
Al superponer los KDEs y CDFs de varios ejemplos, se observa que las diferencias entre las clases son consistentes: las imágenes normales tienden a tener un patrón más “plano” y extendido, mientras que las neumónicas muestran picos pronunciados en ciertas intensidades.

Esto sugiere que estas métricas estadísticas podrían ser útiles para diferenciar clases automáticamente.

---

## 4. Intensidad promedio y variabilidad
Los boxplots indican que la media de intensidad de las imágenes con neumonía suele ser ligeramente mayor que la de las imágenes normales, lo que refleja el incremento de densidad en los pulmones afectados.

También se observa mayor variabilidad entre imágenes con neumonía, lo que puede corresponder a diferentes grados de afectación pulmonar.

---

## 5. Importancia del análisis exploratorio
Estos análisis permiten identificar patrones visuales y estadísticos antes de entrenar modelos de clasificación, asegurando que las diferencias entre clases sean detectables y que el preprocesamiento mejore la discriminación entre imágenes.

![Texto alternativo](ruta/de/la/imagen.extensión)

---

## 3.2. Notebook 02 — Feature Extraction

### Descriptores implementados:
- HOG  
- LBP  
- Haralick/GLCM  
- Zernike Moments  

Se desarrolló una **función integral de extracción de características**, capaz de procesar todo el dataset de manera eficiente y generar una matriz final apta para modelado.

### Consideraciones técnicas:
- Se ajustaron parámetros de ventanas, radios y bloques para maximizar la representatividad del descriptor.  
- Las características se almacenaron vectorizadas y listas para modelos tradicionales.

---

## 3.3. Notebook 03 — Classification

### Contenido del notebook:
1. División del dataset en train/test.  
2. Estandarización de características.  
3. Entrenamiento de modelos clásicos.  
4. Optimización automática con GridSearchCV.  
5. Evaluación mediante:  
   - matrices de confusión  
   - precisión, recall, F1  
   - curvas ROC/AUC  
6. Ranking final de modelos tradicionales.  
7. Implementación de CNN (ResNet).  
8. Comparación entre enfoques clásicos y deep learning.  
9. Análisis interpretativo de importancia de características en modelos clásicos.

---

# 4. Experimentos y Resultados

## 4.1. Descriptores Tradicionales
SVM y Random Forest destacaron como los modelos con mayor rendimiento.  
Sin embargo, presentaron limitaciones inherentes a la naturaleza compleja de las radiografías.

Características de estos resultados:
- Sensibilidad moderada  
- Variabilidad dependiente del descriptor  
- Mayor susceptibilidad a ruido y diferencias de contraste  

## 4.2. Deep Learning — ResNet

La CNN superó significativamente a los modelos clásicos:

- Mayor F1-score  
- Mejor capacidad de generalización  
- Aprendizaje robusto sin necesidad de ingeniería manual  

Además, capturó patrones texturales y estructurales que no son fácilmente representables mediante descriptores tradicionales.

---

# 5. Análisis y Discusión

## 5.1. Comparación entre Enfoques
### Enfoques tradicionales:
- Alto costo computacional en extracción de características.  
- Dependencia directa del preprocesamiento.  
- Representaciones limitadas frente a variabilidad anatómica.

### Deep Learning:
- Aprendizaje autónomo de características.  
- Robusto ante variaciones en calidad de imagen.  
- Menor necesidad de intervención manual.

## 5.2. Limitaciones del proyecto
- Falta de augmentations fuertes para mejorar robustez.  
- Dataset con gran variabilidad visual.  
- No se realizó validación externa con imágenes nuevas (generalización clínica).  
- Los métodos clásicos son sensibles al desbalance si aparece.

## 5.3. Posibles Mejoras
- Fine-tuning completo de ResNet.  
- Incremento del tamaño del dataset.  
- Técnicas de augmentation realistas (ruido, rotaciones clínicas, recortes).  
- Ensemble entre CNN y descriptores tradicionales.  
- Validación cruzada k-fold para mayor estabilidad estadística.

---

# 6. Conclusiones

- El preprocesamiento es un componente esencial debido a la heterogeneidad del dataset.  
- Los descriptores tradicionales permitieron la construcción de clasificadores funcionales, aunque con limitaciones evidentes.  
- La ResNet empleada mediante Transfer Learning mostró mejor desempeño en todas las métricas evaluadas.  
- El pipeline desarrollado es reproducible, modular y escalable para investigaciones futuras.

---

# 7. Referencias (APA)

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.  
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.  
- Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.  
- Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.

---

# 8. Contribución Individual 
---

