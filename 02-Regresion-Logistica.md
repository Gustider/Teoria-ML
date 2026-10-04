# 📊 Módulo 02: Regresión Logística

La **Regresión Logística** es uno de los algoritmos fundamentales para problemas de **clasificación supervisada** (principalmente binaria). A pesar de su nombre, no se utiliza para predecir valores continuos, sino para estimar la **probabilidad** de que una observación pertenezca a una categoría específica.

---

## 🎯 1. Intuición y Planteamiento del Problema

En la Regresión Lineal tradicional, la respuesta del modelo es una combinación lineal continua:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n = z$$

El problema surge al aplicar esto a la clasificación: $z$ puede tomar valores desde $-\infty$ hasta $+\infty$. Las probabilidades, por definición, deben acotarse estrictamente en el intervalo $[0, 1]$.

Para resolver esto, la Regresión Logística mapea el valor continuo $z$ a una probabilidad mediante una **función de activación**: la **función Sigmoide**.

---

## 📐 2. La Función Sigmoide (Logística)

La función Sigmoide se define matemáticamente como:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Donde $z$ es la combinación lineal de las características $z = \boldsymbol{w}^T \boldsymbol{x} + b$.

### Propiedades clave:
* Si $z \to +\infty$, $\sigma(z) \to 1$.
* Si $z \to -\infty$, $\sigma(z) \to 0$.
* Si $z = 0$, $\sigma(z) = 0.5$.

```
   1.0 |                   /-----------
       |                  /
  p    |                 /
  0.5 - - - - - - - - - *  (z = 0)
       |               /
   0.0 |----------/
       +-------------------------------
                     z = 0
```

---

## ⚖️️ 3. Odds Ratio y Log-Odds (Logit)

Para entender cómo interpreta los datos la Regresión Logística, es necesario definir la noción de **Odds** (Razón de Probabilidades):

$$\text{Odds} = \frac{p}{1 - p}$$

Donde $p$ es la probabilidad de que ocurra el evento positivo ($y=1$).

Si aplicamos el logaritmo natural a los Odds, obtenemos la función **Logit**:

$$\text{Logit}(p) = \ln\left(\frac{p}{1 - p}\right) = z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

Esto demuestra que **la Regresión Logística es un modelo lineal en el espacio del logaritmo de los Odds**.

---

## 📉 4. Función de Costo: Binary Cross-Entropy (Log Loss)

En regresión lineal usamos el Error Cuadrático Medio (MSE). En Regresión Logística, combinar el MSE con la función Sigmoide genera una superficie de costo **no convexa** (llena de mínimos locales).

Por ello, se utiliza la **Entropía Cruzada Binaria** (*Binary Cross-Entropy* o *Log Loss*):

$$J(\boldsymbol{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

### Análisis del comportamiento:
* Si $y = 1$: El costo es $-\ln(\hat{y})$. Si $\hat{y} \to 1$, el costo es $0$. Si $\hat{y} \to 0$, el costo tiende a $+\infty$.
* Si $y = 0$: El costo es $-\ln(1 - \hat{y})$. Si $\hat{y} \to 0$, el costo es $0$. Si $\hat{y} \to 1$, el costo tiende a $+\infty$.

Esta función es **estrictamente convexa**, garantizando que el Descenso del Gradiente encuentre el mínimo global.

---

## 📏 5. Frontera de Decisión (*Decision Boundary*)

Por defecto, se define un umbral de decisión ($\text{threshold}$) de $0.5$:

$$\hat{y}_{\text{clase}} = \begin{cases} 1 & \text{si } P(y=1|\boldsymbol{x}) \ge 0.5 \quad (z \ge 0) \\ 0 & \text{si } P(y=1|\boldsymbol{x}) < 0.5 \quad (z < 0) \end{cases}$$

El umbral puede ajustarse dependiendo del problema de negocio:
* **Detección de fraude o enfermedades:** Conviene bajar el umbral (ej. $0.3$) para maximizar la sensibilidad (*Recall*) a costa de aceptar más falsos positivos.

---

## 📌 6. Supuestos del Modelo

A diferencia de la Regresión Lineal, la Regresión Logística **no requiere** normalidad en los residuos ni homocedasticidad, pero sí exige:

1. **Variable dependiente binaria/categórica.**
2. **Independencia de las observaciones:** Sin medidas repetidas ni series de tiempo autocorrelacionadas.
3. **Ausencia de Multicolinealidad alta:** Las variables independientes no deben estar fuertemente correlacionadas entre sí.
4. **Linealidad con el Logit:** Las variables continuas deben tener una relación lineal con $\ln(\text{Odds})$.

---

## 🧪 7. Métricas de Evaluación Principales

| Métrica | Fórmula | Interpretación |
| :--- | :--- | :--- |
| **Accuracy (Exactitud)** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Porcentaje general de aciertos. Engañoso en clases desbalanceadas. |
| **Precision (Precisión)** | $\frac{TP}{TP + FP}$ | De todo lo predicho como positivo, ¿cuánto era realmente positivo? |
| **Recall / Sensibilidad** | $\frac{TP}{TP + FN}$ | De todos los casos positivos reales, ¿cuántos logró detectar el modelo? |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Media armónica entre Precisión y Recall. |
| **ROC-AUC** | Área bajo la curva $TPR$ vs $FPR$ | Capacidad del modelo para separar ambas clases a través de distintos umbrales. |

---

## 💡 8. Ventajas y Limitaciones

### ✅ Ventajas:
* Altamente interpretable (los coeficientes e los exponenciales $e^{\beta_i}$ representan el cambio en los Odds).
* Salidas en forma de probabilidades bien calibradas.
* Eficiente en cómputo y rápido de entrenar.
* Poco proclive al sobreajuste si se aplica regularización ($L_1$ Lasso o $L_2$ Ridge).

### ❌ Limitaciones:
* Asume fronteras de decisión lineales (a menos que se agreguen términos polinómicos).
* Sensible a la presencia de *outliers* extremos.
* Requiere un tamaño muestral razonable para obtener estimaciones estables.