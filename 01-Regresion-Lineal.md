# 📉 Módulo 01: Regresión Lineal y Análisis de Residuos

La **Regresión Lineal** es uno de los algoritmos supervisados fundamentales en estadística y Machine Learning. Su objetivo principal es modelar la relación entre una variable respuesta continua $y$ y una o más variables explicativas $x_1, x_2, \dots, x_n$.

---

## 🎯 1. Intuición y Planteamiento del Problema

El modelo de regresión lineal múltiple supone que la variable dependiente $y$ es una combinación lineal de las características $x_i$ más un término de error no observable $\epsilon$:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon$$

En formato matricial:

$$\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Donde:
* $\boldsymbol{y} \in \mathbb{R}^{m \times 1}$ es el vector de observaciones objetivo.
* $\boldsymbol{X} \in \mathbb{R}^{m \times (n+1)}$ es la matriz de diseño (con una columna de 1s para el sesgo/intercepto).
* $\boldsymbol{\beta} \in \mathbb{R}^{(n+1) \times 1}$ es el vector de coeficientes del modelo.
* $\boldsymbol{\epsilon} \in \mathbb{R}^{m \times 1}$ es el vector de errores irreducibles.

El objetivo es estimar el vector $\boldsymbol{\beta}$ con un conjunto de coeficientes $\hat{\boldsymbol{\beta}}$ tal que minimice la suma de errores al cuadrado.

---

## 📐 2. Mínimos Cuadrados Ordinarios (OLS / MCO)

La estimación por **Mínimos Cuadrados Ordinarios (OLS)** busca minimizar la función de costo del Error Cuadrático Medio ($MSE$):

$$J(\boldsymbol{\beta}) = \frac{1}{2m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2 = \frac{1}{2m} (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta})^T (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta})$$

Al derivar e igualar a cero con respecto a $\boldsymbol{\beta}$, se obtienen las **Ecuaciones Normales**:

$$\boldsymbol{X}^T \boldsymbol{X} \hat{\boldsymbol{\beta}} = \boldsymbol{X}^T \boldsymbol{y}$$

Si la matriz $\boldsymbol{X}^T \boldsymbol{X}$ es invertible, la solución analítica exacta es:

$$\hat{\boldsymbol{\beta}} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$$

---

## 🏛️ 3. Supuestos de Gauss-Markov

El **Teorema de Gauss-Markov** establece que el estimador OLS es el **BLUE** (*Best Linear Unbiased Estimator* / Mejor Estimador Lineal Insesgado) si se cumplen los siguientes supuestos:

1. **Linealidad en los Parámetros:** La relación entre las variables de entrada y la salida es lineal respecto a los coeficientes $\boldsymbol{\beta}$.
2. **Exogeneidad / Esperanza Nula del Error:** $E[\boldsymbol{\epsilon} | \boldsymbol{X}] = 0$. Los errores no contienen información sistemática sobre las características.
3. **Homocedasticidad:** $\text{Var}(\epsilon_i | \boldsymbol{X}) = \sigma^2$ para todo $i$. La varianza del error debe ser constante a lo largo de todas las observaciones.
4. **Ausencia de Autocorrelación:** $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ para todo $i \neq j$. No existe correlación entre los errores de distintas observaciones.
5. **Ausencia de Multicolinealidad Perfecta:** La matriz $\boldsymbol{X}^T \boldsymbol{X}$ debe ser de rango completo (no invertible si hay dependencias lineales exactas entre características).

> **Nota sobre Normalidad:** La condición de que los errores sigan una distribución normal $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ no es estrictamente necesaria para OLS, pero sí para realizar inferencia estadística (intervalos de confianza y pruebas $t$ / $F$).

---

## 🔍 4. Análisis de Residuos

El residuo $e_i$ de una observación representa la diferencia entre el valor real y el valor predicho:

$$e_i = y_i - \hat{y}_i$$

El análisis de residuos es el paso fundamental de diagnóstico para validar si el modelo cumple los supuestos teóricos.

```
 Residuos (e)
      ^
 +2   |       .       .          .
      |   .       .         .        .
  0 --+----------------------------------> Predicciones (\hat{y})
      |     .        .        .
 -2   |          .       .          .
      +----------------------------------
      Homocedasticidad: Dispersión uniforme (Patrón de banda horizontal)
```

### Gráficos Clave de Diagnóstico:

1. **Residuos vs. Valores Ajustados ($\hat{y}$):**
   * *Ideal:* Una distribución aleatoria de puntos alrededor de la línea horizontal $e=0$.
   * *Problemas:* Si se forma un embudo (*heterocedasticidad*) o una curva (falta de linealidad).
2. **Gráfico Q-Q Normal (Quantile-Quantile):**
   * Evalúa la normalidad de los residuos comparando los cuantiles empíricos con los cuantiles teóricos de una distribución normal.
   * *Ideal:* Puntos alineados sobre la diagonal de $45^\circ$.
3. **Scale-Location (Residuos Estandarizados):**
   * Evalúa si la varianza de los residuos cambia según el nivel de la predicción.
4. **Residuos vs. Leverage (Distancia de Cook):**
   * Permite detectar *outliers* y puntos de alto *leverage* (observaciones con influencia desproporcionada en la pendiente de la recta).

---

## 📊 5. Métricas de Evaluación

| Métrica | Fórmula | Descripción |
| :--- | :--- | :--- |
| **MAE (Error Absoluto Medio)** | $\frac{1}{m} \sum \vert y_i - \hat{y}_i \vert$ | Promedio directo de los errores en las mismas unidades que la variable $y$. Robusto a *outliers*. |
| **MSE (Error Cuadrático Medio)** | $\frac{1}{m} \sum (y_i - \hat{y}_i)^2$ | Penaliza severamente los errores grandes debido al cuadrado. |
| **RMSE (Raíz del MSE)** | $\sqrt{\text{MSE}}$ | Interpretable en las mismas unidades que la variable respuesta $y$. |
| **$R^2$ (Coeficiente de Determinación)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proporción de la varianza total de $y$ explicada por el modelo. $R^2 \in (-\infty, 1]$. |
| **$R^2$ Ajustado** | $1 - \left[ \frac{(1 - R^2)(m - 1)}{m - n - 1} \right]$ | Penaliza la inclusión de variables innecesarias que incrementan el $R^2$ de forma artificial. |

---

## 🛠️ 6. Soluciones cuando los Supuestos Fallan

* **Falta de linealidad:** Aplicar transformaciones no lineales a las variables de entrada ($x^2$, $\sqrt{x}$) o incorporar términos polinómicos.
* **Heterocedasticidad:** Aplicar transformación logarítmica a la variable dependiente $\ln(y)$ o utilizar **Mínimos Cuadrados Ponderados (WLS)**.
* **Multicolinealidad Alta:** Eliminar variables redundantes (evaluando el factor VIF - *Variance Inflation Factor*) o aplicar técnicas de reducción de dimensionalidad (PCA).
* **Sobreajuste (*Overfitting*):** Incorporar términos de regularización $L_1$ (Lasso) o $L_2$ (Ridge).