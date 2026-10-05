# 🌳 Módulo 04: Árboles de Decisión y Métodos de Ensamble

Los **Árboles de Decisión** y los **Métodos de Ensamble (*Ensembles*)** constituyen una de las familias de algoritmos más potentes y utilizadas en Machine Learning para datos estructurados (tabulares), destacando por su precisión y capacidad para modelar relaciones no lineales complejas.

---

## 🌲 1. Árboles de Decisión (CART)

El algoritmo **CART** (*Classification and Regression Trees*) construye árboles binarios mediante la división recursiva del espacio de características en regiones hiperrectangulares paralelas a los ejes.

```text
               [ Característica X1 <= 2.5 ]
                      /          \
                    Sí            No
                   /                \
        [ Clase A ]          [ Característica X2 <= 5.0 ]
                                    /          \
                                  Sí            No
                                 /                \
                      [ Clase B ]          [ Clase C ]
```

### 1.1 Criterios de División en Clasificación

Para determinar la mejor partición en un nodo $Q$, se busca la característica $j$ y el umbral $t$ que maximicen la reducción de impureza.

#### A. Impureza de Gini (*Gini Impurity*):
Mide la probabilidad de clasificar incorrectamente un elemento elegido al azar si se etiquetara aleatoriamente según la distribución de clases en el nodo:

$$
G(Q) = 1 - \sum_{k=1}^{K} p_k^2
$$

donde $p_k$ es la proporción de instancias de la clase $k$ en el nodo $Q$.

#### B. Entropía y Ganancia de Información (*Information Gain*):
Basada en la teoría de la información de Shannon, mide la incertidumbre o desorden del nodo:

$$
H(Q) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

La **Ganancia de Información ($IG$)** al dividir el nodo padre $Q$ en dos nodos hijos $Q_{\text{izq}}$ y $Q_{\text{der}}$ con tamaños $N_{\text{izq}}$ y $N_{\text{der}}$ respectivamente es:

$$
IG(Q, j, t) = H(Q) - \left( \frac{N_{\text{izq}}}{N} H(Q_{\text{izq}}) + \frac{N_{\text{der}}}{N} H(Q_{\text{der}}) \right)
$$

### 1.2 Criterio de División en Regresión

Para problemas de regresión, se utiliza la **Reducción del Error Cuadrático Medio ($MSE$)** o la reducción de varianza. La predicción $\hat{y}_R$ para una región $R$ es la media de los valores objetivo de las observaciones en esa región:

$$
\text{MSE}(R) = \frac{1}{|R|} \sum_{i \in R} \left( y^{(i)} - \hat{y}_R \right)^2 \quad \text{donde} \quad \hat{y}_R = \frac{1}{|R|} \sum_{i \in R} y^{(i)}
$$

---

## ✂️ 2. Control del Sobreajuste (Poda / *Pruning*)

Un árbol sin restricciones crecerá hasta memorizar los datos de entrenamiento (alta varianza, *overfitting*).

### Estrategias de Control:
1. **Pre-poda (*Pre-pruning* / Detención Temprana):**
   * Limitar la profundidad máxima (`max_depth`).
   * Requerir un número mínimo de muestras para dividir un nodo (`min_samples_split`).
   * Requerir un número mínimo de muestras en un nodo hoja (`min_samples_leaf`).

2. **Post-poda (*Post-pruning* / Cost Complexity Pruning):**
   * Permite que el árbol crezca por completo y luego elimina ramas reduciendo la función de costo parametrizada por $\alpha$:
   
   $$
   R_\alpha(T) = R(T) + \alpha |T|
   $$
   
   donde $R(T)$ es el error del árbol y $|T|$ es el número de nodos hoja.

---

## 🎒 3. Métodos de Ensamble: Bagging y Random Forest

Los métodos de ensamble combinan múltiples modelos base (*weak learners*) para construir un estimador más robusto.

### 3.1 Bagging (*Bootstrap Aggregating*)
Reduce la **varianza** combinando modelos independientes entrenados en distintas submuestras con reemplazo.

1. Se generan $B$ muestras *bootstrap* del conjunto original.
2. Se entrena un modelo no restringido (de alta varianza) en cada muestra.
3. La predicción final es el promedio (regresión) o el voto mayoritario (clasificación):

$$
\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{*b}(x)
$$

### 3.2 Random Forest (Bosques Aleatorios)
Mejora Bagging resolviendo la correlación entre árboles. Si existe una característica dominante, todos los árboles de Bagging la seleccionarán en la raíz.

* **Descorrelación de Árboles:** En cada división del árbol, el algoritmo selecciona aleatoriamente un subconjunto de $m$ características de las $n$ totales (típicamente $m = \sqrt{n}$ para clasificación y $m = \frac{n}{3}$ para regresión).

```text
       Muestra Bootstrap 1             Muestra Bootstrap 2
     [ Subconjunto m vars ]          [ Subconjunto m vars ]
              │                               │
              ▼                               ▼
          Árbol 1                         Árbol 2
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   Predicción Promedio / Voto
```

#### Error Out-Of-Bag (OOB):
Dado que el muestreo *bootstrap* deja fuera aproximadamente un $36.8\%$ de los datos en cada iteración ($e^{-1} \approx 0.368$), estas muestras no observadas se utilizan para evaluar el modelo sin necesidad de validación cruzada explícita.

---

## 🚀 4. Métodos de Ensamble: Boosting

El **Boosting** construye modelos de forma **secuencial**, donde cada nuevo modelo se enfoca en corregir los errores cometidos por los modelos anteriores. Reduce principalmente el **sesgo**.

### 4.1 AdaBoost (*Adaptive Boosting*)
* Asigna pesos a cada observación del conjunto de datos.
* Aumenta los pesos de las observaciones mal clasificadas por el árbol actual para que el siguiente árbol les dé prioridad.

### 4.2 Gradient Boosting (GBM)
* Entrena nuevos árboles para predecir los **residuos (pseudo-residuos)** del modelo acumulado previo mediante el descenso del gradiente sobre una función de pérdida $L(y, \hat{y})$.

Para una iteración $m$:

$$
r_{i m} = -\left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)}
$$

El nuevo modelo se actualiza con una tasa de aprendizaje $\eta \in (0, 1]$:

$$
f_m(x) = f_{m-1}(x) + \eta \cdot h_m(x)
$$

### 4.3 Implementaciones Optimizadas:
* **XGBoost:** Agrega regularización $L_1/L_2$ a la función de pérdida del árbol y utiliza expansiones de Taylor de segundo orden.
* **LightGBM:** Utiliza crecimiento orientado a hojas (*leaf-wise*) y agrupamiento por histogramas para mayor rapidez.
* **CatBoost:** Optimizado para el manejo automático de variables categóricas.

---

## 📊 5. Importancia de Características (*Feature Importance*)

1. **Mean Decrease Impurity (MDI):** Suma total de la reducción de impureza (Gini/Entropía) aportada por una característica a lo largo de todos los árboles del bosque. *Sesgado hacia características continuas o con alta cardinalidad.*
2. **Permutation Feature Importance:** Mide el incremento en el error del modelo al permutar aleatoriamente los valores de una característica en el conjunto de prueba/validación.

---

## 💡 6. Cuadro Comparativo

| Algoritmo | Reducción de | Construcción | Sensibilidad a Outliers | Complejidad Cómputo |
| :--- | :--- | :--- | :--- | :--- |
| **Árbol de Decisión** | Ninguna (sensible) | Secuencial / Un solo árbol | Alta | Muy Baja |
| **Random Forest** | **Varianza** | Paralelo | Baja / Robusto | Media |
| **Gradient Boosting** | **Sesgo y Varianza** | Secuencial | Sensible (según loss) | Alta |