import plotly.express as px
import numpy as np
import random, re, gzip
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

seed = 42

#os.environ["PYTHONHASHSEED"] = str(seed)
#os.environ["TF_DETERMINISTIC_OPS"] = "1"   # fuerza operaciones deterministas

random.seed(seed)
np.random.seed(seed)
#tf.random.set_seed(seed)

###### Visualización completa del dataset. Mantener comentado a menos que se quera ver todo el contenido. #######

#with gzip.open("./data/GSE2034_series_matrix.txt.gz", "rt") as f:
#    for line in f:
#        print(line.strip())

labels = []

with gzip.open("./data/GSE2034_series_matrix.txt.gz", 'rt') as f:
    for line in f:
        if line.startswith("!Sample_characteristics_ch1") and "bone relapses" in line.lower():
            # Quitar comillas
            clean = line.replace('"', "")
            # Separar por tabulaciones
            parts = clean.strip().split("\t")

            for p in parts[1:]:
                match = re.search(r":\s*([01])\s*$", p.strip())
                if match:
                    labels.append(int(match.group(1)))




with gzip.open("./data/GSE2034_series_matrix.txt.gz", 'rt') as f:
    lines = f.readlines()

start = lines.index("!series_matrix_table_begin\n") + 1
end = lines.index("!series_matrix_table_end\n")

import pandas as pd
from io import StringIO

expr = pd.read_csv(StringIO("".join(lines[start:end])), sep="\t")

# Quitar columna ID 
X = expr.iloc[:, 1:].T  #  muestras x genes

# Convertir etiquetas a Series con mismo índice que X
y = pd.Series(labels, index=X.index)

print(X.shape)
print(y.shape)



# mapa de colores por clase (HTML hex)
color_map = {
    0: "#d12b78",  # Negativos
    1: "#ffa0d0"   # Positivos
}

# Asegurar que las etiquetas son del mismo tipo que las claves del mapa
y = y.astype(int)

# Histograma con colores HTML
fig = px.histogram(y, x=y, color=y, color_discrete_map=color_map)
fig.update_layout(bargap=0.6)
fig.update_xaxes(title_text='Metástasis ósea', title_font=dict(size=18))
fig.update_yaxes(title_text='Número de muestras', title_font=dict(size=18))
fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        tickfont=dict(color='black')
    )
)
fig.show()
fig.write_image("Histograma.png", engine="kaleido", scale=4)






# --- 1. Asegurar que X y y tienen el mismo índice ---
X = X.loc[y.index] 


# --- 2. Estandarizar datos (para aplicar PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Aplicamos PCA
pca = PCA(n_components=45)
X_pca = pca.fit_transform(X_scaled)


print("Varianza explicada por cada componente:")
print(pca.explained_variance_ratio_)

print("\nVarianza acumulada:")
print(pca.explained_variance_ratio_.cumsum())

# --- 4. DataFrame para graficar (PC1 vs PC2)
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'recaida': y.values
})
pca_df['recaida'] = pca_df['recaida'].astype('category')


# --- 5. Gráfica PCA
fig = px.scatter(
    pca_df,
    x='PC1', y='PC2',
    color='recaida',
    color_discrete_map={0: '#d12b78', 1: '#ffa0d0'},
    labels={'recaida': 'Clase'}
)
fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        tickfont=dict(color='black')
    )
)
fig.show()
fig.write_image("PCA.png", scale=4)







####################################
#       MODELOS DE CLASIFICACIÓN
####################################

# Dividir en train/test
x_train, x_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)




#=====================================
# Naive Bayes
#=====================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)





#=====================================
# k-NN
#=====================================
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5
)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)




#=====================================
# Logistic Regression
#=====================================
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)
log_reg.fit(x_train, y_train)

probas = log_reg.predict_proba(x_test)[:,1]

# ============= Escoger mejor umbral por ROC (Youden J) ============= #

fpr, tpr, thresholds = roc_curve(y_test, probas)

# Youden J = tpr - fpr
youden = tpr - fpr

best_idx = np.argmax(youden)
best_thr = thresholds[best_idx]

y_pred_lr = (probas >= best_thr).astype(int)






#=====================================
# Random Forest
#=====================================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)
rf.fit(x_train, y_train)

probas = rf.predict_proba(x_test)[:,1]

# ============= Escoger mejor umbral por ROC (Youden J) ============= #

fpr, tpr, thresholds = roc_curve(y_test, probas)

# Youden J = tpr - fpr
youden = tpr - fpr

best_idx = np.argmax(youden)
best_thr = thresholds[best_idx]

y_pred_rf = (probas >= best_thr).astype(int)







#=====================================
# SVM
#=====================================

from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=1,
    gamma='scale',
    class_weight="balanced",
    probability=True,
    random_state=42
)
svm.fit(x_train, y_train)

y_pred_svm = svm.predict(x_test)







#=====================================
# Matrices de confusión
#=====================================
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Lista de modelos ya entrenados
models_pred = {
    "Naive Bayes": y_pred_nb,
    "k-NN": y_pred_knn,
    "Regresión logística": y_pred_lr,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm
}

# Graficar matrices de confusión

for name, pred in models_pred.items():
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(name)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()





















# ============================================================
#          RED NEURONAL CON TENSORFLOW/KERAS    
# ============================================================


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


clases = np.unique(y_train)
pesos = compute_class_weight(
    class_weight="balanced",
    classes=clases,
    y=y_train
)
class_weights = dict(zip(clases, pesos))

# ---------------------------------------------------------------------
# Modelo para clasificación binaria
# ---------------------------------------------------------------------
modelo = models.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.BatchNormalization(),

    layers.Dense(128, activation="elu"),
    layers.Dropout(0.10, seed = 42),

    layers.Dense(64, activation="tanh"),
    layers.Dropout(0.10, seed = 42),

    layers.Dense(32, activation="elu"),

    layers.Dense(1, activation="sigmoid")
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="AUC")]
)

modelo.summary()

# ---------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# ---------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------
hist = modelo.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=16,
    epochs=200,
    callbacks=[early_stop],
    class_weight=class_weights,
    shuffle = False,
    verbose=1
)

# ---------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------
print("\nEvaluación en test:")
modelo.evaluate(x_test, y_test, verbose=2)

# ---------------------------------------------------------------------
# Matriz de confusión (Seaborn)
# ---------------------------------------------------------------------
# Predicciones binarias

y_pred_prob = modelo.predict(x_test)

y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar = False)
plt.title("Red Neuronal")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()








# ============================================================
#  VALIDACIÓN CRUZADA 5-FOLD CON AJUSTE DE UMBRAL (ROC)
# ============================================================
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

def select_best_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thr[best_idx]


# ----------- función para crear el modelo Keras -----------
def build_nn():
    model = models.Sequential([
        layers.Input(shape=(X_pca.shape[1],)),
        layers.BatchNormalization(),

        layers.Dense(128, activation="elu"),
        layers.Dropout(0.10),

        layers.Dense(64, activation="tanh"),
        layers.Dropout(0.10),

        layers.Dense(32, activation="elu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ----------- Lista de modelos a validar -----------
models_cv = {
    "Naive Bayes": GaussianNB(),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regresión Logística": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=42
    ),
    "SVM": SVC(kernel="rbf", C=1,
    gamma='scale',
    class_weight="balanced",
    probability=True,
    random_state=42),
    "Red Neuronal": KerasClassifier(build_fn=build_nn, epochs=50, batch_size=16, verbose=0)
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics_table = {
    "Modelo": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": []
}


for model_name, model in models_cv.items():
    accs = []
    precs = []
    recalls = []
    f1s = []
    aucs = []

    for train_idx, val_idx in kf.split(X_pca, y):
        X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Entrenar
        model.fit(X_tr, y_tr)

        # Probabilidades
        y_prob = model.predict_proba(X_val)[:, 1] if model_name != "Red Neuronal" else model.predict(X_val).ravel()

        # Elegir umbral en los modelos que requieren ajuste
        if model_name in ["Regresión Logística", "Random Forest", "SVM"]:
            thr = select_best_threshold(y_val, y_prob)
        else:
            thr = 0.5

        y_pred = (y_prob >= thr).astype(int)

        # Métricas
        accs.append(accuracy_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_prob))

    metrics_table["Modelo"].append(model_name)
    metrics_table["Accuracy"].append(f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
    metrics_table["Precision"].append(f"{np.mean(precs):.3f} ± {np.std(precs):.3f}")
    metrics_table["Recall"].append(f"{np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    metrics_table["F1"].append(f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


df_cv = pd.DataFrame(metrics_table)
print("\n===== VALIDACIÓN CRUZADA 5-FOLD =====\n")
print(df_cv)




