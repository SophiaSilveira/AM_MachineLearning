######################### Libraries Import #########################
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

######################### Learning Models Import #########################
from models.dt import runDT
#from models.knn import runKNN
#from models.naive import runNaive
from models.rf import runRF

######################### Start - pre-processing  Academic Stress -> Try 1#########################
# Carregar o dataset
df_asl = pd.read_csv("./database/academic-stress-level.csv", sep=';')

# Remover a coluna Timestamp
df_asl = df_asl.drop(columns=["Timestamp"])

# Remover linhas com valores vazios
df_asl = df_asl.dropna(subset=["Study Environment"])

# Definir a ordem dos valores manualmente
categories = [
    ["high school", "undergraduate", "post-graduate"],   # Your Academic Stage
    ["Noisy", "disrupted", "Peaceful"],                      # Study Environment
    ["Emotional breakdown (crying a lot)", 
     "Social support (friends, family)",
     "Analyze the situation and handle it with intellect"],           # Coping strategy (exemplo)
    ["No",
     "prefer not to say",
     "Yes"
    ]
]

# Montando dataset pós pre-processing
encoder = OrdinalEncoder(categories=categories)

df_encoded = df_asl.copy()
df_encoded[[
    "Your Academic Stage",
    "Study Environment",
    "What coping strategy you use as a student?",
    "Do you have any bad habits like smoking, drinking on a daily basis?"
]] = encoder.fit_transform(df_asl[[
    "Your Academic Stage",
    "Study Environment",
    "What coping strategy you use as a student?",
    "Do you have any bad habits like smoking, drinking on a daily basis?"
]]) + 1

######################### Result file of pre-processing #########################
#df_encoded.to_csv("./database/academic-stress-level-encoded.csv", index=False)

######################### Separetion Data - Treining/Testing #########################
# Supondo que a coluna alvo seja 'Rate your academic stress index'
X = df_encoded.drop(columns=["Rate your academic stress index"])
y = df_encoded["Rate your academic stress index"]

# Divisão 80% treino / 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

######################### Call Learning Models #########################
#print("########### Decision Tree ##########")
#runDT(X_train, X_test, y_train, y_test)

#print("\n\n")
#print("########### KNN ##########")
#runKNN(X_train, X_test, y_train, y_test)

#print("\n\n")
#print("########### Naive bayes ##########")
#runNaive(df)

#print("########### Ramdom Forest ##########")
#runRF(X_train, X_test, y_train, y_test)


######################### Start - pre-processing  Mental Health -> Try 2#########################

# Carregar o dataset
df_mht = pd.read_csv("./database/mental_health_and_technology_usage_2024.csv")

# Remover a coluna Id - Pois pode causar problemas na predição Ruído/Irrelevância, Overfitting
df_mht = df_mht.drop(columns=["User_ID"])

# Definindo a distância dos atributos categóricos de forma manual
categories_mht = [
    ["Female", "Male", "Other"],
    ["Excellent", "Fair", "Good", "Poor"],
    ["High", "Medium", "Low"],
    ["Yes", "No"],
    ["Positive", "Neutral", "Negative"],
    ["Yes", "No"]
]

# Montando dataset pós pre-processing
encoder_mht = OrdinalEncoder(categories=categories_mht)

# Copiar para não alterar o original
df_encoded_mht = df_mht.copy()

# Features categóricas (sem incluir o target)
categorical_features = [
    "Gender",
    "Stress_Level",
    "Support_Systems_Access",
    "Work_Environment_Impact",
    "Online_Support_Usage"
]

# Definir categorias apenas para features
categories_mht = [
    ["Female", "Male", "Other"],
    ["High", "Medium", "Low"],
    ["Yes", "No"],
    ["Positive", "Neutral", "Negative"],
    ["Yes", "No"]
]

# Aplicar encoder apenas nas features
encoder_mht = OrdinalEncoder(categories=categories_mht)
df_encoded_mht[categorical_features] = encoder_mht.fit_transform(df_mht[categorical_features]) + 1

# Supondo que a coluna alvo seja 'Mental_Health_Status'
# O target continua como estava, só será usado em y_mht
X_mht = df_encoded_mht.drop(columns=["Mental_Health_Status"])
y_mht = df_mht["Mental_Health_Status"]   # <- target original, não encodado

target_encoder = OrdinalEncoder(categories=[["Excellent", "Fair", "Good", "Poor"]])
y_mht = target_encoder.fit_transform(df_mht[["Mental_Health_Status"]]).ravel() + 1
df_encoded_mht["Mental_Health_Status"] = y_mht

######################### Result file of pre-processing #########################
df_encoded_mht.to_csv("./database/mental_health_and_technology_usage_2024_encoded.csv", index=False)

######################### Separetion Data - Treining/Testing #########################
# Divisão 80% treino / 20% teste
X_train_mht, X_test_mht, y_train_mht, y_test_mht = train_test_split(
    X_mht, y_mht, test_size=0.2, random_state=42, shuffle=True, stratify=y_mht
)

######################### Call Learning Models #########################
#print("Dataset: mental_health_and_technology_usage_2024")

print("########### Decision Tree ##########")
runDT(X_train_mht, X_test_mht, y_train_mht, y_test_mht)
