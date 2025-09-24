######################### Libraries Import #########################
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

######################### Learning Models Import #########################
from dt import runDT
from knn import runKNN
from naive import runNaive

######################### pre-processing #########################
# Carregar o dataset
df_asl = pd.read_csv("./academic-stress-level.csv", sep=';')
df_mht = pd.read_csv("./mental_health_and_technology_usage_2024.csv")
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
#df_encoded.to_csv("academic-stress-level-encoded.csv", index=False)

######################### Separetion Data - Treining/Testing #########################
# Supondo que a coluna alvo seja 'Rate your academic stress index'
X = df_encoded.drop(columns=["Rate your academic stress index"])
y = df_encoded["Rate your academic stress index"]

# Divisão 80% treino / 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

######################### Call Learning Models #########################
print("########### Decision Tree ##########")
runDT(X_train, X_test, y_train, y_test)

print("\n\n")
print("########### KNN ##########")
runKNN(X_train, X_test, y_train, y_test)

print("\n\n")
print("########### Naive bayes ##########")
runNaive(df)
