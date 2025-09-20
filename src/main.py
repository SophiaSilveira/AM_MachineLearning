import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Carregar o dataset
df = pd.read_csv("../academic-stress-level.csv", sep=';')
print(df.columns)
# Remover a coluna Timestamp
df = df.drop(columns=["Timestamp"])

# Remover linhas com valores vazios
df = df.dropna(subset=["Study Environment"])

print("Your Academic Stage:", df["Your Academic Stage"].unique())
print("Study Environment:", df["Study Environment"].unique())
print("Coping strategy:", df["What coping strategy you use as a student?"].unique())
print("Bad habits:", df["Do you have any bad habits like smoking, drinking on a daily basis?"].unique())

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

df_encoded = df.copy()
df_encoded[[
    "Your Academic Stage",
    "Study Environment",
    "What coping strategy you use as a student?",
    "Do you have any bad habits like smoking, drinking on a daily basis?"
]] = encoder.fit_transform(df[[
    "Your Academic Stage",
    "Study Environment",
    "What coping strategy you use as a student?",
    "Do you have any bad habits like smoking, drinking on a daily basis?"
]]) + 1



#df_encoded.to_csv("academic-stress-level-encoded.csv", index=False)
# Supondo que a coluna alvo seja 'Rate your academic stress index'
X = df_encoded.drop(columns=["Rate your academic stress index"])
y = df_encoded["Rate your academic stress index"]

# Divisão 80% treino / 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(df_encoded.head())