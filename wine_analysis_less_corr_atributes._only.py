## Importação das bibliotecas necessárias para o projeto
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Carregando Wine Recognition Dataset ##
wine = datasets.load_wine()

## Conversão do dataset em um dataframe para análise das correlações, os nomes dos atributos são o cabeçalho ##
df = pd.DataFrame(wine.data, columns= wine.feature_names)
print(df)

'''
## Criação da matriz de correlação dos atributos através do dataframe ##
matriz_correlacao = df.corr()


## Comando de criação da janela que exibirá o 'mapa de calor' de correlações ##
plt.figure(figsize=(12, 15))

## Criação do mapa de calor a ser utilizado na plotagem ##
sns.heatmap(
    matriz_correlacao,
    annot=True,     # Mostra os valores numéricos na célula
    cmap='coolwarm', # Escolha um mapa de cores (ex: 'coolwarm', 'viridis', 'RdBu')
    fmt=".2f",      # Formata os números com 2 casas decimais
    linewidths=.5,  # Linhas entre as células
    cbar=True       # Mostra a barra de cores
)

## Comando de criação do título da figura e comando de exibição na tela ##
plt.title('Mapa de Calor da Matriz de Correlação')
plt.show()
'''
print("Variáveis sem correlação que serão estudadas:\n"
      "1) ash\n"
      "2) alcalinity_of_ash\n"
      "3) magnesium\n"
      "4) malic_acid\n"
      "5) alcohol\n"
      "6) color_intensity")

#print("\n Variáveis independentes:\n", wine.feature_names)
print("\n Classes das variáveis alvo (target):\n", wine.target_names)

#### Criação das bases de treinamento de cada modelo ####

## Base de treinamento modelo APENAS ash sendo criada e exibida ##
X_mod_ash = df.drop(columns=['od280/od315_of_diluted_wines', 'magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols', 'alcohol', 'flavanoids'])
print("\n Wine Data mod ash: ")
print(X_mod_ash)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_ash.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo ash ##
X_treino_mod_ash, X_teste_mod_ash, y_treino_mod_ash, y_teste_mod_ash = train_test_split(X_mod_ash, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Base de treinamento modelo APENAS alcalinity_of_ash sendo criada e exibida ##
X_mod_alcalinity_of_ash = df.drop(columns=['od280/od315_of_diluted_wines', 'magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'malic_acid', 'total_phenols', 'alcohol', 'flavanoids'])
print("\n Wine Data mod alcalinity_of_ash: ")
print(X_mod_alcalinity_of_ash)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_alcalinity_of_ash.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo alcalinity_of_ash ##
X_treino_mod_alcalinity_of_ash, X_teste_mod_alcalinity_of_ash, y_treino_mod_alcalinity_of_ash, y_teste_mod_alcalinity_of_ash = train_test_split(X_mod_alcalinity_of_ash, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Base de treinamento modelo APENAS magnesium sendo criada e exibida ##
X_mod_magnesium = df.drop(columns=['od280/od315_of_diluted_wines', 'alcalinity_of_ash', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'malic_acid', 'total_phenols', 'alcohol', 'flavanoids'])
print("\n Wine Data mod magnesium: ")
print(X_mod_magnesium)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_magnesium.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo magnesium ##
X_treino_mod_magnesium, X_teste_mod_magnesium, y_treino_mod_magnesium, y_teste_mod_magnesium = train_test_split(X_mod_magnesium, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Base de treinamento modelo APENAS malic_acid sendo criada e exibida ##
X_mod_malic_acid = df.drop(columns=['od280/od315_of_diluted_wines', 'alcalinity_of_ash', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols', 'alcohol', 'flavanoids'])
print("\n Wine Data mod malic_acid: ")
print(X_mod_malic_acid)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_malic_acid.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo malic_acid ##
X_treino_mod_malic_acid, X_teste_mod_malic_acid, y_treino_mod_malic_acid, y_teste_mod_malic_acid = train_test_split(X_mod_malic_acid, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Base de treinamento modelo APENAS alcohol sendo criada e exibida ##
X_mod_alcohol = df.drop(columns=['od280/od315_of_diluted_wines', 'alcalinity_of_ash', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols', 'malic_acid', 'flavanoids'])
print("\n Wine Data mod alcohol: ")
print(X_mod_alcohol)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_alcohol.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo alcohol ##
X_treino_mod_alcohol, X_teste_mod_alcohol, y_treino_mod_alcohol, y_teste_mod_alcohol = train_test_split(X_mod_alcohol, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Base de treinamento modelo APENAS color_intensity sendo criada e exibida ##
X_mod_color_intensity = df.drop(columns=['od280/od315_of_diluted_wines', 'alcalinity_of_ash', 'proanthocyanins', 'proline', 'hue', 'alcohol', 'nonflavanoid_phenols', 'color_intensity', 'magnesium', 'total_phenols', 'malic_acid', 'flavanoids'])
print("\n Wine Data mod color_intensity: ")
print(X_mod_color_intensity)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_color_intensity.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo alcohol ##
X_treino_mod_color_intensity, X_teste_mod_color_intensity, y_treino_mod_color_intensity, y_teste_mod_color_intensity = train_test_split(X_mod_color_intensity, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Criação das listas que receberão os percentuais de acerto ##
list_mod_ash = []
list_mod_alcalinity_of_ash = []
list_mod_magnesium = []
list_mod_malic_acid = []
list_mod_alcohol = []
list_mod_color_intensity = []


## Laço que executa cada modelo 400x ##
for i in range(400):

    print(f"\nEsta é a {i + 1}ª execução!")

    ## Random Forest Modelo ash ##
    myrf_mod_ash = RandomForestClassifier(n_estimators= 100)
    myrf_mod_ash.fit(X_treino_mod_ash, y_treino_mod_ash)
    randomf_mod_ash = myrf_mod_ash.score(X_teste_mod_ash, y_teste_mod_ash)
    print("Rf Mod ash:", randomf_mod_ash)
    list_mod_ash.append(randomf_mod_ash)

    ## Random Forest Modelo alcalinity_of_ash ##
    myrf_mod_alcalinity_of_ash = RandomForestClassifier(n_estimators= 100)
    myrf_mod_alcalinity_of_ash.fit(X_treino_mod_alcalinity_of_ash, y_treino_mod_alcalinity_of_ash)
    randomf_mod_alcalinity_of_ash = myrf_mod_alcalinity_of_ash.score(X_teste_mod_alcalinity_of_ash, y_teste_mod_alcalinity_of_ash)
    print("Rf Mod alcalinity_of_ash:", randomf_mod_alcalinity_of_ash)
    list_mod_alcalinity_of_ash.append(randomf_mod_alcalinity_of_ash)

    ## Random Forest Modelo magnesium ##
    myrf_mod_magnesium = RandomForestClassifier(n_estimators= 100)
    myrf_mod_magnesium.fit(X_treino_mod_magnesium, y_treino_mod_magnesium)
    randomf_mod_magnesium = myrf_mod_magnesium.score(X_teste_mod_magnesium, y_teste_mod_magnesium)
    print("Rf Mod magnesium:", randomf_mod_magnesium)
    list_mod_magnesium.append(randomf_mod_magnesium)

    ## Random Forest Modelo malic_acid ##
    myrf_mod_malic_acid = RandomForestClassifier(n_estimators= 100)
    myrf_mod_malic_acid.fit(X_treino_mod_malic_acid, y_treino_mod_malic_acid)
    randomf_mod_malic_acid = myrf_mod_malic_acid.score(X_teste_mod_malic_acid, y_teste_mod_malic_acid)
    print("Rf Mod malic_acid:", randomf_mod_malic_acid)
    list_mod_malic_acid.append(randomf_mod_malic_acid)

    ## Random Forest Modelo alcohol ##
    myrf_mod_alcohol = RandomForestClassifier(n_estimators= 100)
    myrf_mod_alcohol.fit(X_treino_mod_alcohol, y_treino_mod_alcohol)
    randomf_mod_alcohol = myrf_mod_alcohol.score(X_teste_mod_alcohol, y_teste_mod_alcohol)
    print("Rf Mod alcohol:", randomf_mod_alcohol)
    list_mod_alcohol.append(randomf_mod_alcohol)
    
    ## Random Forest Modelo color_intensity ##
    myrf_mod_color_intensity = RandomForestClassifier(n_estimators= 100)
    myrf_mod_color_intensity.fit(X_treino_mod_color_intensity, y_treino_mod_color_intensity)
    randomf_mod_color_intensity = myrf_mod_color_intensity.score(X_teste_mod_color_intensity, y_teste_mod_color_intensity)
    print("Rf Mod color_intensity:", randomf_mod_color_intensity)
    list_mod_color_intensity.append(randomf_mod_color_intensity)


## Exibindo as listas com os dados das 100 execuções e anexar dos valores médios na lista acima ##
print("\n---------------------------")
print("\nValores Random Forest Modelo ash:", list_mod_ash)
print("\nValores Random Forest Modelo alcalinity_of_ash:", list_mod_alcalinity_of_ash)
print("\nValores Random Forest Modelo magnesium:", list_mod_magnesium)
print("\nValores Random Forest Modelo malic_acid:", list_mod_malic_acid)
print("\nValores Random Forest Modelo alcohol:", list_mod_alcohol)
print("\nValores Random Forest Modelo color_intensity:", list_mod_color_intensity)
print("\n---------------------------\n")


print("\nO valor médio da Random Forest Modelo ash é:", mean(list_mod_ash))
print("O valor médio da Random Forest Modelo alcalinity_of_ash é:", mean(list_mod_alcalinity_of_ash))
print("O valor médio da Random Forest Modelo magnesium é:", mean(list_mod_magnesium))
print("O valor médio da Random Forest Modelo malic_acid é:", mean(list_mod_malic_acid))
print("O valor médio da Random Forest Modelo alcohol é:", mean(list_mod_alcohol))
print("O valor médio da Random Forest Modelo color_intensity é:", mean(list_mod_color_intensity))

print("\n---------------------------\n")