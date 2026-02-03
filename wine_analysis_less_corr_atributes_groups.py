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
      "1) alcohol\n"
      "2) malic_acid\n"
      "3) alcalinity_of_ash")

#print("\n Variáveis independentes:\n", wine.feature_names)
print("\n Classes das variáveis alvo (target):\n", wine.target_names)

#### Criação das bases de treinamento de cada modelo ####

## Base de treinamento modelo alcohol + malic_acid sendo criada e exibida ##
X_mod_alcohol_malic_acid = df.drop(columns=['od280/od315_of_diluted_wines', 'magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'total_phenols', 'ash', 'flavanoids'])
print("\n Wine Data mod alcohol + malic_acid: ")
print(X_mod_alcohol_malic_acid)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_alcohol_malic_acid.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo alcohol + malic_acid ##
X_treino_mod_alcohol_malic_acid, X_teste_mod_alcohol_malic_acid, y_treino_mod_alcohol_malic_acid, y_teste_mod_alcohol_malic_acid = train_test_split(X_mod_alcohol_malic_acid, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Base de treinamento modelo alcohol + alcalinity_of_ash sendo criada e exibida ##
X_mod_alcohol_alcalinity_of_ash = df.drop(columns=['od280/od315_of_diluted_wines', 'magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'malic_acid', 'total_phenols', 'flavanoids'])
print("\n Wine Data mod alcohol + alcalinity_of_ash: ")
print(X_mod_alcohol_alcalinity_of_ash)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_alcohol_alcalinity_of_ash.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo alcohol + alcalinity_of_ash ##
X_treino_mod_alcohol_alcalinity_of_ash, X_teste_mod_alcohol_alcalinity_of_ash, y_treino_mod_alcohol_alcalinity_of_ash, y_teste_mod_alcohol_alcalinity_of_ash = train_test_split(X_mod_alcohol_alcalinity_of_ash, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Base de treinamento modelo malic_acid + alcalinity_of_ash sendo criada e exibida ##
X_mod_malic_acid_alcalinity_of_ash = df.drop(columns=['od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols', 'alcohol', 'flavanoids'])
print("\n Wine Data mod malic_acid + alcalinity_of_ash: ")
print(X_mod_malic_acid_alcalinity_of_ash)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_malic_acid_alcalinity_of_ash.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo malic_acid + alcalinity_of_ash ##
X_treino_mod_malic_acid_alcalinity_of_ash, X_teste_mod_malic_acid_alcalinity_of_ash, y_treino_mod_malic_acid_alcalinity_of_ash, y_teste_mod_malic_acid_alcalinity_of_ash = train_test_split(X_mod_malic_acid_alcalinity_of_ash, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Base de treinamento modelo com TODAS as variáveis sendo criada e exibida ##
X_mod_all = df.drop(columns=['od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols', 'flavanoids'])
print("\n Wine Data mod all: ")
print(X_mod_all)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_all.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo all ##
X_treino_mod_all, X_teste_mod_all, y_treino_mod_all, y_teste_mod_all = train_test_split(X_mod_all, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Criação das listas que receberão os percentuais de acerto ##
list_mod_alcohol_malic_acid = []
list_mod_alcohol_alcalinity_of_ash = []
list_mod_malic_acid_alcalinity_of_ash = []
list_mod_all = []


## Laço que executa cada modelo 500x ##
for i in range(500):

    print(f"\nEsta é a {i + 1}ª execução!")

    ## Random Forest Modelo alcohol + malic_acid ##
    myrf_mod_alcohol_malic_acid = RandomForestClassifier(n_estimators= 100)
    myrf_mod_alcohol_malic_acid.fit(X_treino_mod_alcohol_malic_acid, y_treino_mod_alcohol_malic_acid)
    randomf_mod_alcohol_malic_acid = myrf_mod_alcohol_malic_acid.score(X_teste_mod_alcohol_malic_acid, y_teste_mod_alcohol_malic_acid)
    print("Rf Mod alcohol + malic_acid:", randomf_mod_alcohol_malic_acid)
    list_mod_alcohol_malic_acid.append(randomf_mod_alcohol_malic_acid)

    ## Random Forest Modelo alcohol + alcalinity_of_ash ##
    myrf_mod_alcohol_alcalinity_of_ash = RandomForestClassifier(n_estimators= 100)
    myrf_mod_alcohol_alcalinity_of_ash.fit(X_treino_mod_alcohol_alcalinity_of_ash, y_treino_mod_alcohol_alcalinity_of_ash)
    randomf_mod_alcohol_alcalinity_of_ash = myrf_mod_alcohol_alcalinity_of_ash.score(X_teste_mod_alcohol_alcalinity_of_ash, y_teste_mod_alcohol_alcalinity_of_ash)
    print("Rf Mod alcohol + alcalinity_of_ash:", randomf_mod_alcohol_alcalinity_of_ash)
    list_mod_alcohol_alcalinity_of_ash.append(randomf_mod_alcohol_alcalinity_of_ash)

    ## Random Forest Modelo malic_acid + alcalinity_of_ash ##
    myrf_mod_malic_acid_alcalinity_of_ash = RandomForestClassifier(n_estimators= 100)
    myrf_mod_malic_acid_alcalinity_of_ash.fit(X_treino_mod_malic_acid_alcalinity_of_ash, y_treino_mod_malic_acid_alcalinity_of_ash)
    randomf_mod_malic_acid_alcalinity_of_ash = myrf_mod_malic_acid_alcalinity_of_ash.score(X_teste_mod_malic_acid_alcalinity_of_ash, y_teste_mod_malic_acid_alcalinity_of_ash)
    print("Rf Mod malic_acid + alcalinity_of_ash:", randomf_mod_malic_acid_alcalinity_of_ash)
    list_mod_malic_acid_alcalinity_of_ash.append(randomf_mod_malic_acid_alcalinity_of_ash)

    ## Random Forest Modelo all ##
    myrf_mod_all = RandomForestClassifier(n_estimators= 100)
    myrf_mod_all.fit(X_treino_mod_all, y_treino_mod_all)
    randomf_mod_all = myrf_mod_all.score(X_teste_mod_all, y_teste_mod_all)
    print("Rf Mod all:", randomf_mod_all)
    list_mod_all.append(randomf_mod_all)

## Exibindo as listas com os dados das 500 execuções e anexar dos valores médios na lista acima ##
print("\n---------------------------")
print("\nValores Random Forest Modelo ash:", list_mod_alcohol_malic_acid)
print("\nValores Random Forest Modelo alcalinity_of_ash:", list_mod_alcohol_alcalinity_of_ash)
print("\nValores Random Forest Modelo magnesium:", list_mod_malic_acid_alcalinity_of_ash)
print("\nValores Random Forest Modelo malic_acid:", list_mod_all)
print("\n---------------------------\n")


print("\nO valor médio da Random Forest Modelo alcohol + malic_acid é:", mean(list_mod_alcohol_malic_acid))
print("O valor médio da Random Forest Modelo alcohol + alcalinity_of_ash é:", mean(list_mod_alcohol_alcalinity_of_ash))
print("O valor médio da Random Forest Modelo malic_acid + alcalinity-of-ash é:", mean(list_mod_malic_acid_alcalinity_of_ash))
print("O valor médio da Random Forest Modelo all é:", mean(list_mod_all))

print("\n---------------------------\n")