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

## Base de treinamento modelo com TODAS as variáveis SEM CORRELAÇÃO + od280/od315_of_diluted_wines sendo criada e exibida ##
X_mod_all_od = df.drop(columns=['proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols', 'flavanoids'])
print("\n Wine Data mod all + od: ")
print(X_mod_all_od)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_all_od.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo all + od280/od315_of_diluted_wines ##
X_treino_mod_all_od, X_teste_mod_all_od, y_treino_mod_all_od, y_teste_mod_all_od = train_test_split(X_mod_all_od, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Base de treinamento modelo com TODAS as variáveis SEM CORRELAÇÃO + flavanoids sendo criada e exibida ##
X_mod_all_flavanoids = df.drop(columns=['od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols'])
print("\n Wine Data mod all + flavanoids: ")
print(X_mod_all_flavanoids)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_all_flavanoids.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo all + flavanoids ##
X_treino_mod_all_flavanoids, X_teste_mod_all_flavanoids, y_treino_mod_all_flavanoids, y_teste_mod_all_flavanoids = train_test_split(X_mod_all_flavanoids, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Base de treinamento modelo com TODAS as variáveis SEM CORRELAÇÃO + od280/od315_of_diluted_wines + flavanoids sendo criada e exibida ##
X_mod_all_od_flavanoids = df.drop(columns=['proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'ash', 'magnesium', 'total_phenols'])
print("\n Wine Data mod all + od + flavanoids: ")
print(X_mod_all_od_flavanoids)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod_all_od_flavanoids.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo all + od280/od315_of_diluted_wines + flavanoids ##
X_treino_mod_all_od_flavanoids, X_teste_mod_all_od_flavanoids, y_treino_mod_all_od_flavanoids, y_teste_mod_all_od_flavanoids = train_test_split(X_mod_all_od_flavanoids, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)


## Criação das listas que receberão os percentuais de acerto ##
list_mod_all_od = []
list_mod_all_flavanoids = []
list_mod_all_od_flavanoids = []

## Laço que executa cada modelo 500x ##
for i in range(500):

    print(f"\nEsta é a {i + 1}ª execução!")

    ## Random Forest Modelo all + od280/od315_of_diluted_wines ##
    myrf_mod_all_od = RandomForestClassifier(n_estimators= 100)
    myrf_mod_all_od.fit(X_treino_mod_all_od, y_treino_mod_all_od)
    randomf_mod_all_od = myrf_mod_all_od.score(X_teste_mod_all_od, y_teste_mod_all_od)
    print("Rf Mod all + od280/od315_of_diluted_wines:", randomf_mod_all_od)
    list_mod_all_od.append(randomf_mod_all_od)

    ## Random Forest Modelo all + flavanoids ##
    myrf_mod_all_flavanoids = RandomForestClassifier(n_estimators= 100)
    myrf_mod_all_flavanoids.fit(X_treino_mod_all_flavanoids, y_treino_mod_all_flavanoids)
    randomf_mod_all_flavanoids = myrf_mod_all_flavanoids.score(X_teste_mod_all_flavanoids, y_teste_mod_all_flavanoids)
    print("Rf Mod all + flavanoids:", randomf_mod_all_flavanoids)
    list_mod_all_flavanoids.append(randomf_mod_all_flavanoids)

    ## Random Forest Modelo all + od280/od315_of_diluted_wines + flavanoids ##
    myrf_mod_all_od_flavanoids = RandomForestClassifier(n_estimators= 100)
    myrf_mod_all_od_flavanoids.fit(X_treino_mod_all_od_flavanoids, y_treino_mod_all_od_flavanoids)
    randomf_mod_all_od_flavanoids = myrf_mod_all_od_flavanoids.score(X_teste_mod_all_od_flavanoids, y_teste_mod_all_od_flavanoids)
    print("Rf Mod all + od280/od315_of_diluted_wines + flavanoids:", randomf_mod_all_od_flavanoids)
    list_mod_all_od_flavanoids.append(randomf_mod_all_od_flavanoids)

## Exibindo as listas com os dados das 500 execuções e anexar dos valores médios na lista acima ##
print("\n---------------------------")
print("\nValores Random Forest Modelo all + od280/od315_of_diluted_wines:", list_mod_all_od)
print("\nValores Random Forest Modelo all + flavanoids:", list_mod_all_flavanoids)
print("\nValores Random Forest Modelo all + od280/od315_of_diluted_wines + flavanoids:", list_mod_all_od_flavanoids)
print("\n---------------------------\n")


print("\nO valor médio da Random Forest Modelo all + od280/od315_of_diluted_wines é:", mean(list_mod_all_od))
print("O valor médio da Random Forest Modelo all + flavanoids é:", mean(list_mod_all_flavanoids))
print("O valor médio da Random Forest Modelo all + od280/od315_of_diluted_wines + flavanoids:", mean(list_mod_all_od_flavanoids))
print("\n---------------------------\n")