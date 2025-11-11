## Importação das bibliotecas necessárias para o projeto
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
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

print("\n Variáveis independentes:\n", wine.feature_names)
print("\n Classes das variáveis alvo (target):\n", wine.target_names)



#### Criação das bases de treinamento de 13 a 1 atributo: ####


## Exibindo a base de dados na tela da base com 13 atributos ##
X_13 = df
print("\n Wine Data: ")
print(X_13)
## Classificando o 'target' e exibindo as classes da base com 13 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 13 atributos ##
print('\nVariáveis independentes:', X_13.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 13 atributos ##
X_treino13, X_teste13, y_treino13, y_teste13 = train_test_split(X_13, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 13 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino13))
print("\nShape dos dados de treino: ", X_treino13.shape, y_treino13.shape)
print("\nX Treino veja:\n ", X_treino13)
print("\ny Treino:\n ", y_treino13)
## Exibindo dados de teste com 13 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste13))
print("\nShape dos dados de teste: ", X_teste13.shape, y_teste13.shape)
print("\nX Teste veja:\n ", X_teste13)
print("\ny Teste:\n ", y_teste13)
print("---------------------------\n")


## Exibindo a base de dados na tela da base com 12 atributos ##
X_12 = df.drop(columns='flavanoids')
print("\n Wine Data: ")
print(X_12)
## Classificando o 'target' e exibindo as classes da base com 12 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 12 atributos ##
print('\nVariáveis independentes:', X_12.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 12 atributos ##
X_treino12, X_teste12, y_treino12, y_teste12 = train_test_split(X_12, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 12 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino12))
print("\nShape dos dados de treino: ", X_treino12.shape, y_treino12.shape)
print("\nX Treino veja:\n ", X_treino12)
print("\ny Treino:\n ", y_treino12)
## Exibindo dados de teste com 12 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste12))
print("\nShape dos dados de teste: ", X_teste12.shape, y_teste12.shape)
print("\nX Teste veja:\n ", X_teste12)
print("\ny Teste:\n ", y_teste12)
print("---------------------------\n")


## Exibindo a base de dados na tela da base com 11 atributos ##
X_11 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines'])
print("\n Wine Data: ")
print(X_11)
## Classificando o 'target' e exibindo as classes da base com 11 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 11 atributos ##
print('\nVariáveis independentes:', X_11.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 11 atributos ##
X_treino11, X_teste11, y_treino11, y_teste11 = train_test_split(X_11, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 11 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino11))
print("\nShape dos dados de treino: ", X_treino11.shape, y_treino11.shape)
print("\nX Treino veja:\n ", X_treino11)
print("\ny Treino:\n ", y_treino11)
## Exibindo dados de teste com 11 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste11))
print("\nShape dos dados de teste: ", X_teste11.shape, y_teste11.shape)
print("\nX Teste veja:\n ", X_teste11)
print("\ny Teste:\n ", y_teste11)
print("---------------------------\n")


## Exibindo a base de dados na tela da base com 10 atributos ##
X_10 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins'])
print("\n Wine Data: ")
print(X_10)
## Classificando o 'target' e exibindo as classes da base com 10 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 10 atributos ##
print('\nVariáveis independentes:', X_10.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 10 atributos ##
X_treino10, X_teste10, y_treino10, y_teste10 = train_test_split(X_10, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 10 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino10))
print("\nShape dos dados de treino: ", X_treino10.shape, y_treino10.shape)
print("\nX Treino veja:\n ", X_treino10)
print("\ny Treino:\n ", y_treino10)
## Exibindo dados de teste com 10 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste10))
print("\nShape dos dados de teste: ", X_teste10.shape, y_teste10.shape)
print("\nX Teste veja:\n ", X_teste10)
print("\ny Teste:\n ", y_teste10)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 9 atributos ##
X_9 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline'])
print("\n Wine Data: ")
print(X_9)
## Classificando o 'target' e exibindo as classes da base com 9 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 9 atributos ##
print('\nVariáveis independentes:', X_9.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 9 atributos ##
X_treino9, X_teste9, y_treino9, y_teste9 = train_test_split(X_9, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 9 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino9))
print("\nShape dos dados de treino: ", X_treino9.shape, y_treino9.shape)
print("\nX Treino veja:\n ", X_treino9)
print("\ny Treino:\n ", y_treino9)
## Exibindo dados de teste com 9 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste9))
print("\nShape dos dados de teste: ", X_teste9.shape, y_teste9.shape)
print("\nX Teste veja:\n ", X_teste9)
print("\ny Teste:\n ", y_teste9)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 8 atributos ##
X_8 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines','proanthocyanins','proline','hue'])
print("\n Wine Data: ")
print(X_8)
## Classificando o 'target' e exibindo as classes da base com 8 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 8 atributos ##
print('\nVariáveis independentes:', X_8.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 8 atributos ##
X_treino8, X_teste8, y_treino8, y_teste8 = train_test_split(X_8, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 8 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino8))
print("\nShape dos dados de treino: ", X_treino8.shape, y_treino8.shape)
print("\nX Treino veja:\n ", X_treino8)
print("\ny Treino:\n ", y_treino8)
## Exibindo dados de teste com 8 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste8))
print("\nShape dos dados de teste: ", X_teste8.shape, y_teste8.shape)
print("\nX Teste veja:\n ", X_teste8)
print("\ny Teste:\n ", y_teste8)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 7 atributos ##
X_7 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines','proanthocyanins','proline','hue','color_intensity'])
print("\n Wine Data: ")
print(X_7)
## Classificando o 'target' e exibindo as classes da base com 7 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 7 atributos ##
print('\nVariáveis independentes:', X_7.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 7 atributos ##
X_treino7, X_teste7, y_treino7, y_teste7 = train_test_split(X_7, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 7 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino7))
print("\nShape dos dados de treino: ", X_treino7.shape, y_treino7.shape)
print("\nX Treino veja:\n ", X_treino7)
print("\ny Treino:\n ", y_treino7)
## Exibindo dados de teste com 7 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste7))
print("\nShape dos dados de teste: ", X_teste7.shape, y_teste7.shape)
print("\nX Teste veja:\n ", X_teste7)
print("\ny Teste:\n ", y_teste7)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 6 atributos ##
X_6 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols'])
print("\n Wine Data: ")
print(X_6)
## Classificando o 'target' e exibindo as classes da base com 6 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 6 atributos ##
print('\nVariáveis independentes:', X_6.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 6 atributos ##
X_treino6, X_teste6, y_treino6, y_teste6 = train_test_split(X_6, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 6 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino6))
print("\nShape dos dados de treino: ", X_treino6.shape, y_treino6.shape)
print("\nX Treino veja:\n ", X_treino6)
print("\ny Treino:\n ", y_treino6)
## Exibindo dados de teste com 6 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste6))
print("\nShape dos dados de teste: ", X_teste6.shape, y_teste6.shape)
print("\nX Teste veja:\n ", X_teste6)
print("\ny Teste:\n ", y_teste6)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 5 atributos ##
X_5 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash'])
print("\n Wine Data: ")
print(X_5)
## Classificando o 'target' e exibindo as classes da base com 5 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 5 atributos ##
print('\nVariáveis independentes:', X_5.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 5 atributos ##
X_treino5, X_teste5, y_treino5, y_teste5 = train_test_split(X_5, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 5 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino5))
print("\nShape dos dados de treino: ", X_treino5.shape, y_treino5.shape)
print("\nX Treino veja:\n ", X_treino5)
print("\ny Treino:\n ", y_treino5)
## Exibindo dados de teste com 5 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste5))
print("\nShape dos dados de teste: ", X_teste5.shape, y_teste5.shape)
print("\nX Teste veja:\n ", X_teste5)
print("\ny Teste:\n ", y_teste5)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 4 atributos ##
X_4 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid'])
print("\n Wine Data: ")
print(X_4)
## Classificando o 'target' e exibindo as classes da base com 4 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 4 atributos ##
print('\nVariáveis independentes:', X_4.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 4 atributos ##
X_treino4, X_teste4, y_treino4, y_teste4 = train_test_split(X_4, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 4 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino4))
print("\nShape dos dados de treino: ", X_treino4.shape, y_treino4.shape)
print("\nX Treino veja:\n ", X_treino4)
print("\ny Treino:\n ", y_treino4)
## Exibindo dados de teste com 4 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste4))
print("\nShape dos dados de teste: ", X_teste4.shape, y_teste4.shape)
print("\nX Teste veja:\n ", X_teste4)
print("\ny Teste:\n ", y_teste4)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 3 atributos ##
X_3 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols'])
print("\n Wine Data: ")
print(X_3)
## Classificando o 'target' e exibindo as classes da base com 3 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 3 atributos ##
print('\nVariáveis independentes:', X_3.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 3 atributos ##
X_treino3, X_teste3, y_treino3, y_teste3 = train_test_split(X_3, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 3 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino3))
print("\nShape dos dados de treino: ", X_treino3.shape, y_treino3.shape)
print("\nX Treino veja:\n ", X_treino3)
print("\ny Treino:\n ", y_treino3)
## Exibindo dados de teste com 3 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste3))
print("\nShape dos dados de teste: ", X_teste3.shape, y_teste3.shape)
print("\nX Teste veja:\n ", X_teste3)
print("\ny Teste:\n ", y_teste3)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 2 atributos ##
X_2 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols', 'alcohol'])
print("\n Wine Data: ")
print(X_2)
## Classificando o 'target' e exibindo as classes da base com 2 atributos ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 2 atributos ##
print('\nVariáveis independentes:', X_2.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 2 atributos ##
X_treino2, X_teste2, y_treino2, y_teste2 = train_test_split(X_2, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 2 atributos ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino2))
print("\nShape dos dados de treino: ", X_treino2.shape, y_treino2.shape)
print("\nX Treino veja:\n ", X_treino2)
print("\ny Treino:\n ", y_treino2)
## Exibindo dados de teste com 2 atributos ##
print("---------------------------")
print("\nX Teste: ", len(X_teste2))
print("\nShape dos dados de teste: ", X_teste2.shape, y_teste2.shape)
print("\nX Teste veja:\n ", X_teste2)
print("\ny Teste:\n ", y_teste2)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 1 atributo ##
X_1 = df.drop(columns=['flavanoids','od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols', 'alcohol', 'ash'])
print("\n Wine Data: ")
print(X_1)
## Classificando o 'target' e exibindo as classes da base com 1 atributo ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 1 atributo ##
print('\nVariáveis independentes:', X_1.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste com 1 atributo ##
X_treino1, X_teste1, y_treino1, y_teste1 = train_test_split(X_1, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 1 atributo ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino1))
print("\nShape dos dados de treino: ", X_treino1.shape, y_treino1.shape)
print("\nX Treino veja:\n ", X_treino1)
print("\ny Treino:\n ", y_treino1)
## Exibindo dados de teste com 1 atributo ##
print("---------------------------")
print("\nX Teste: ", len(X_teste1))
print("\nShape dos dados de teste: ", X_teste1.shape, y_teste1.shape)
print("\nX Teste veja:\n ", X_teste1)
print("\ny Teste:\n ", y_teste1)
print("---------------------------\n")

## Criação das listas que receberão os percentuais de acerto ##
list_13atributes = []
list_12atributes = []
list_11atributes = []
list_10atributes = []
list_9atributes = []
list_8atributes = []
list_7atributes = []
list_6atributes = []
list_5atributes = []
list_4atributes = []
list_3atributes = []
list_2atributes = []
list_1atribute = []

## Laço que executa cada modelo 100x ##
for i in range(100):

    print(f"\nEsta é a {i + 1}ª execução!")

    ## Modelo RandomForest com todos os 13 atributos ##
    myrf_13 = RandomForestClassifier(n_estimators= 100)
    myrf_13.fit(X_treino13, y_treino13)
    randomf_13atributes = myrf_13.score(X_teste13, y_teste13)
    print("Rf_13", randomf_13atributes)
    list_13atributes.append(randomf_13atributes)

    ## Modelo RandomForest com 12 atributos ##
    myrf_12 = RandomForestClassifier(n_estimators= 100)
    myrf_12.fit(X_treino12, y_treino12)
    randomf_12atributes = myrf_12.score(X_teste12, y_teste12)
    print("Rf_12", randomf_12atributes)
    list_12atributes.append(randomf_12atributes)
    
    ## Modelo RandomForest com 11 atributos ##
    myrf_11 = RandomForestClassifier(n_estimators= 100)
    myrf_11.fit(X_treino11, y_treino11)
    randomf_11atributes = myrf_11.score(X_teste11, y_teste11)
    print("Rf_11", randomf_11atributes)
    list_11atributes.append(randomf_11atributes)

    ## Modelo RandomForest com 10 atributos aleatórios e 100 árvores ##
    myrf_10 = RandomForestClassifier(n_estimators= 100)
    myrf_10.fit(X_treino10, y_treino10)
    randomf_10atributes = myrf_10.score(X_teste10, y_teste10)
    print("Rf_10", randomf_10atributes)
    list_10atributes.append(randomf_10atributes)
    
    ## Modelo RandomForest com 9 atributos aleatórios e 100 árvores ##
    myrf_9 = RandomForestClassifier(n_estimators= 100)
    myrf_9.fit(X_treino9, y_treino9)
    randomf_9atributes = myrf_9.score(X_teste9, y_teste9)
    print("Rf_9", randomf_9atributes)
    list_9atributes.append(randomf_9atributes)
    
    ## Modelo RandomForest com 8 atributos aleatórios e 100 árvores ##
    myrf_8 = RandomForestClassifier(n_estimators= 100)
    myrf_8.fit(X_treino8, y_treino8)
    randomf_8atributes = myrf_8.score(X_teste8, y_teste8)
    print("Rf_8", randomf_8atributes)
    list_8atributes.append(randomf_8atributes)

    ## Modelo RandomForest com 7 atributos aleatórios e 100 árvores ##
    myrf_7 = RandomForestClassifier(n_estimators= 100)
    myrf_7.fit(X_treino7, y_treino7)
    randomf_7atributes = myrf_7.score(X_teste7, y_teste7)
    print("Rf_7", randomf_7atributes)
    list_7atributes.append(randomf_7atributes)

    ## Modelo RandomForest com 6 atributos aleatórios e 100 árvores ##
    myrf_6 = RandomForestClassifier(n_estimators= 100)
    myrf_6.fit(X_treino6, y_treino6)
    randomf_6atributes = myrf_6.score(X_teste6, y_teste6)
    print("Rf_6", randomf_6atributes)
    list_6atributes.append(randomf_6atributes)

    ## Modelo RandomForest com 5 atributos aleatórios e 100 árvores ##
    myrf_5 = RandomForestClassifier(n_estimators= 100)
    myrf_5.fit(X_treino5, y_treino5)
    randomf_5atributes = myrf_5.score(X_teste5, y_teste5)
    print("Rf_5", randomf_5atributes)
    list_5atributes.append(randomf_5atributes)

    ## Modelo RandomForest com 4 atributos aleatórios e 100 árvores ##
    myrf_4 = RandomForestClassifier(n_estimators= 100)
    myrf_4.fit(X_treino4, y_treino4)
    randomf_4atributes = myrf_4.score(X_teste4, y_teste4)
    print("Rf_4", randomf_4atributes)
    list_4atributes.append(randomf_4atributes)

    ## Modelo RandomForest com 3 atributos aleatórios e 100 árvores ##
    myrf_3 = RandomForestClassifier(n_estimators= 100)
    myrf_3.fit(X_treino3, y_treino3)
    randomf_3atributes = myrf_3.score(X_teste3, y_teste3)
    print("Rf_3", randomf_3atributes)
    list_3atributes.append(randomf_3atributes)

    ## Modelo RandomForest com 2 atributos aleatórios e 100 árvores ##
    myrf_2 = RandomForestClassifier(n_estimators= 100)
    myrf_2.fit(X_treino2, y_treino2)
    randomf_2atributes = myrf_2.score(X_teste2, y_teste2)
    print("Rf_2", randomf_2atributes)
    list_2atributes.append(randomf_2atributes)

    ## Modelo RandomForest com 1 atributos aleatórios e 100 árvores ##
    myrf_1 = RandomForestClassifier(n_estimators= 100)
    myrf_1.fit(X_treino1, y_treino1)
    randomf_1atribute = myrf_1.score(X_teste1, y_teste1)
    print("Rf_1\n", randomf_1atribute)
    list_1atribute.append(randomf_1atribute)

## Exibindo as listas com os dados das 100 execuções e anexar dos valores médios na lista acima ##
print("\n---------------------------")
print("\nValores Random Forest 13 atributos:", list_13atributes)
print("\nValores Random Forest 12 atributos:", list_12atributes)
print("\nValores Random Forest 11 atributos:", list_11atributes)
print("\nValores Random Forest 10 atributos:", list_10atributes)
print("\nValores Random Forest 9 atributos:", list_9atributes)
print("\nValores Random Forest 8 atributos:", list_8atributes)
print("\nValores Random Forest 7 atributos:", list_7atributes)
print("\nValores Random Forest 6 atributos:", list_6atributes)
print("\nValores Random Forest 5 atributos:", list_5atributes)
print("\nValores Random Forest 4 atributos:", list_4atributes)
print("\nValores Random Forest 3 atributos:", list_3atributes)
print("\nValores Random Forest 2 atributos:", list_2atributes)
print("\nValores Random Forest 1 atributos:", list_1atribute)
print("\n---------------------------\n")


print("\nO valor médio da Random Forest com 13 atributos é:", mean(list_13atributes))
print("O valor médio da Random Forest com 12 atributos é:", mean(list_12atributes))
print("O valor médio da Random Forest com 11 atributos é:", mean(list_11atributes))
print("O valor médio da Random Forest com 10 atributos é:", mean(list_10atributes))
print("O valor médio da Random Forest com 9 atributos é:", mean(list_9atributes))
print("O valor médio da Random Forest com 8 atributos é:", mean(list_8atributes))
print("O valor médio da Random Forest com 7 atributos é:", mean(list_7atributes))
print("O valor médio da Random Forest com 6 atributos é:", mean(list_6atributes))
print("O valor médio da Random Forest com 5 atributos é:", mean(list_5atributes))
print("O valor médio da Random Forest com 4 atributos é:", mean(list_4atributes))
print("O valor médio da Random Forest com 3 atributos é:", mean(list_3atributes))
print("O valor médio da Random Forest com 2 atributos é:", mean(list_2atributes))
print("O valor médio da Random Forest com 1 atributos é:", mean(list_1atribute))
