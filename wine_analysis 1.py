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

print("\n Variáveis independentes:\n", wine.feature_names)
print("\n Classes das variáveis alvo (target):\n", wine.target_names)



#### Criação das bases de treinamento dos modelos utilizando os atributos indicados: ####




## Exibindo a base de dados na tela da base em 1 atributo (flavanoids) ##
X_modelo1 = df.drop(columns=['od280/od315_of_diluted_wines', 'magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols', 'alcohol', 'ash'])
print("\n Wine Data: ")
print(X_modelo1)
## Classificando o 'target' e exibindo as classes da base em 1 atributo (flavanoids) ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base em 1 atributo (flavanoids) ##
print('\nVariáveis independentes:', X_modelo1.shape)
print('Variáveis alvo (target)', y.shape)

## Fazendo o Split da base de treino e da base de teste com 1 atributo específico (flavanoids) ##
X_treino_Modelo1, X_teste_Modelo1, y_treino_Modelo1, y_teste_Modelo1 = train_test_split(X_modelo1, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 1 atributo (flavanoids) ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino_Modelo1))
print("\nShape dos dados de treino: ", X_treino_Modelo1.shape, y_treino_Modelo1.shape)
print("\nX Treino veja:\n ", X_treino_Modelo1)
print("\ny Treino:\n ", y_treino_Modelo1)
## Exibindo dados de teste com 1 atributo (flavanoids) ##
print("---------------------------")
print("\nX Teste: ", len(X_teste_Modelo1))
print("\nShape dos dados de teste: ", X_teste_Modelo1.shape, y_teste_Modelo1.shape)
print("\nX Teste veja:\n ", X_teste_Modelo1)
print("\ny Teste:\n ", y_teste_Modelo1)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 1 atributos específico (od280/od315_of_diluted_wines)##
X_modelo2 = df.drop(columns=['flavanoids', 'magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols', 'alcohol', 'ash'])
print("\n Wine Data: ")
print(X_modelo2)
## Classificando o 'target' e exibindo as classes da base com 1 atributos (od280/od315_of_diluted_wines) ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 1 atributos (od280/od315_of_diluted_wines) ##
print('\nVariáveis independentes:', X_modelo2.shape)
print('Variáveis alvo (target)', y.shape)

## Fazendo o Split da base de treino e da base de teste com 1 atributos (od280/od315_of_diluted_wines) ##
X_treino_Modelo2, X_teste_Modelo2, y_treino_Modelo2, y_teste_Modelo2 = train_test_split(X_modelo2, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 1 atributo (od280/od315_of_diluted_wines) ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino_Modelo2))
print("\nShape dos dados de treino: ", X_treino_Modelo2.shape, y_treino_Modelo2.shape)
print("\nX Treino veja:\n ", X_treino_Modelo2)
print("\ny Treino:\n ", y_treino_Modelo2)
## Exibindo dados de teste com 1 atributo (od280/od315_of_diluted_wines) ##
print("---------------------------")
print("\nX Teste: ", len(X_teste_Modelo2))
print("\nShape dos dados de teste: ", X_teste_Modelo2.shape, y_teste_Modelo2.shape)
print("\nX Teste veja:\n ", X_teste_Modelo2)
print("\ny Teste:\n ", y_teste_Modelo2)
print("---------------------------\n")

## Exibindo a base de dados na tela da base com 2 atributos (flavanoids e od280/od315_of_diluted_wines) ##
X_modelo3 = df.drop(columns=['magnesium', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'malic_acid', 'total_phenols', 'alcohol', 'ash'])
print("\n Wine Data: ")
print(X_modelo3)
## Classificando o 'target' e exibindo as classes da base com 2 atributos (flavanoids e od280/od315_of_diluted_wines) ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base com 2 atributos (flavanoids e od280/od315_of_diluted_wines) ##
print('\nVariáveis independentes:', X_modelo3.shape)
print('Variáveis alvo (target)', y.shape)

## Fazendo o Split da base de treino e da base de teste com 2 atributos (flavanoids e od280/od315_of_diluted_wines) ##
X_treino_Modelo3, X_teste_Modelo3, y_treino_Modelo3, y_teste_Modelo3 = train_test_split(X_modelo3, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento com 2 atributo (flavanoids e od280/od315_of_diluted_wines) ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino_Modelo3))
print("\nShape dos dados de treino: ", X_treino_Modelo3.shape, y_treino_Modelo3.shape)
print("\nX Treino veja:\n ", X_treino_Modelo3)
print("\ny Treino:\n ", y_treino_Modelo3)
## Exibindo dados de teste com 2 atributo (flavanoids e od280/od315_of_diluted_wines) ##
print("---------------------------")
print("\nX Teste: ", len(X_teste_Modelo3))
print("\nShape dos dados de teste: ", X_teste_Modelo3.shape, y_teste_Modelo3.shape)
print("\nX Teste veja:\n ", X_teste_Modelo3)
print("\ny Teste:\n ", y_teste_Modelo3)
print("---------------------------\n")

## Criação das listas que receberão os percentuais de acerto ##
list_modelo1 = []
list_modelo2 = []
list_modelo3 = []

## Laço que executa cada modelo 100x ##
for i in range(100):

    print(f"\nEsta é a {i + 1}ª execução!")

    ## Modelo RandomForest com 2 atributos específico e 100 árvores ##
    myrf_modelo1 = RandomForestClassifier(n_estimators= 100)
    myrf_modelo1.fit(X_treino_Modelo1, y_treino_Modelo1)
    randomf_modelo1 = myrf_modelo1.score(X_teste_Modelo1, y_teste_Modelo1)
    print("Rf_modelo1\n", randomf_modelo1)
    list_modelo1.append(randomf_modelo1)

    ## Modelo RandomForest com 2 atributos específico e 100 árvores ##
    myrf_modelo2 = RandomForestClassifier(n_estimators= 100)
    myrf_modelo2.fit(X_treino_Modelo2, y_treino_Modelo2)
    randomf_modelo2 = myrf_modelo2.score(X_teste_Modelo2, y_teste_Modelo2)
    print("Rf_modelo2\n", randomf_modelo2)
    list_modelo2.append(randomf_modelo2)

    ## Modelo RandomForest com 2 atributos específicos e 100 árvores ##
    myrf_modelo3 = RandomForestClassifier(n_estimators= 100)
    myrf_modelo3.fit(X_treino_Modelo3, y_treino_Modelo3)
    randomf_modelo3 = myrf_modelo3.score(X_teste_Modelo3, y_teste_Modelo3)
    print("Rf_modelo3\n", randomf_modelo3)
    list_modelo3.append(randomf_modelo3)


## Exibindo as listas com os dados das 100 execuções e anexar dos valores médios na lista acima ##
print("\n---------------------------")
print("\nValores Random Forest modelo 1 (flavanoids):", list_modelo1)
print("\nValores Random Forest modelo 2 (od280/od315_of_diluted_wines)", list_modelo2)
print("\nValores Random Forest modelo 3 (flavanoids + od280/od315_of_diluted_wines):", list_modelo3)
print("\n---------------------------\n")


print("\nO valor médio da Random Forest com o atributo flavanoids é:", mean(list_modelo1))
print("\nO valor médio da Random Forest com o atributo od280/od315_of_diluted_wines é:", mean(list_modelo2))
print("\nO valor médio da Random Forest com os atributos od280/od315_of_diluted_wines e flavanoids é:", mean(list_modelo3))

