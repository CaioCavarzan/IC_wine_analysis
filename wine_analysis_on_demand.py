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


print("\n Variáveis independentes:\n", wine.feature_names)
print("\n Classes das variáveis alvo (target):\n", wine.target_names)

print("\nConsiderando que: "
        "\n--> Os atributos 'ash', 'alcalinity_ash' e 'magnesium' não possuem correlação acima de 0,5 com os demais atributos"
        "\n--> Os atributos 'flavanoids' e 'od280/od315_of_diluted_wines' possuem 5 correlações acima de 0,5 com os demais atributos"
        "\n--> O atributo 'total_phenols' possui 4 correlações acima de 0,5 com os demais atributos\n"
        "-------------------------------------------------------------------------------------------\n"
        "\nSerão realizadas as seguintes análises:"
        "\n1) Modelo treinado apenas com a coluna 'flavanoids'"
        "\n2) Modelo treinado apenas com a coluna 'od280/od315_of_diluted_wines'"
        "\n3) Modelo treinado com as colunas 'flavanoids' + 'od280/od315_of_diluted_wines'"
        "\n4) Modelo treinado com as colunas 'flavanoids' + 'total_phenols' + 'ash' + 'alcalinity_of_ash' + 'magnesium'"
        "\n5) Modelo treinado com as colunas 'od280/od315_of_diluted_wines' + 'total_phenols + 'ash' + 'alcalinity_of_ash' + 'magnesium'"
        "\n6) Modelo treinado com as colunas 'flavanoids' + 'od280/od315_of_diluted_wines' + 'total_phenols + 'ash' + 'alcalinity_of_ash' + 'magnesium'")


#### Criação das bases de treinamento de cada modelo ####


## Base de treinamento modelo 4 sendo criada e exibida ##
X_mod4 = df.drop(columns=['od280/od315_of_diluted_wines', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'malic_acid', 'alcohol'])
print("\n Wine Data mod 4: ")
print(X_mod4)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod4.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo 4 ##
X_treino_mod4, X_teste_mod4, y_treino_mod4, y_teste_mod4 = train_test_split(X_mod4, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento do modelo 4 ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino_mod4))
print("\nShape dos dados de treino: ", X_treino_mod4.shape, y_treino_mod4.shape)
print("\nX Treino veja:\n ", X_treino_mod4)
print("\ny Treino:\n ", y_treino_mod4)
## Exibindo dados de teste do modelo 4 ##
print("---------------------------")
print("\nX Teste: ", len(X_teste_mod4))
print("\nShape dos dados de teste: ", X_teste_mod4.shape, y_teste_mod4.shape)
print("\nX Teste veja:\n ", X_teste_mod4)
print("\ny Teste:\n ", y_teste_mod4)
print("---------------------------\n")


## Base de treinamento modelo 5 sendo criada e exibida ##
X_mod5 = df.drop(columns=['flavanoids', 'proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'malic_acid', 'alcohol'])
print("\n Wine Data mod 5: ")
print(X_mod5)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod5.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo 5 ##
X_treino_mod5, X_teste_mod5, y_treino_mod5, y_teste_mod5 = train_test_split(X_mod5, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento do modelo 5 ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino_mod5))
print("\nShape dos dados de treino: ", X_treino_mod5.shape, y_treino_mod5.shape)
print("\nX Treino veja:\n ", X_treino_mod5)
print("\ny Treino:\n ", y_treino_mod5)
## Exibindo dados de teste do modelo 5 ##
print("---------------------------")
print("\nX Teste: ", len(X_teste_mod5))
print("\nShape dos dados de teste: ", X_teste_mod5.shape, y_teste_mod5.shape)
print("\nX Teste veja:\n ", X_teste_mod5)
print("\ny Teste:\n ", y_teste_mod5)
print("---------------------------\n")


## Base de treinamento modelo 6 sendo criada e exibida ##
X_mod6 = df.drop(columns=['proanthocyanins', 'proline', 'hue', 'color_intensity', 'nonflavanoid_phenols', 'malic_acid', 'alcohol'])
print("\n Wine Data mod 6: ")
print(X_mod6)
## Classificando o 'target' e exibindo as classes da base ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)
## Exibindo a quantidade de instâncias e de variáveis da base de treinamento ##
print('\nVariáveis independentes:', X_mod6.shape)
print('Variáveis alvo (target)', y.shape)
## Fazendo o Split da base de treino e da base de teste do modelo 6 ##
X_treino_mod6, X_teste_mod6, y_treino_mod6, y_teste_mod6 = train_test_split(X_mod6, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)
## Exibindo dados do treinamento do modelo 6 ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino_mod6))
print("\nShape dos dados de treino: ", X_treino_mod6.shape, y_treino_mod6.shape)
print("\nX Treino veja:\n ", X_treino_mod6)
print("\ny Treino:\n ", y_treino_mod6)
## Exibindo dados de teste do modelo 6 ##
print("---------------------------")
print("\nX Teste: ", len(X_teste_mod6))
print("\nShape dos dados de teste: ", X_teste_mod6.shape, y_teste_mod6.shape)
print("\nX Teste veja:\n ", X_teste_mod6)
print("\ny Teste:\n ", y_teste_mod6)
print("---------------------------\n")


## Criação das listas que receberão os percentuais de acerto ##
list_mod4 = []
list_mod5 = []
list_mod6 = []


## Laço que executa cada modelo 100x ##
for i in range(100):

    print(f"\nEsta é a {i + 1}ª execução!")

    ## Random Forest Modelo 4 ##
    myrf_mod4 = RandomForestClassifier(n_estimators= 100)
    myrf_mod4.fit(X_treino_mod4, y_treino_mod4)
    randomf_mod4 = myrf_mod4.score(X_teste_mod4, y_teste_mod4)
    print("Rf Mod 4:", randomf_mod4)
    list_mod4.append(randomf_mod4)

    ## Random Forest Modelo 5 ##
    myrf_mod5 = RandomForestClassifier(n_estimators= 100)
    myrf_mod5.fit(X_treino_mod5, y_treino_mod5)
    randomf_mod5 = myrf_mod5.score(X_teste_mod5, y_teste_mod5)
    print("Rf Mod 5:", randomf_mod5)
    list_mod5.append(randomf_mod5)
    
    ## Random Forest Modelo 6 ##
    myrf_mod6 = RandomForestClassifier(n_estimators= 100)
    myrf_mod6.fit(X_treino_mod6, y_treino_mod6)
    randomf_mod6 = myrf_mod6.score(X_teste_mod6, y_teste_mod6)
    print("Rf Mod 6:", randomf_mod6)
    list_mod6.append(randomf_mod6)


## Exibindo as listas com os dados das 100 execuções e anexar dos valores médios na lista acima ##
print("\n---------------------------")
print("\nValores Random Forest Modelo 4:", list_mod4)
print("\nValores Random Forest Modelo 5:", list_mod5)
print("\nValores Random Forest Modelo 6:", list_mod6)
print("\n---------------------------\n")


print("\nO valor médio da Random Forest Modelo 4 é:", mean(list_mod4))
print("O valor médio da Random Forest Modelo 5 é:", mean(list_mod5))
print("O valor médio da Random Forest Modelo 6 é::", mean(list_mod6))