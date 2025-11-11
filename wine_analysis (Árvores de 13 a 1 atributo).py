from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from numpy import mean

## Carregando Wine Recognition Dataset ##
wine = datasets.load_wine()

print("\n Variáveis independentes:\n", wine.feature_names)
print("\n Classes das variáveis alvo (target):\n", wine.target_names)

## Exibindo a base de dados na tela ##
X = wine.data
print("\n Wine Data: ")
print(X)

## Classificando o 'target' e exibindo as classes ##
print("\nClassificação Alvo (0,1,2): ")
y = wine.target
print(y)

## Exibindo a quantidade de instâncias e de variáveis ##
print('\nVariáveis independentes:', X.shape)
print('Variáveis alvo (target)', y.shape)

## Fazendo o Split da base de treino e da base de teste ##
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size= 0.3, stratify= y, shuffle= True, random_state= None)

## Exibindo dados do treinamento ##
print("\n---------------------------")
print("\nX Treino: ", len(X_treino))
print("\nShape dos dados de treino: ", X_treino.shape, y_treino.shape)
print("\nX Treino veja:\n ", X_treino)
print("\ny Treino:\n ", y_treino)

## Exibindo dados de teste ##
print("---------------------------")
print("\nX Teste: ", len(X_teste))
print("\nShape dos dados de teste: ", X_teste.shape, y_teste.shape)
print("\nX Teste veja:\n ", X_teste)
print("\ny Teste:\n ", y_teste)
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
    ## Modelo RandomForest com todos os 13 atributos e 100 árvores ##
    myrf_13 = RandomForestClassifier(n_estimators= 100)
    myrf_13.fit(X_treino, y_treino)
    randomf_13atributes = myrf_13.score(X_teste, y_teste)
    print("Rf_13", randomf_13atributes)
    list_13atributes.append(randomf_13atributes)

    ## Modelo RandomForest com 12 atributos aleatórios e 100 árvores ##
    myrf_12 = RandomForestClassifier(n_estimators= 100, max_features= 12)
    myrf_12.fit(X_treino, y_treino)
    randomf_12atributes = myrf_12.score(X_teste, y_teste)
    print("Rf_12", randomf_12atributes)
    list_12atributes.append(randomf_12atributes)

    ## Modelo RandomForest com 11 atributos aleatórios e 100 árvores ##
    myrf_11 = RandomForestClassifier(n_estimators= 100, max_features= 11)
    myrf_11.fit(X_treino, y_treino)
    randomf_11atributes = myrf_11.score(X_teste, y_teste)
    print("Rf_11", randomf_11atributes)
    list_11atributes.append(randomf_11atributes)

    ## Modelo RandomForest com 10 atributos aleatórios e 100 árvores ##
    myrf_10 = RandomForestClassifier(n_estimators= 100, max_features= 10)
    myrf_10.fit(X_treino, y_treino)
    randomf_10atributes = myrf_10.score(X_teste, y_teste)
    print("Rf_10", randomf_10atributes)
    list_10atributes.append(randomf_10atributes)

    ## Modelo RandomForest com 9 atributos aleatórios e 100 árvores ##
    myrf_9 = RandomForestClassifier(n_estimators= 100, max_features= 9)
    myrf_9.fit(X_treino, y_treino)
    randomf_9atributes = myrf_9.score(X_teste, y_teste)
    print("Rf_9", randomf_9atributes)
    list_9atributes.append(randomf_9atributes)

    ## Modelo RandomForest com 8 atributos aleatórios e 100 árvores ##
    myrf_8 = RandomForestClassifier(n_estimators= 100, max_features= 8)
    myrf_8.fit(X_treino, y_treino)
    randomf_8atributes = myrf_8.score(X_teste, y_teste)
    print("Rf_8", randomf_8atributes)
    list_8atributes.append(randomf_8atributes)

    ## Modelo RandomForest com 7 atributos aleatórios e 100 árvores ##
    myrf_7 = RandomForestClassifier(n_estimators= 100, max_features= 7)
    myrf_7.fit(X_treino, y_treino)
    randomf_7atributes = myrf_7.score(X_teste, y_teste)
    print("Rf_7", randomf_7atributes)
    list_7atributes.append(randomf_7atributes)

    ## Modelo RandomForest com 6 atributos aleatórios e 100 árvores ##
    myrf_6 = RandomForestClassifier(n_estimators= 100, max_features= 6)
    myrf_6.fit(X_treino, y_treino)
    randomf_6atributes = myrf_6.score(X_teste, y_teste)
    print("Rf_6", randomf_6atributes)
    list_6atributes.append(randomf_6atributes)

    ## Modelo RandomForest com 5 atributos aleatórios e 100 árvores ##
    myrf_5 = RandomForestClassifier(n_estimators= 100, max_features= 5)
    myrf_5.fit(X_treino, y_treino)
    randomf_5atributes = myrf_5.score(X_teste, y_teste)
    print("Rf_5", randomf_5atributes)
    list_5atributes.append(randomf_5atributes)

    ## Modelo RandomForest com 4 atributos aleatórios e 100 árvores ##
    myrf_4 = RandomForestClassifier(n_estimators= 100, max_features= 4)
    myrf_4.fit(X_treino, y_treino)
    randomf_4atributes = myrf_4.score(X_teste, y_teste)
    print("Rf_4", randomf_4atributes)
    list_4atributes.append(randomf_4atributes)

    ## Modelo RandomForest com 3 atributos aleatórios e 100 árvores ##
    myrf_3 = RandomForestClassifier(n_estimators= 100, max_features= 3)
    myrf_3.fit(X_treino, y_treino)
    randomf_3atributes = myrf_3.score(X_teste, y_teste)
    print("Rf_3", randomf_3atributes)
    list_3atributes.append(randomf_3atributes)

    ## Modelo RandomForest com 2 atributos aleatórios e 100 árvores ##
    myrf_2 = RandomForestClassifier(n_estimators= 100, max_features= 2)
    myrf_2.fit(X_treino, y_treino)
    randomf_2atributes = myrf_2.score(X_teste, y_teste)
    print("Rf_2", randomf_2atributes)
    list_2atributes.append(randomf_2atributes)

    ## Modelo RandomForest com 1 atributos aleatórios e 100 árvores ##
    myrf_1 = RandomForestClassifier(n_estimators= 100, max_features= 1)
    myrf_1.fit(X_treino, y_treino)
    randomf_1atribute = myrf_1.score(X_teste, y_teste)
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