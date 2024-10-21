from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

diabetes = datasets.load_diabetes()
features = diabetes.data
targets = diabetes.target

featuresAll = []
featuresTar = []

for i in range(len(features)):
    featuresAll.append(features[i][2])  
    featuresTar.append(targets[i])  

featuresAll = np.array(featuresAll).reshape(-1, 1) 
featuresTar = np.array(featuresTar)


model = LinearRegression()
model.fit(featuresAll, featuresTar)

predictions = model.predict(featuresAll)

coef_angular = model.coef_[0]  
coef_linear = model.intercept_  

print(f"Coeficiente angular (slope): {coef_angular}")
print(f"Coeficiente linear (intercepto): {coef_linear}")

limite1 = np.percentile(targets, 33)  # 33% dos valores
limite2 = np.percentile(targets, 66)  # 66% dos valores

colors = []
for target in targets:
    if target <= limite1:
        colors.append('blue')  # Baixa progressão
    elif target <= limite2:
        colors.append('orange')  # Média progressão
    else:
        colors.append('red')  # Alta progressão

scatter = plt.scatter(featuresAll, featuresTar, c=colors, alpha=0.7, label='Dados reais')
plt.plot(featuresAll, predictions, color='black', label=f'Regressão Linear (y = {coef_angular:.2f}x + {coef_linear:.2f})')

blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Baixa Progressão', markerfacecolor='blue', markersize=10)
orange_patch = plt.Line2D([0], [0], marker='o', color='w', label='Média Progressão', markerfacecolor='orange', markersize=10)
red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Alta Progressão', markerfacecolor='red', markersize=10)

plt.legend(handles=[blue_patch, orange_patch, red_patch])

plt.title('Relação entre IMC e Progressão da Doença no Conjunto Diabetes')
plt.xlabel('IMC (Índice de Massa Corporal)')
plt.ylabel('Progressão da Doença')
plt.show()

novo_valor = [[30.0]] 
y_pred = model.predict(novo_valor)

print(f"\nPara um IMC de {novo_valor[0][0]}, o valor previsto de progressão da doença é: {y_pred[0]:.2f} unidades")
