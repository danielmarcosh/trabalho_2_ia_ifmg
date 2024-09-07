import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pre_processamento import load_and_preprocess_data


# Carregar e pré-processar os dados
feature_vectors, labels = load_and_preprocess_data()

# Dividir os dados em conjunto de treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

# Treinar a SVM usando os vetores de características e suas classes
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm_model.predict(X_test)

# Avaliar o modelo
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=1))

print("Acurácia:", accuracy_score(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')  # Salvar a imagem
plt.close()  # Fechar a figura

# Salvar o modelo treinado
joblib.dump(svm_model, 'svm_model.pkl')

# 'perfeito': 0, 
# 'medio': 1, 
# 'ruim': 2
