# -*- coding: utf-8 -*-

#Решение задачи распознавания рукописных цифр при помощи нейронной сети
#Работу выполнил Михеев Дмитрий Васильевич

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import seaborn as sns
from sklearn import metrics


count_train = 700 # Размер  обучающей выборки
count_test = 300 # Размер тестовой выборки
epochs = 400 # Количество эпох (количество обучений)
learning_rate = 1 # Коэффициент обучения
classes =  10 # Количество нейронов в  1 слое сети

max_val, min_val = 1, -1
range_size = (max_val - min_val)

#Функция построения изображений для распознавания
def plot_images(images, titles, columns=5, rows=1, fontsize=20):
    fig=plt.figure(figsize=(columns, rows))
    for i, img in enumerate(images[:columns*rows]):
        fig.add_subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.title(titles[i], fontsize=fontsize)
        plt.imshow(img, cmap='gray')
    plt.show()

#Функция активации (условие, при котором нейрон передаст сигнал в следующий слой сети)
def activation(x):
    return np.where(x>=0, 1, 0)

#Загрузка изображений из внешнего ресурса и их отрисовка при помощи функции  построения
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('\n' 'Обучающая выборка')
plot_images(X_train[:5], y_train[:5])
print('\n' 'Тестовая выборка')
plot_images(X_test[:5], y_test[:5])

X_train = X_train[:count_train]
y_train = y_train[:count_train]
X_test = X_test[:count_test]
y_test = y_test[:count_test]

#Видоизменяем матрицу каждого изображения в тренировочной выборке
X_train = X_train.reshape(count_train, 784) # 784 - это количество пикселей в изображении
X_train = X_train/255 #  255 - это количество оттенков черного цвета

#Видоизменяем матрицу каждого изображения в тестовой выборке
X_test = X_test.reshape(count_test, 784)
X_test = X_test/255

X_train = np.c_[X_train, np.ones(X_train.shape[0])]
X_test = np.c_[X_test, np.ones(X_test.shape[0])]

targets = np.array([y_train]).reshape(-1)
y_train = np.eye(classes)[targets]

#Генерация значений синаптических весов
W = np.random.rand(X_train.shape[1], classes) * range_size + min_val

#Обучение
for i in range(epochs):
  for xi, yi in zip(X_train, y_train):
    out_y = activation(np.dot(xi, W))  
    error = yi - out_y 
    error = error.reshape(1, 10)
    D = learning_rate * np.dot(xi[np.newaxis, :].T, error)  
    W = W + D

y_tmp_prediction = activation(np.dot(X_test, W))
y_predict = []
for i in y_tmp_prediction:
  y_predict.append(np.argmax(i))

#Вывод точности распознавания и матрицы  ошибок
print('\n' 'Точность = ', metrics.accuracy_score(y_test, y_predict))
print('\n' 'Матрица ошибок')
plt.figure()
sns.heatmap(metrics.confusion_matrix(y_test, y_predict), annot=True)


