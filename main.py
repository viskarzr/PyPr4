from statistics import correlation

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# sns.set_theme(style="whitegrid", palette="pastel")
# plt.rcParams['figure.figsize'] = (12,8)
# print("Анализ данных о размерах лепестков и чашелистников для различных типов цветов ириса")
# print("="*60)
#
# #1. Загрузка встроенного датасета
# iris = sns.load_dataset('iris')
# print(f"Размер датасета: {iris.shape}")
# print(f"\nПервые пять строк данных: ")
# print(iris.head())
# print(f"\nИнформация о данных: ")
# print(iris.info())
# print(f"\nПропущенные значения: ")
# print(iris.isnull().sum())
#
# #2. Количество
# print("\n" + "="*60)
# print("1. ВИЗУАЛИЗАЦИЯ КОЛИЧЕСТВА")
# print("="*60)
#
# plt.figure(figsize=(15,10))
# plt.subplot(2,3,1)
# sns.countplot(data=iris, x='species', palette=['lightcoral', 'lightgreen'])
# plt.title('Общее количество лепестков и чашелистников')
# plt.xlabel('Выжил (0 - нет, 1 - да)')
# plt.ylabel('Количество')
# for i, count in enumerate(iris['species'].value_counts().sort_index()):
#     plt.text(i, count+5, str(count), ha='center', va='bottom', fontweight='bold')


sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (12, 8)

print("Анализ данных о размерах лепестков и чашелистников для различных типов цветов ириса")
print("="*60)
iris = sns.load_dataset('iris')
print(f"Размер датасета: {iris.shape}")
print(f"\nПервые пять строк данных: ")
print(iris.head())
print(f"\nИнформация о данных: ")
print(iris.info())
print(f"\nПропущенные значения: ")
print(iris.isnull().sum())


#1. Визуализация количства
print("\n" + "="*60)
print("1. ВИЗУАЛИЗАЦИЯ КОЛИЧЕСТВА")
print("="*60)
plt.figure(figsize=(15, 10))
# plt.subplot(2, 3, 1)
sns.countplot(data=iris, x='species', order=['setosa', 'versicolor', 'virginica'])
plt.title('Распределение видов ириса в датасете')
plt.xlabel('Вид ириса')
plt.ylabel('Количество образцов')

for i, count in enumerate(iris['species'].value_counts().sort_index()):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

#2. Визуализация количества длины чашелистника
print("\n" + "="*60)
print("2. Визуализация количества длины чашелистника")
print("="*60)
ax = sns.histplot(data=iris, x='sepal_length', hue='species', multiple='stack', palette='pastel')
plt.title('Распределение длины чашелистика по видам')
plt.xlabel('Длина чашелистика (см)')
plt.ylabel('Количество')

legend = ax.get_legend()
if legend:
    legend.set_title('Вид ириса')
plt.tight_layout()
plt.show()


# 3. Распрешеление длины и ширины лепестка
print("\n" + "="*60)
print("3. Распрешеление длины и ширины лепестка")
print("="*60)
sns.histplot( data=iris, x='petal_length', hue='species', kde=True, palette='viridis',alpha=0.6)
plt.title('Распределение длины лепестка по видам')
plt.xlabel('Длина лепестка (см)')
plt.ylabel('Количество')
plt.tight_layout()
plt.show()

#4. Корреляционная матрица
print("\n" + "="*60)
print("4. Корреляционная матрица")
print("="*60)
plt.figure(figsize=(10, 8))
numeric_data = iris.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap( correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
plt.title("Корреляционная матрица", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()