from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType, FloatType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


spark = SparkSession.builder \
    .appName("Student Depression Analysis") \
    .getOrCreate()

df = spark.read.csv('student_depression_dataset.csv', header=True, inferSchema=True)

df_cleaned = df.na.drop()

df_cleaned = df_cleaned.withColumn(
    "Sleep_Duration_Numeric",
    when(col("Sleep Duration") == "'Less than 5 hours'", 4) \
    .when(col("Sleep Duration") == "'5-6 hours'", 5.5) \
    .when(col("Sleep Duration") == "'7-8 hours'", 7.5) \
    .when(col("Sleep Duration") == "'More than 8 hours'", 9) \
    .otherwise(0)
)
df_cleaned = df_cleaned.withColumn(
    "Dietary_Habits_Numeric",
    when(col("Dietary Habits") == "Unhealthy", 1) \
    .when(col("Dietary Habits") == "Moderate", 2) \
    .when(col("Dietary Habits") == "Healthy", 3) \
    .otherwise(0)
)
df_cleaned = df_cleaned.withColumn(
    "Suicidal_Thoughts_Numeric",
    when(col("Have you ever had suicidal thoughts ?") == "Yes", 1) \
    .otherwise(0)
)
df_cleaned = df_cleaned.withColumn(
    "Family_History_Numeric",
    when(col("Family History of Mental Illness") == "Yes", 1) \
    .otherwise(0)
)
df_pd = df_cleaned.toPandas()

# Распределение по типам питания
sns.countplot(data=df_pd, x='Dietary Habits', hue='Depression', order=['Healthy', 'Moderate', 'Unhealthy'])
plt.title('Распределение случаев депрессии по типам питания')
plt.xlabel('Тип диеты')
plt.ylabel('Количество наблюдений')
plt.legend(title='Депрессия', labels=['Нет', 'Есть'])

# Распределение депрессии с учетом суицидальных мыслей
plt.figure(figsize=(12, 6))
sns.countplot(x='Depression', hue='Suicidal_Thoughts_Numeric', data=df_pd,
              palette={0: "#66c2a5", 1: "#fc8d62"})
plt.title('Распределение депрессии с учетом суицидальных мыслей', fontsize=16)
plt.xlabel('Депрессия')
plt.ylabel('Количество студентов')
plt.legend(title='Суицидальные мысли', labels=['Нет', 'Есть'])
plt.show()

# Корреляционная матрица
plt.figure(figsize=(12, 8))
numeric_cols = ['Depression', 'Sleep_Duration_Numeric', 'Dietary_Habits_Numeric',
                'Suicidal_Thoughts_Numeric', 'Family_History_Numeric', 
                'Academic Pressure', 'Financial Stress', 'CGPA']
corr_matrix = df_cleaned[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Корреляция между преобразованными признаками', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Цвета по депрессии
colors = ['green' if x == 0 else 'red' for x in df_pd['Depression']]

ax.scatter(df_pd['Age'], 
           df_pd['Sleep_Duration_Numeric'], 
           df_pd['Depression'],
           c=colors, alpha=0.5)

ax.set_xlabel('Возраст')
ax.set_ylabel('Продолжительность сна')
ax.set_zlabel('Депрессия')
ax.set_yticks([4, 5.5, 7.5, 9])
ax.set_yticklabels(['<5', '5-6', '7-8', '>8'])
ax.dist = 12
plt.title('3D визуализация влияния возраста и сна на депрессию')
plt.show()

# распределение депрессии по группам сна
plt.figure(figsize=(12, 12))
sns.countplot(data=df_pd, x='Sleep Duration', hue='Depression')
plt.title('Распределение депрессии по группам сна')
plt.xlabel('Продолжительность сна')
plt.ylabel('Количество наблюдений')
plt.legend(title='Депрессия', labels=['Нет', 'Есть'])

# распределение студентов по наличию депресии
plt.figure(figsize=(10, 6))
depression_counts = df_pd['Depression'].value_counts(normalize=True) * 100
depression_labels = ['No Depression', 'Depression']
plt.pie(depression_counts, labels=depression_labels, autopct='%1.1f%%', 
        colors=['#66c2a5', '#fc8d62'], startangle=90)
plt.title('Распределение студентов по наличию депрессии', fontsize=16)
plt.show()

# связь удолетворенности обучением с депресией
plt.figure(figsize=(10, 6))
satisfaction_depression = df_pd.groupby('Study Satisfaction')['Depression'].mean().reset_index()
sns.lineplot(x='Study Satisfaction', y='Depression', data=satisfaction_depression, 
             marker='o', linewidth=2.5, color='#d73027')
plt.title('Связь удовлетворенности обучением с депрессией', fontsize=16)
plt.xlabel('Удовлетворенность обучением (1-5)')
plt.ylabel('Вероятность депрессии')
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
