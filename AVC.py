# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 16:05:11 2024

@author: rem20
"""

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk

class Dataset:
    def __init__(self, data_path):
        # Charger les données depuis le fichier CSV en spécifiant le séparateur correct
        self.data = pd.read_csv(data_path, delimiter=',')
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.label_encoders = {}  # Ajout pour stocker les encodeurs
    
    def clean_data(self):
        # Supprimer les lignes où la colonne 'gender' est marquée 'Other'
        self.data = self.data[self.data['gender'] != 'Other']
        # Convertir 'N/A' en NaN pour la colonne 'bmi' et supprimer les lignes avec NaN
        self.data['bmi'] = pd.to_numeric(self.data['bmi'], errors='coerce')
        self.data = self.data.dropna(subset=['bmi'])
        self.data = self.data[self.data['smoking_status'] != 'Unknown']
        self.data = self.data.dropna()
        
    def visualize_correlation_matrix(self):
        data_for_correlation = self.data.drop(columns=['id'])
        corr_matrix = data_for_correlation.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Matrice de corrélation')
        plt.show()

    def check_normality(self, column_names):
        for col in column_names:
            if col in self.data.columns:
                plt.figure(figsize=(12, 6))
                sns.histplot(self.data[col], kde=True, color='blue', stat="density", linewidth=0)
                plt.title(f'Distribution de {col}')
                stat, p = stats.shapiro(self.data[col].dropna())
                plt.show()
                print(f'Résultat du test de normalité pour {col}: Statistique={stat:.3f}, p-value={p:.3f}')
                if p > 0.05:
                    print(f'La distribution de {col} semble suivre une loi normale (ne rejette pas H0)\n')
                else:
                    print(f'La distribution de {col} ne semble pas suivre une loi normale (rejette H0)\n')

    def preprocess_data(self):
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for feature in categorical_features:
            le = LabelEncoder()
            self.data[feature] = le.fit_transform(self.data[feature])
            self.label_encoders[feature] = le

        X = self.data.drop(['stroke', 'id'], axis=1).values
        y = self.data['stroke'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class Model:
    def __init__(self, dataset):
        self.X_train = dataset.X_train
        self.X_test = dataset.X_test
        self.y_train = dataset.y_train
        self.y_test = dataset.y_test
        self.accuracies = []
        self.best_k = 1
        self.label_encoders = dataset.label_encoders  

    def train_and_evaluate(self, max_k):
        max_accuracy = 0
        for k in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            y_pred = knn.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            self.accuracies.append(acc)
            if acc > max_accuracy:
                max_accuracy = acc
                self.best_k = k

        self.calibrate_model()

    def calibrate_model(self):
        self.knn_best = KNeighborsClassifier(n_neighbors=self.best_k)
        self.knn_best.fit(self.X_train, self.y_train)
        y_pred_best = self.knn_best.predict(self.X_test)
        print(y_pred_best)
        best_accuracy = accuracy_score(self.y_test, y_pred_best)
        print(f"Le meilleur nombre de voisins est {self.best_k} avec un taux de réussite de {best_accuracy:.2f}")
    
    def predict(self, new_data):
        new_data_encoded = []
        for feature, value in zip(['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'], new_data):
            if feature in self.label_encoders:
                le = self.label_encoders[feature]
                # Convertir la valeur en chaîne si ce n'est pas déjà le cas
                value_str = str(value)
                # Vérifiez si la valeur est dans les classes connues de l'encodeur
                if value_str in le.classes_:
                    encoded = le.transform([value_str])[0]
                else:
                    # Gérer les valeurs inconnues comme vous le jugez nécessaire
                    encoded = -1  # ou autre code pour valeurs inconnues
                new_data_encoded.append(encoded)
            else:
                new_data_encoded.append(value)
    
        new_data_encoded = np.array([new_data_encoded])  # Assurez-vous que c'est un tableau 2D avec une seule ligne
        return self.knn_best.predict(new_data_encoded)




class Visual:
    def __init__(self, model):
        self.accuracies = model.accuracies

    def plot_accuracies(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.accuracies) + 1), self.accuracies, marker='o')
        plt.title('Taux de réussite en fonction du nombre de voisins')
        plt.xlabel('Nombre de voisins')
        plt.ylabel('Taux de réussite')
        plt.grid(True)
        plt.show()

class StrokePredictionGUI:
    def __init__(self, model):
        self.model = model
        self.user_input = {}
        self.setup_gui()

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("Prédiction de risque d'AVC")

        labels = ['Gender', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 'Work Type', 'Residence Type', 'Avg Glucose Level', 'BMI', 'Smoking Status']
        self.entries = {}

        for i, label in enumerate(labels):
            tk.Label(self.window, text=label + ':').grid(row=i, column=0)
            if label in ['Gender', 'Hypertension', 'Heart Disease', 'Ever Married', 'Work Type', 'Residence Type', 'Smoking Status']:
                self.entries[label] = ttk.Combobox(self.window, state="readonly")
                self.entries[label].grid(row=i, column=1)
                self.entries[label]['values'] = ('Male', 'Female') if label == 'Gender' else \
                                               ('Yes', 'No') if label in ['Hypertension', 'Heart Disease', 'Ever Married'] else \
                                               ('Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked') if label == 'Work Type' else \
                                               ('Urban', 'Rural') if label == 'Residence Type' else \
                                               ('Formerly smoked', 'Never smoked', 'Smokes')
            else:
                self.entries[label] = tk.Entry(self.window)
                self.entries[label].grid(row=i, column=1)

        predict_button = tk.Button(self.window, text='Prédire', command=self.make_prediction)
        predict_button.grid(row=len(labels), column=1)
        self.window.mainloop()

    def collect_responses(self):
        for label, entry in self.entries.items():
            self.user_input[label] = entry.get()

    def format_user_input(self):
        formatted_input = []
        input_mapping = {
            'Gender': 'gender',
            'Age': 'age',
            'Hypertension': 'hypertension',
            'Heart Disease': 'heart_disease',
            'Ever Married': 'ever_married',
            'Work Type': 'work_type',
            'Residence Type': 'Residence_type',
            'Avg Glucose Level': 'avg_glucose_level',
            'BMI': 'bmi',
            'Smoking Status': 'smoking_status'
        }

        for gui_label, model_feature in input_mapping.items():
            value = self.user_input[gui_label]
            if gui_label in ['Hypertension', 'Heart Disease']:
                value = 1 if value == 'Yes' else 0
            elif gui_label in ['Age', 'Avg Glucose Level', 'BMI']:
                value = float(value)
            elif model_feature in self.model.label_encoders:
                le = self.model.label_encoders[model_feature]
                value = le.transform([value])[0] if value in le.classes_ else le.transform([le.classes_[0]])[0]
            
            formatted_input.append(value)
        
        return formatted_input

    def make_prediction(self):
        self.collect_responses()
        formatted_user_data = self.format_user_input()
        prediction = self.model.predict(formatted_user_data)

        # Afficher le résultat
        result_message = "Risque élevé d'AVC" if prediction[0] == 1 else "Risque faible d'AVC"
        messagebox.showinfo("Résultat de prédiction", result_message)

# Utilisation des classes
data_path = 'healthcare-dataset-stroke-data.csv'  # Assurez-vous que le chemin est correct
dataset = Dataset(data_path)
dataset.clean_data()  # Nettoie les données

# Visualiser la matrice de corrélation
dataset.visualize_correlation_matrix()

# Vérifier la normalité pour certaines colonnes
dataset.check_normality(['age', 'bmi', 'avg_glucose_level'])
dataset.preprocess_data()

model = Model(dataset)
model.train_and_evaluate(15)  # Vous pouvez ajuster ce nombre selon vos besoins

visual = Visual(model)
visual.plot_accuracies()

# Initialiser et lancer l'interface graphique
gui = StrokePredictionGUI(model)

