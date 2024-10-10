# IA Chatbot avec TensorFlow/Keras et Streamlit

Ce projet chatbot basé sur TensorFlow et Keras, utilise des intentions définies dans un fichier JSON (`intents.json`). Ce modèle est capable de répondre à des questions en fonction des intentions et des motifs de requêtes ajoutés au fichier JSON. Ce chatbot est conçu pour réentraîner automatiquement son modèle à partir des nouvelles données ajoutées quotidiennement au fichier `intents.json`.

## Fonctionnalités

- **Apprentissage des intentions** : Le chatbot est entraîné pour répondre à des questions sur la base d'un ensemble d'intentions pré-définies dans intents.json
- **Réentraînement quotidien** : Le modèle est automatiquement mis à jour et réentraîné pour inclure de nouvelles données ajoutées à `intents.json` chaque jour.
- **Traitement du Langage Naturel (NLP)** : Utilisation de la tokenisation, de la lemmatisation et du bag of words pour le traitement des entrées utilisateur.
- **Interface interactive** : Utilisation de Streamlit pour créer une interface utilisateur simple et interactive où l'utilisateur peut poser des questions et obtenir des réponses instantanées.

## Structure du projet

Voici un aperçu des principaux fichiers du projet :

- `chatbot_model.h5` : Le modèle TensorFlow/Keras sauvegardé après l'entraînement.
- `intents.json` : Le fichier JSON contenant les intentions, les motifs (patterns) et les réponses du chatbot.
- `words.pkl` : Les mots utilisés dans les motifs, après le processus de lemmatisation et de nettoyage.
- `classes.pkl` : Les classes (ou intentions) extraites du fichier `intents.json`.
- `chatbot.py` : Le fichier principal contenant le code pour le fonctionnement du chatbot, y compris le réentraînement quotidien.
- `app.py` : Le fichier pour lancer l'application Streamlit et interagir avec le chatbot.

## Prérequis

Avant de lancer le projet, assurez-vous d'avoir les dépendances suivantes installées :

- Python 3.x
- TensorFlow
- Keras
- NumPy
- NLTK
- Pickle
- Streamlit

Vous pouvez installer les dépendances avec la commande suivante :

```bash
pip install tensorflow keras numpy nltk pickle-mixin streamlit
```

## Description des étapes

### 1. Prétraitement des données

Le fichier `intents.json` contient des intentions (tags), des motifs (patterns) et des réponses. Le code extrait les mots clés de chaque motif, applique une lemmatisation pour simplifier les mots à leur forme de base, puis génère un **bag of words** pour chaque phrase.

### 2. Réentraînement automatique

Chaque fois que des modifications sont apportées au fichier `intents.json` (ajout de nouveaux motifs, intentions, ou réponses), le modèle est réentraîné pour prendre en compte ces nouvelles données. Voici les principales étapes du réentraînement :

- **Chargement des nouvelles données** : Le fichier `intents.json` est rechargé et analysé pour extraire les nouveaux motifs et intentions.
- **Mise à jour des fichiers pickle** : Les fichiers `words.pkl` et `classes.pkl` sont mis à jour pour stocker les nouveaux mots et intentions.
- **Réentraîner le modèle** : Un nouveau modèle est entraîné à partir des nouvelles données et est sauvegardé sous `chatbot_model.h5`.

### 3. Prédiction et génération de réponse

Lorsque l'utilisateur interagit avec le chatbot, les étapes suivantes sont effectuées :

- La phrase est tokenisée et lemmatisée.
- Un **bag of words** est généré à partir des mots présents dans la phrase.
- Le modèle prédit l'intention la plus probable de la phrase.
- Le chatbot génère une réponse en sélectionnant une réponse aléatoire associée à l'intention prédite.

### 4. Interface utilisateur avec Streamlit

L'interface utilisateur est développée avec Streamlit, où l'utilisateur peut poser des questions et obtenir des réponses instantanées. Les questions et réponses sont affichées dans des bulles de texte colorées (verte pour la question, bleue pour la réponse du chatbot).

## Exemple de code pour réentraîner le modèle

Voici un extrait de code pour réentraîner le modèle chaque jour avec les nouvelles données ajoutées au fichier `intents.json` :

```python
import random, json, pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Charger les nouvelles données intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Initialisation des listes
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Boucle pour extraire les mots et classes depuis intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatisation et nettoyage des mots
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

classes = sorted(set(classes))

# Sauvegarder les nouveaux fichiers words.pkl et classes.pkl
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Création de l'entraînement des données
training = []
output_empty = [0] * len(classes)

# Créer le bag of words pour chaque pattern
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)
  
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Réentraîner le modèle
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilation du modèle
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entraînement du modèle
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Sauvegarder le nouveau modèle
model.save('chatbot_model.h5')

print("Réentraînement terminé et modèle sauvegardé.")
```

## Lancement du projet avec Streamlit

Pour lancer l'application et interagir avec le chatbot via l'interface Streamlit, exécutez le fichier `app.py` :

```bash
streamlit run app.py
```

L'application Streamlit se lancera dans votre navigateur, vous permettant de poser des questions et d'obtenir des réponses instantanées avec l'interface interactive.
