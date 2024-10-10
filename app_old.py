import random, nltk, numpy as np, json, pickle

from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialiser Flask
app = Flask(__name__)

# Charger les fichiers et modèles
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.h5')

# Fonction pour nettoyer et lemmatiser la phrase
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Fonction pour créer un bag of words
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Fonction pour prédire l'intention
def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Fonction pour obtenir la réponse
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Route principale pour rendre la page HTML
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.json["message"]
    print("User message:", user_message)  # Débogage
    try:
        # Prédire l'intention
        predicted_intents = predict_class(user_message, model)
        print("Predicted intents:", predicted_intents)  # Débogage

        # Obtenir la réponse basée sur les intentions
        response = get_response(predicted_intents, intents)
        return jsonify({"response": response})
    except Exception as e:
        print("Error:", str(e))  # Afficher l'erreur
        return jsonify({"response": "Sorry, something went wrong."})

if __name__ == "__main__":
    app.run(debug=True)