import streamlit as st, random, json, numpy as np, pickle, nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialiser le lemmatiseur
lemmatizer = WordNetLemmatizer()

# Charger les fichiers et modèles
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

# Interface Streamlit
st.title("🤖 AI Chatbot")
st.markdown("##### Posez une question au chatbot et obtenez une réponse instantanée!")

# Créer un container pour afficher l'historique de la conversation
conversation_container = st.container()

# Champ de saisie de l'utilisateur sans texte dans le champ
user_input = st.text_input("")  # Champ de saisie vide

# Stocker l'historique de la conversation
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Lorsque l'utilisateur envoie un message
if user_input:
    # Ajouter le message utilisateur à l'historique
    st.session_state['history'].append({"role": "user", "text": user_input})
    
    # Prédire l'intention et générer une réponse
    predicted_intents = predict_class(user_input, model)
    response = get_response(predicted_intents, intents)
    
    # Ajouter la réponse du chatbot à l'historique
    st.session_state['history'].append({"role": "chatbot", "text": response})

# Afficher l'historique de la conversation dans l'ordre
with conversation_container:
    for i in range(0, len(st.session_state['history']), 2):  # Nous prenons par paire les questions et réponses
        if i+1 < len(st.session_state['history']):  # Assurez-vous qu'il y a une réponse correspondante
            # Afficher la question dans une bulle verte
            st.markdown(f"<p style='text-align: right; border-radius: 15px; padding: 10px; background-color: green; color: white;'>{st.session_state['history'][i]['text']}</p>", unsafe_allow_html=True)
            # Afficher la réponse dans une bulle bleue
            st.markdown(f"<p style='border-radius: 15px; padding: 10px; background-color: blue; color: white;'>🤖 {st.session_state['history'][i+1]['text']}</p>", unsafe_allow_html=True)
