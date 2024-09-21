const sendButton = document.getElementById('sendButton');
const inputBox = document.getElementById('inputBox');
const chatbox = document.getElementById('chatbox');

sendButton.addEventListener('click', function() {
  const userMessage = inputBox.value;
  
  // Ajouter le message utilisateur à l'interface
  const userMessageDiv = document.createElement('div');
  userMessageDiv.textContent = "You: " + userMessage;
  chatbox.appendChild(userMessageDiv);

  // Envoyer une requête POST au serveur Flask
  fetch("/get", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: userMessage }),
  })
  .then(response => response.json())
  .then(data => {
    // Ajouter la réponse du bot à l'interface
    const botResponseDiv = document.createElement('div');
    botResponseDiv.textContent = "Bot: " + data.response;
    chatbox.appendChild(botResponseDiv);
  });

  // Vider la zone de saisie
  inputBox.value = "";
});
