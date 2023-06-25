import speech_recognition as sr
import pyttsx3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#import openai
# Set up your OpenAI API credentials
#openai.api_key = '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'

# Define the questions and answers for the assistant
qa_pairs = [
    {
        'question': 'What is your return policy?',
        'answer': 'Our return policy allows customers to return the product within 30 days of purchase for a full refund.'
    },
    {
        'question': 'How can I track my order?',
        'answer': 'To track your order, please visit our website and go to the order tracking page. Enter your order number and email address to get the latest updates.'
    },
    {
        'question':'When will Big Billion Days Start',
        'answer':'It Around first  week of October '
    },
    {
        'question':' Are there any discounts or promotions available for online shopping?',
        'answer':'Check the stores website or subscribe to their newsletter for updates on discounts and promotions. '
    },
    {
        'question': 'How To Change Payment Mode?',
        'answer': 'You can Add or Update payment method by selecting payment options.'
    },
    {
        'question': 'I Ordered a wrong item how can i stop the order',
        'answer': 'If you order wrong item you can cancel the order Incase the order is already shipped  and out of delivery you can return  the order.'
    },
    {
        'question':'Want to contact  Market seller ',
        'answer':'In zozo.in contains the seller store  Front name and address at which you can reach the seller. '
    },
    {
       'question':'How To Cancel Order ',
        'answer':'To cancel an order, contact customer support Call 0431 6380122. '
    },
    {
        'question': 'Are there any shipping charges for online orders?',
        'answer': 'Shipping charges may apply depending on the stores policy, order value, and shipping destination.'
    },
    {
        'question': 'How long will it take for my order to arrive??',
        'answer': 'The delivery time may vary depending on your location and the shipping method chosen, typically ranging from a few days to a couple of           weeks..'
    },
    {
        'question':'Can I change my shipping address after placing an online order?',
        'answer':'Please Contact customer support immediately to inquire about address changes. '
    },
 
 
  
    # Add more question-answer pairs here
]

# Generate a list of questions and answers
questions = [pair['question'] for pair in qa_pairs]
answers = [pair['answer'] for pair in qa_pairs]

# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Set the speech rate (speed) of the assistant
engine.setProperty('rate', 150)

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_answer(question):
    # Compute cosine similarity between question and answer pairs
    question_embedding = model.encode([question])[0]
    answer_embeddings = model.encode(answers)
    similarities = cosine_similarity([question_embedding], answer_embeddings)

    # Find the most similar answer
    most_similar_idx = similarities.argmax()
    answer = answers[most_similar_idx]

    return answer

# Function to convert speech to text
def listen_for_speech():
    with sr.Microphone() as source:
        print("Assistant: Speak now...")
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        print("User:", user_input)
        return user_input
    
    except sr.UnknownValueError:
        print("Assistant: Sorry, I couldn't understand your speech.")
    except sr.RequestError as e:
        print(f" Assistant: Recognition request failed: {e}")

    return None

# Function to convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Chat loop
speak_text("Assistant: Hi! How can I assist you today?")
while True:
    user_input = listen_for_speech()

    if user_input is None:
        continue

    if user_input.lower() in ['exit', 'quit', 'bye']:
        speak_text(" Assistant: Goodbye!")
        break

    answer = get_answer(user_input)
    print(answer)
    speak_text(" Assistant: " + answer+"Hope it will be helpful")
   
