import tempfile
import time
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from gtts import gTTS
import playsound
import chromadb
from chromadb.config import Settings
import speech_recognition as sr
import PyPDF2
from groq import Groq
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class InterviewAssistant:
    def __init__(self, api_key, pdf_path, collection_name="interview_answers",n_q = 2,duration= 30):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.pdf_path = pdf_path
        self.chromadb_client = chromadb.Client(Settings())
        self.collection = self.chromadb_client.create_collection(collection_name)
        self.resume = self.extract_text_from_pdf(pdf_path)
        self.n_q = n_q
        self.duration = duration
        
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def retrieve_all_q_a(self):
        results = self.collection.get(include=["metadatas", "documents"])
        q_a_dict = {}
        if not results['documents']:
            print("No entries found in the database.")
            return q_a_dict
        for metadata, document in zip(results['metadatas'], results['documents']):
            question = metadata['question']
            answer = document
            q_a_dict[question] = answer
        return q_a_dict

    def text_to_speech(self, text, lang='en'):
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_audio_file:
            tts.save(temp_audio_file.name)
            playsound.playsound(temp_audio_file.name)

    def record_answer(self, threshold=0.1, fs=44100):
        print("Recording your answer...")
        audio = []
        start_time = time.time()
        while (time.time() - start_time) < self.duration:
            data = sd.rec(int(fs * 3), samplerate=fs, channels=1, blocking=True)
            audio.append(data)
            if np.max(np.abs(data)) < threshold:
                print("Silence detected, stopping recording.")
                break
        audio = np.concatenate(audio, axis=0)
        print("Recording finished.")
        return audio

    def convert_audio_to_text(self, audio_data, fs=44100):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
            wavfile.write(temp_wav_file.name, fs, (audio_data * 32767).astype(np.int16))
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav_file.name) as source:
                audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return None

    def generate_question(self, context):
        prompt = f"""
        Based on the project details in this resume: {self.resume}, and the previous questions asked: {context}, 
        create a concise and formal technical question related to the candidate's projects, ensuring it remains professional without any extra phrasing.
        Ask the question directly in less than 15 words.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an interviewer who asks only formal and concise questions."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            max_tokens=50
        )
        return chat_completion.choices[0].message.content.strip()

    def analyze_responses(self, list_n):
        prompt = f"""
        Based on the following list of question-answer pairs, create an overall analysis of the person's performance. provide specific areas where they can improve. Offer constructive feedback in a professional tone, focusing on actionable improvements in communication, technical understanding, or any other relevant aspects and give an example of how they could have answered the questioned that were asked to them in a detailed and professional way output should be written in a way that you are giving feedback to the person the evaluation of the person should be comprehensiz.
        
        Question-Answer pairs: {list_n}
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an evaluator providing a performance review"},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            max_tokens=500
        )
        return chat_completion.choices[0].message.content.strip()

    def conduct_interview(self):
        context_old = self.retrieve_all_q_a()
        for _ in range(self.n_q):
            question = self.generate_question(context_old)
            self.text_to_speech(question)

            audio_answer = self.record_answer()
            answer_text = self.convert_audio_to_text(audio_answer)

            if answer_text:
                self.collection.add(documents=[answer_text], metadatas=[{"question": question}], ids=[str(hash(question))])
        all_ques_ans = self.retrieve_all_q_a()
        list_n = [{'Question': q, 'Answer': a} for q, a in all_ques_ans.items()]
        insight = self.analyze_responses(list_n)
        print(insight)
        self.text_to_speech(insight)

pdf_path = "ak.pdf"

interview_assistant = InterviewAssistant(api_key, pdf_path,n_q = 2,duration=1000)
interview_assistant.conduct_interview()