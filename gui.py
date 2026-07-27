import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import queue
import logging

# --- Graceful Dependency Imports ---
try:
    import pyttsx3
    TTS_ENABLED = True
except ImportError:
    TTS_ENABLED = False
    logging.warning("pyttsx3 not found. AI voice output will be disabled. `pip install pyttsx3`")

try:
    import speech_recognition as sr
    STT_ENABLED = True
except ImportError:
    STT_ENABLED = False
    logging.warning("SpeechRecognition not found. Voice input will be disabled. `pip install SpeechRecognition`")

# --- Project-Specific Imports ---
# These imports assume the GUI is run from the project root.
from qa_system import QuestionAnsweringSystem
from book_manager import BookManager
from memory_manager import Memory