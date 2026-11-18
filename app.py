import html
from pathlib import Path

import gradio as gr
import joblib

MODEL_PATH = Path(__file__).parent / "linguadetect_model.joblib"

LANGUAGES = {
    "rus": "Russian",
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "nld": "Dutch",
    "pol": "Polish",
    "tur": "Turkish",
}

SUPPORTED_LANGUAGES = [
    ("🇷🇺", "Russian", "Русский"),
    ("🇬🇧", "English", "Английский"),
    ("🇩🇪", "German", "Немецкий"),
    ("🇫🇷", "French", "Французский"),
    ("🇪🇸", "Spanish", "Испанский"),
    ("🇮🇹", "Italian", "Итальянский"),
    ("🇵🇹", "Portuguese", "Португальский"),
    ("🇳🇱", "Dutch", "Нидерландский"),
    ("🇵🇱", "Polish", "Польский"),
    ("🇹🇷", "Turkish", "Турецкий"),
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file is missing. Please add linguadetect_model.joblib "
        "to the same folder as app.py."
    )

model = joblib.load(MODEL_PATH)


def run_detection(text):
    if not isinstance(text, str) or not text.strip():
        return "Please enter some text.", "Your result will appear here."

    code = model.predict([text.strip()])[0]
    language = LANGUAGES.get(code, "Unknown language")
    return language, f"LinguaDetect identified the language as **{language}**."

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.gradio-container { max-width: 1150px !important; margin: auto !important; font-family: 'DM Sans', sans-serif !important; background: #f8f7fc !important; }
#main { max-width: 1050px; margin: auto; }
#hero { background: linear-gradient(130deg, #172554, #293b75, #514b83); color: white; padding: 55px 35px; border-radius: 25px; text-align: center; margin: 20px 0; }
#hero h1 { font-family: 'Playfair Display', serif; font-size: 52px; margin-bottom: 10px; }
#hero p { font-size: 18px; color: #e5e1f4; }
#hero .tag { color: #f4cce5; letter-spacing: 2px; font-size: 12px; font-weight: 700; }
#detector { background: white; padding: 28px; border-radius: 20px; border: 1px solid #e9e5f2; box-shadow: 0 8px 25px rgba(35,35,80,.06); }
#detect-btn button { background: linear-gradient(90deg, #b77da9, #d18bb9) !important; color: white !important; border: none !important; border-radius: 12px !important; min-height: 48px; font-weight: 700; }
#result { background: #f0edf9; border-radius: 14px; padding: 15px; }
.section-title { color: #202b56; font-family: 'Playfair Display', serif; font-size: 30px; margin: 45px 0 10px; }
.section-subtitle { color: #77778e; margin-bottom: 20px; }
.language-card, .info-card { background: white; border: 1px solid #e9e5f2; border-radius: 16px; padding: 18px; text-align: center; box-shadow: 0 4px 14px rgba(35,35,80,.04); }
.language-flag { font-size: 30px; margin-bottom: 8px; }
.language-name { font-weight: 700; color: #202b56; }
.language-translation { color: #89899e; font-size: 13px; }
.info-card { text-align: left; height: 100%; }
.info-card h3 { color: #293b75; }
.info-card p { color: #77778e; line-height: 1.6; }
#footer { background: #172554; color: white; text-align: center; padding: 25px; border-radius: 18px; margin-top: 40px; }
textarea { border-radius: 12px !important; }
@media (max-width: 600px) { #hero { padding: 35px 18px; } #hero h1 { font-size: 36px; } #detector { padding: 16px; } }
"""

language_cards = "".join(
    f'<div class="language-card"><div class="language-flag">{flag}</div>'
    f'<div class="language-name">{name}</div>'
    f'<div class="language-translation">{translation}</div></div>'
    for flag, name, translation in SUPPORTED_LANGUAGES
)

languages_html = f"""
<div class="section-title">Supported Languages</div>
<p class="section-subtitle">LinguaDetect can identify text written in these 10 languages.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:14px;">{language_cards}</div>
"""

how_it_works = """
<div class="section-title">How It Works</div>
<p class="section-subtitle">Three simple steps to identify a language.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;">
<div class="info-card"><h3>01. Enter Text</h3><p>Type or paste a text sample into the input field.</p></div>
<div class="info-card"><h3>02. Analyze</h3><p>The machine learning model analyzes the text using TF-IDF features and LinearSVC.</p></div>
<div class="info-card"><h3>03. Get Result</h3><p>LinguaDetect identifies the language and displays the result.</p></div>
</div>
"""

features_html = """
<div class="section-title">Features</div>
<p class="section-subtitle">What makes LinguaDetect useful?</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;">
<div class="info-card"><h3>10 Languages</h3><p>Supports Russian, English, German, French, Spanish, Italian, Portuguese, Dutch, Polish and Turkish.</p></div>
<div class="info-card"><h3>Machine Learning</h3><p>Uses a LinearSVC classification model with TF-IDF feature extraction.</p></div>
<div class="info-card"><h3>Simple Interface</h3><p>An easy-to-use interface for quick language identification.</p></div>
<div class="info-card"><h3>Fast Detection</h3><p>Get a language prediction after submitting your text.</p></div>
</div>
"""

about_html = """
<div class="section-title">About LinguaDetect</div>
<div class="info-card"><h3>About the Project</h3>
<p>LinguaDetect is a machine learning project designed to identify the language of a given text. It was developed using the WiLI-2018 dataset.</p>
<p>The system uses TF-IDF for text feature extraction and LinearSVC for language classification.</p>
<p>The project currently supports 10 languages and provides a simple interface for users.</p></div>
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="pink"),
    css=custom_css,
    title="LinguaDetect",
) as demo:
    with gr.Column(elem_id="main"):
        gr.HTML("""<div id="hero"><div class="tag">AI-POWERED LANGUAGE IDENTIFICATION</div><h1>✦ LinguaDetect</h1><p>One text. One click. Discover the language.</p></div>""")
        with gr.Column(elem_id="detector"):
            gr.Markdown("## Identify a Language")
            gr.Markdown("Enter your text below to discover its language.")
            text_input = gr.Textbox(label="Your text", placeholder="Type or paste your text here...", lines=6, max_lines=12)
            with gr.Row():
                detect_button = gr.Button("Detect Language", variant="primary", elem_id="detect-btn")
                clear_button = gr.ClearButton(components=[text_input], value="Clear")
            detected_language = gr.Textbox(label="Detected Language", interactive=False)
            result_message = gr.Markdown(elem_id="result")
        gr.Markdown("## Try These Examples")
        gr.Examples(
            examples=[
                ["Hello! How are you today?"], ["Привет! Как твои дела?"],
                ["Hallo! Wie geht es dir?"], ["Bonjour! Comment allez-vous?"],
                ["¡Hola! ¿Cómo estás?"], ["Ciao! Come stai?"],
                ["Olá! Como você está?"], ["Hallo! Hoe gaat het met je?"],
                ["Cześć! Jak się masz?"], ["Merhaba! Nasılsın?"],
            ], inputs=text_input
        )
        gr.HTML(languages_html)
        gr.HTML(how_it_works)
        gr.HTML(features_html)
        gr.HTML(about_html)
        gr.HTML("""<div id="footer"><h2>LinguaDetect</h2><p>TF-IDF + LinearSVC | 10 Supported Languages</p><p>Machine Learning Language Identification</p></div>""")
    detect_button.click(fn=run_detection, inputs=text_input, outputs=[detected_language, result_message])
    text_input.submit(fn=run_detection, inputs=text_input, outputs=[detected_language, result_message])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
