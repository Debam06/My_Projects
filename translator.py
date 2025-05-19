import tkinter as tk
from deep_translator import GoogleTranslator

def translate_word():
    """Translates the input English word into the selected languages."""
    word = entry.get().strip()
    if not word:
        output_label.config(text="Please enter an English word.")
        return

    languages = {
        "Russian": "ru",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Japanese": "ja"
    }
    
    translations = []
    
    for language, lang_code in languages.items():
        try:
            translated_text = GoogleTranslator(source='en', target=lang_code).translate(word)
            translations.append(f"{language}: {translated_text}")
        except Exception as e:
            translations.append(f"{language}: Error - {e}")
    
    output_label.config(text="\n".join(translations))

def clear_fields():
    """Resets the input field and the translation output."""
    entry.delete(0, tk.END)
    output_label.config(text="Translations will appear here")

# Create the main window.
root = tk.Tk()
root.title("Word Translator")
root.geometry("600x300")

# Create two frames: one for input/buttons and one for output.
left_frame = tk.Frame(root, padx=20, pady=20)
left_frame.grid(row=0, column=0, sticky="nsew")
right_frame = tk.Frame(root, padx=20, pady=20)
right_frame.grid(row=0, column=1, sticky="nsew")

# Make sure both frames expand equally.
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# --- Left Frame: Input Section ---
input_label = tk.Label(left_frame, text="Enter an English word:", font=("Arial", 12))
input_label.pack(pady=(0, 10))

entry = tk.Entry(left_frame, width=30, font=("Arial", 12))
entry.pack(pady=(0, 10))

translate_button = tk.Button(left_frame, text="Translate", font=("Arial", 12), command=translate_word)
translate_button.pack(pady=(0, 10))

clear_button = tk.Button(left_frame, text="Clear", font=("Arial", 12), command=clear_fields)
clear_button.pack(pady=(0, 10))

# --- Right Frame: Output Section ---
output_label = tk.Label(right_frame, text="Translations will appear here", justify="left", anchor="nw", font=("Arial", 12))
output_label.pack(fill="both", expand=True)

# Start the GUI event loop.
root.mainloop()