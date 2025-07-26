import tkinter as tk
from tkinter import scrolledtext
from transformers import T5ForConditionalGeneration, T5Tokenizer
from spellchecker import SpellChecker
import threading

# Load model and tokenizer (takes time, so load once)
model_name = "vennify/t5-base-grammar-correction"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
spell = SpellChecker()

# Spell correct function
def correct_spelling(text):
    corrected_words = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)

# Processing function
def process_text():
    input_text = input_box.get("1.0", tk.END).strip()
    if not input_text:
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Please enter text.")
        return

    status_label.config(text="Processing...")

    # AI Processing in separate thread to avoid GUI freeze
    def run_correction():
        formatted_input = "correct: " + input_text
        inputs = tokenizer.encode(formatted_input, return_tensors="pt")

        outputs = model.generate(
            inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_output = correct_spelling(model_output)

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, f"Model Output:\n{model_output}\n\nFinal Corrected:\n{final_output}")
        status_label.config(text="Done!")

    threading.Thread(target=run_correction).start()

# Tkinter Window Setup
root = tk.Tk()
root.title("AI AutoCorrect Tool - Soumik's Project")

tk.Label(root, text="Enter Text Below:").pack()
input_box = scrolledtext.ScrolledText(root, width=60, height=10)
input_box.pack()

tk.Button(root, text="Correct Text", command=process_text).pack(pady=10)

tk.Label(root, text="Output:").pack()
output_box = scrolledtext.ScrolledText(root, width=60, height=10)
output_box.pack()

status_label = tk.Label(root, text="", fg="green")
status_label.pack(pady=5)

root.mainloop()
