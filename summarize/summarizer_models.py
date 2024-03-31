import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def summarize(input_csv, output_csv_prefix, selected_model):
    df = pd.read_csv(input_csv, header=None)
    dataset = df[0].tolist()

    processed_notes = []
    first_77_chars = []
    first_77_words = []

    for i, note in enumerate(dataset):
        print('INPUT NOTE:')
        print(note)

        if selected_model == 'flan-t5-base-samsum':
            model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-base-samsum").to(device)
            tokenizer = AutoTokenizer.from_pretrained("philschmid/flan-t5-base-samsum")
            input_ids = tokenizer(note, return_tensors='pt').input_ids.to(device)
            outputs = model.generate(input_ids)
            processed_note = tokenizer.decode(outputs[0])

        elif selected_model == 'flan-t5-base':
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
            prompt = 'Given the following medical note, please provide a summary that retains all essential medical information. Aim to preserve as much relevant information as possible, while ensuring that the token length does not exceed 77.\nMedical Note:\n'
            input_ids = tokenizer(prompt + note, return_tensors='pt').input_ids.to(device)
            outputs = model.generate(input_ids)
            processed_note = tokenizer.decode(outputs[0])

        elif selected_model in ['bart-large-samsum', 'bart-large-cnn-samsum']:
            if selected_model == 'bart-large-samsum':
                summarizer = pipeline("summarization", model="linydub/bart-large-samsum")
            else:
                summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
            processed_note = summarizer(note)[0]['summary_text']

        print('PROCESSED NOTE:')
        print(processed_note)
        processed_notes.append(processed_note)
        first_77_chars.append(processed_note[:77])
        first_77_words.append(" ".join(processed_note.split()[:77]))

    output_df = pd.DataFrame({
        'Original Note': dataset, 
        'Summarized Note': processed_notes,
        'First 77 Characters': first_77_chars,
        'First 77 Words': first_77_words
    })
    output_csv = f"{selected_model}_{output_csv_prefix}.csv"  # Changed the name format here
    output_df.to_csv(output_csv, index=False)
    print(f"Summarized notes saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "40_De-Identified_Clinical_Notes.csv"
    output_csv_prefix = "summarized_notes"
    # choose from flan-t5-base-samsum, flan-t5-base, bart-large-samsum, bart-large-cnn-samsum
    selected_model = 'bart-large-samsum'  # Change this to your desired model
    summarize(input_csv, output_csv_prefix, selected_model)
