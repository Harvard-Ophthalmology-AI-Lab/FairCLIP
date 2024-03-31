# https://huggingface.co/m42-health/med42-70b
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import pandas as pd
import openai
import argparse
import os
import numpy as np

CACHE_DIR = ''

def preprocess(model_name, dataset, mode, dataset_path):
    if model_name == 'med42':
        tokenizer = AutoTokenizer.from_pretrained("m42-health/med42-70b", use_fast=False, cache_dir=CACHE_DIR) # llama-2 model fine-tuned for medical domain
        model = AutoModelForCausalLM.from_pretrained("m42-health/med42-70b", torch_dtype=torch.float16, cache_dir=CACHE_DIR, device_map='auto')
    elif model_name == 'palmyra-med':
        tokenizer = AutoTokenizer.from_pretrained("Writer/palmyra-med-20b", use_fast=False, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained("Writer/palmyra-med-20b", device_map="auto", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    elif model_name == 'pmc-llama':
        tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B', cache_dir=CACHE_DIR)
        model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B', cache_dir=CACHE_DIR, device_map='auto')
        
    if mode == 'deidentify':
        prompts = [
                    'Given the following medical note, repeat every word verbatim, apart from sensitive Personal Health Information (PHI) such as names, addresses, and dates, which should be replaced with the tag [REDACTED].\nMedical Note:\n',
                    '''Given the following medical note, repeat every word verbatim, apart from sensitive Personal Health Information (PHI) such as names, addresses, and dates, which should be replaced with the tag [REDACTED]. Here are some input and output example pairs for reference: Input example 1: 62 y.o. Mike works as investor, forestry by training 1. Glaucoma Suspect OD / Advanced Primary Open Angle Glaucoma OS - Tmax (high teens OU by patient history prior to treatment); thin CCT (447/424); +FHx of glaucoma (mother and maternal uncle) - Diagnosed 2010 while living in Birmingham, AL and followed by Dr. Mike Fintch at UAB prior to moving to Boston and establishing care at MEE 2/2017 - s/p SLT OS in 2014 by Dr. Girkin; s/p SLT #2 OS (8/25/17) --> good response to SLT #2 - no facial/ocular trauma - no chronic steroid use (injections, oral, nasal sprays, creams) - HVF full OD and IA > SA OS encroaching fixation - OCT-RNFL (94/45) with diffuse thinning OS - per old notes in AL, IOPs have been ranging 8-12 OU (12/2015-5/2016) - seen by Dr. Mike Fintch (Neuro-op NY) who believes likely unilateral NTG OS - Tg mid teens OD; low teens OS - at goal today OU with VF stability 2. Myopia OU - followed previously by optometrist, Dr. John  Fintch in NY- continue current MRx 3. Trace cataract OU - not visually significant - follow Social/systemic: Marginal HTN, PLAN - continue Latanoprost QHS OU - continue Combigan BID OS RTC 10 months IOP check. Output example 1: [REDACTED] y.o. [REDACTED] works as investor, forestry by training 1. Glaucoma Suspect OD / Advanced Primary Open Angle Glaucoma OS - Tmax (high teens OU by patient history prior to treatment); thin CCT (447/424); +FHx of glaucoma (mother and maternal uncle) - Diagnosed [REDACTED] while living in [REDACTED] and followed by Dr. [REDACTED] at [REDACTED] prior to moving to [REDACTED] and establishing care at [REDACTED] - s/p SLT OS in [REDACTED] by Dr. [REDACTED]; s/p SLT #2 OS ([REDACTED]) --> good response to SLT #2 - no facial/ocular trauma - no chronic steroid use (injections, oral, nasal sprays, creams) - HVF full OD and IA > SA OS encroaching fixation - OCT-RNFL (94/45) with diffuse thinning OS - per old notes in AL, IOPs have been ranging 8-12 OU ([REDACTED]-[REDACTED]) - seen by Dr. [REDACTED] ([REDACTED]) who believes likely unilateral NTG OS - Tg mid teens OD; low teens OS - at goal today OU with VF stability 2. Myopia OU - followed previously by optometrist, Dr. [REDACTED] in [REDACTED]- continue current MRx 3. Trace cataract OU - not visually significant - follow Social/systemic: Marginal HTN, PLAN - continue Latanoprost QHS OU - continue Combigan BID OS RTC 10 months IOP check. Input example 2: 64 yro female 1. Glaucoma suspect OU secondary moderate enlarged C/d ratio OU - F/h glaucoma: father - IOP today: 15/15 - C/d ratio OD: 0.60; OS: 0.55; rim healthy OU - CCT today 2023: 581/580 - OCT RNFL in 2018, 2022 and today 2023: WNL OU - HVF 2021 and today 2022: full OU - Pt edu, will monitor yearly with OCT and fundus phote 2. H/o HZV dermatitis, involving right upper and lower lids - Resolved on 8/2020 3. Myopia astigmatism OU - Small change in OU - New Rx=Mrx given to Pt 4. SCLs wearer - Current Acuvue Oasys spherical, Pt feels blurry with CLs; wears glasses more often; interested in toric lens to improve vision - discussed dailies vs biweekly vs monthly toric; Pt prefers biweekly - Good fit with Acuvue Oasys for astigmatism OU; Pt is happy with OR vision OU - Will order trial lens today and deliver to Pt's home - Pt can purchase lens if likes them; CLs Rx given to Pt. - Still has lots supply at home; will RTC next year for toric CLs fitting - CLs hygiene edu Pt. No sleeping, showering or swimming in CLs. Replace every two weeks. Wear less than 12 hours per day. Annual CLs exam. RTC sooner and d/c CLs if experience pain, redness or blurry vision. Output example 2: [REDACTED] yro [REDACTED] 1. Glaucoma suspect OU secondary moderate enlarged C/d ratio OU - F/h glaucoma: father - IOP today: 15/15 - C/d ratio OD: 0.60; OS: 0.55; rim healthy OU - CCT [REDACTED]: 581/580 - OCT RNFL in [REDACTED], [REDACTED] and [REDACTED]: WNL OU - HVF [REDACTED] and [REDACTED]: full OU - Pt edu, will monitor yearly with OCT and fundus phote 2. H/o HZV dermatitis, involving right upper and lower lids - Resolved on [REDACTED] 3. Myopia astigmatism OU - Small change in OU - New Rx=Mrx given to Pt 4. SCLs wearer - Current Acuvue Oasys spherical, Pt feels blurry with CLs; wears glasses more often; interested in toric lens to improve vision - discussed dailies vs biweekly vs monthly toric; Pt prefers biweekly - Good fit with Acuvue Oasys for astigmatism OU; Pt is happy with OR vision OU - Will order trial lens today and deliver to Pt's home - Pt can purchase lens if likes them; CLs Rx given to Pt. - Still has lots supply at home; will RTC next year for toric CLs fitting - CLs hygiene edu Pt. No sleeping, showering or swimming in CLs. Replace every two weeks. Wear less than 12 hours per day. Annual CLs exam. RTC sooner and d/c CLs if experience pain, redness or blurry vision. Medical Note: ''',
        ]
    elif mode == 'summarize':
        prompts = [
                    'Summarize the key details, including the presence of glaucoma, from the clinical note within 180 characters.\nClinical Note:\n',
        ]
    
    for prompt in prompts:
        original_notes = []
        processed_notes = []
        file_names = []
        for i in range(len(dataset)):
            file_name = dataset[i]
            orig_note = np.load(os.path.join(dataset_path, file_name))['note'].item().strip()
            note = prompt + orig_note + '\nSummary:\n'
            print('INPUT NOTE:')
            print(note)
            if model_name in ['med42']:
                input_ids = tokenizer(note, return_tensors='pt').input_ids.cuda()
                output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True,eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=512)
                processed_note = tokenizer.decode(output[0])
                processed_note = processed_note[processed_note.find(note) + len(note):].strip('</s>')
            elif model_name == 'palmyra-med':
                model_inputs = tokenizer(note, return_tensors='pt').to("cuda")
                gen_conf = {"temperature": 0.7, "repetition_penalty": 1.0, "max_new_tokens": 512, "do_sample": True}
                out_tokens = model.generate(**model_inputs, **gen_conf)
                response_ids = out_tokens[0][len(model_inputs.input_ids[0]) :]
                processed_note = tokenizer.decode(response_ids, skip_special_tokens=True)
            elif model_name == 'pmc-llama':
                batch = tokenizer(note, return_tensors="pt", add_special_tokens=False)
                with torch.no_grad():
                    generated = model.generate(inputs = batch["input_ids"].cuda(), max_new_tokens=512, do_sample=True, top_k=50)
                processed_note = tokenizer.decode(generated[0])
                processed_note = processed_note[processed_note.find(note) + len(note):].strip('</s>')
            elif model_name in ['gpt-3.5-turbo', 'gpt-4']:
                completion = openai.ChatCompletion.create(
                model=model_name,
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": orig_note},
                ]
                )
                processed_note = completion.choices[0].message.content

            print('PROCESSED NOTE:')
            print(processed_note)
            original_notes.append(orig_note)
            processed_notes.append(processed_note)
            file_names.append(file_name)

            # write every iteration to avoid losing progress
            output_df = pd.DataFrame({
                'File Name': file_names,
                'Original Note': original_notes, 
                'Summarized Note': processed_notes,
            })
            output_csv = f"{model_name}_{mode}.csv"
            output_df.to_csv(output_csv, index=False)

def merge_all_models_summaries():
    merged_df = pd.DataFrame()

    csv_files = ['flan-t5-base-samsum_summarized_notes.csv', 'bart-large-samsum_summarized_notes.csv', 'bart-large-cnn-samsum_summarized_notes.csv', 'pmc-llama_summarize.csv', 'palmyra-med_summarize.csv', 'med42_summarize.csv', 'gpt-4_summarize.csv']
    model_names = ['Flan-T5', 'BART-LARGE', 'BART-LARGE-CNN', 'PMC-LLAMA (13B)', 'Palmyra-Med (20B)', 'Med42 (70B)', 'GPT-4']

    for csv_file, model_name in zip(csv_files, model_names):
        df = pd.read_csv(csv_file)        
        df.rename(columns={'Summarized Note': f'Summarized Note ({model_name})'}, inplace=True)        
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Original Note', how='outer')
    merged_df.to_csv('all_models_summaries.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str, choices=['med42', 'palmyra-med', 'pmc-llama', 'gpt-3.5-turbo', 'gpt-4'])
    parser.add_argument('--openai_key', type=str)
    args = parser.parse_args()

    openai.api_key = args.openai_key
    dataset_path = "../FUNDUS_Dataset/FairVLMed/data"
    dataset = sorted(os.listdir(dataset_path))
    for model_name in args.models:
        # preprocess(model_name, dataset, 'deidentify', dataset_path)
        preprocess(model_name, dataset, 'summarize', dataset_path)
