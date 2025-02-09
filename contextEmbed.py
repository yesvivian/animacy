

from transformers import BertTokenizerFast, BertModel, GPT2TokenizerFast, GPT2Model
import torch
import numpy as np


def extract_contextualized(file_path, model, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        line = line.strip()
        if line:
            sentence, target_index_str, target_word = line.split('\t')[:3]
            target_index = int(target_index_str)

            words = sentence.split()

            encoded = tokenizer(sentence, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)
           
            input_ids = encoded['input_ids']
            offset_mapping = encoded['offset_mapping']
            offsets = encoded['offset_mapping'][0] 
            
            current_char_pos = 0
            target_word_start = None
            target_word_end = None

            for i, word in enumerate(words):
                if i == target_index:
                    target_word_start = current_char_pos
                    target_word_end = current_char_pos + len(word)
                    break
                current_char_pos += len(word) + 1 

            target_token_indices = []
            for idx, (start, end) in enumerate(offsets):
                if start >= target_word_start-1 and end <= target_word_end: 
                    target_token_indices.append(idx)
            if not target_token_indices:
                raise ValueError(f"Target word '{target_word}' at index {target_index} not found in tokens for sentence: '{sentence}'")


            #########
            ###########

            start_index = target_token_indices[0] 
            end_index = target_token_indices[-1] 

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            target_word_embedding = []
            for layer_index, layer_embedding in enumerate(hidden_states):
                layer_embedding = layer_embedding.squeeze(0)  
                
                target_embeddings = [layer_embedding[i] for i in range(start_index, end_index + 1)]


                if target_embeddings:
                    combined_embedding = torch.mean(torch.stack(target_embeddings), dim=0) 

                    target_word_embedding.append(combined_embedding.numpy()) 

            results.append(target_word_embedding)

            
    return np.array(results) 


def average_embeddings_by_target_word(file_path, embeddings):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    target_word_to_embeddings = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            sentence, target_index_str, target_word = line.split('\t')[:3]
            target_word = target_word.lower()

            if target_word not in target_word_to_embeddings:
                target_word_to_embeddings[target_word] = []

            target_word_to_embeddings[target_word].append(embeddings[i])


    averaged_embeddings = []
    target_words = []

    
    for target_word, word_embeddings in target_word_to_embeddings.items():
        word_embeddings = np.array(word_embeddings)
        avg_word_embedding = np.mean(word_embeddings, axis=0)

        averaged_embeddings.append(avg_word_embedding)
        target_words.append(target_word)

    return target_words,np.array(averaged_embeddings)



def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


def compute_unit_vector(avg_emb1, avg_emb2):
    vector = avg_emb2 - avg_emb1
    unit_vector = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
    return unit_vector


def project_embeddings(embeddings, unit_vectors):

    num_words = embeddings.shape[0]
    num_layers = embeddings.shape[1]
    projected_values = np.zeros((num_words, num_layers))

    for i in range(num_words):
        for j in range(num_layers):
            # Project the j-th layer embedding of the i-th word onto the j-th unit vector
            projected_values[i, j] = np.dot(embeddings[i, j], unit_vectors[j])

    return projected_values




##########################

file_animal = 'wordnet_animal_fin_context.txt'
file_obj = 'wordnet_object_fin_context.txt'
file_person = 'wordnet_person_fin_context.txt'
file_pronoun='pronouns_context.txt'


bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


bert_embed_animal = extract_contextualized(file_animal, bert_model, bert_tokenizer)
bert_embed_obj = extract_contextualized(file_obj, bert_model, bert_tokenizer)
bert_embed_person = extract_contextualized(file_person, bert_model, bert_tokenizer)
bert_embed_pronoun = extract_contextualized(file_pronoun, bert_model, bert_tokenizer)


bert_embed_animal_avgByWd=average_embeddings_by_target_word(file_animal,bert_embed_animal)[1]
bert_embed_obj_avgByWd=average_embeddings_by_target_word(file_obj,bert_embed_obj)[1]
bert_embed_person_avgByWd=average_embeddings_by_target_word(file_person,bert_embed_person)[1]
bert_embed_pronoun_avgByWd=average_embeddings_by_target_word(file_pronoun,bert_embed_pronoun)[1]


avg_bert_embed_animal = average_embeddings(bert_embed_animal_avgByWd)
avg_bert_embed_obj = average_embeddings(bert_embed_obj_avgByWd)
bert_unit_vectors = compute_unit_vector(avg_bert_embed_animal, avg_bert_embed_obj)



bert_projected_person = project_embeddings(bert_embed_person_avgByWd, bert_unit_vectors)
bert_projected_animal = project_embeddings(bert_embed_animal_avgByWd, bert_unit_vectors)
bert_projected_obj = project_embeddings(bert_embed_obj_avgByWd, bert_unit_vectors)
bert_projected_pronoun = project_embeddings(bert_embed_pronoun_avgByWd, bert_unit_vectors)



###########################################################

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
gpt2_model = GPT2Model.from_pretrained('gpt2')

gpt2_embed_animal = extract_contextualized(file_animal, gpt2_model, gpt2_tokenizer)
gpt2_embed_obj = extract_contextualized(file_obj, gpt2_model, gpt2_tokenizer)
gpt2_embed_person = extract_contextualized(file_person, gpt2_model, gpt2_tokenizer)
gpt2_embed_pronoun = extract_contextualized(file_pronoun, gpt2_model, gpt2_tokenizer)

gpt2_embed_animal_avgByWd=average_embeddings_by_target_word(file_animal,gpt2_embed_animal)[1]
gpt2_embed_obj_avgByWd=average_embeddings_by_target_word(file_obj,gpt2_embed_obj)[1]
gpt2_embed_person_avgByWd=average_embeddings_by_target_word(file_person,gpt2_embed_person)[1]
gpt2_embed_pronoun_avgByWd=average_embeddings_by_target_word(file_pronoun,gpt2_embed_pronoun)[1]


avg_gpt2_embed_animal = average_embeddings(gpt2_embed_animal_avgByWd)
avg_gpt2_embed_obj = average_embeddings(gpt2_embed_obj_avgByWd)
gpt2_unit_vectors= compute_unit_vector(avg_gpt2_embed_animal, avg_gpt2_embed_obj)


gpt2_projected_person = project_embeddings(gpt2_embed_person_avgByWd, gpt2_unit_vectors)
gpt2_projected_animal = project_embeddings(gpt2_embed_animal_avgByWd, gpt2_unit_vectors)
gpt2_projected_obj = project_embeddings(gpt2_embed_obj_avgByWd, gpt2_unit_vectors)
gpt2_projected_pronoun = project_embeddings(gpt2_embed_pronoun_avgByWd, gpt2_unit_vectors)



###########################################################



mylist=['person','animal','obj','pronoun']
modlist=['bert','gpt2']
for mod in modlist:
    output_unit_vector=f'{mod}_unitVec.txt'
    unitveclist=f'{mod}_unit_vectors'
    with open(output_unit_vector, 'w') as f:
        #for array in arrays:
        np.savetxt(f, globals()[unitveclist], delimiter='\t', fmt='%f')
        f.write('\n')  

    for x in mylist:
        output = f'{mod}_proj_{x}.txt'
        wordlist=f'words_{x}'
        projlist=f'{mod}_projected_{x}'

        with open(output, 'w') as file:
            for word, projection in zip(globals()[wordlist], globals()[projlist]):
                projection_str = '\t'.join(map(str, projection))
                file.write(f"{word}\t{projection_str}\n")

        print(f"Projection results saved to {output}")



