import fasttext
import numpy as np
import csv

if __name__ == '__main__':
    model_path = "/workspace/datasets/fasttext/title_model.bin"
    main_word_file_path = "/workspace/datasets/fasttext/top_words.txt"
    output_file_path = "/workspace/datasets/fasttext/synonyms.csv"
    threshold = 0.70

    model = fasttext.load_model(model_path)
    all_synonyms = []

    with open(main_word_file_path, 'r' ) as f:
        for word in f:
            neighbors = model.get_nearest_neighbors(word, k = 100)
            synonyms = [word.strip()]
            for neighbor in neighbors:
                if neighbor[0] >= threshold:
                    synonyms.append(neighbor[1])
            if len(synonyms) > 1:
                all_synonyms.append(synonyms)

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for word_synonyms in all_synonyms:
            writer.writerow(word_synonyms)
    
        