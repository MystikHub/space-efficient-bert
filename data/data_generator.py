import argparse
import numpy
import random
from bert_tensorflow2 import tokenization

parser = argparse.ArgumentParser(
    prog="Data generator for the Space Efficient Transformer Neural Network Implementation",
    description=
        '''
        Generates a data file with token IDs used by the transformer during
        training and inference for masked language modelling (MLM) and next
        sentence prediction (NSP)
        '''
)
parser.add_argument("-d", "--dataset_path", required=True)
args = parser.parse_args()

tokenizer = tokenization.FullTokenizer("vocab.txt")
vocab = tokenization.load_vocab("vocab.txt")

data_file = open(args.dataset_path, "r")
mlm_file = open("training_mlm.dat", "a")
nsp_file = open("training_nsp.dat", "a")
all_sentences = data_file.readlines()

for i in range(len(all_sentences)):
# for i in range(10):

    sentence = all_sentences[i]

    tokens = tokenizer.tokenize(sentence)

    mlm_token_ids = tokenization.convert_tokens_to_ids(vocab, tokens)
    nsp_token_ids = tokenization.convert_tokens_to_ids(vocab, tokens)
    
    ###
    # MLM task
    ###

    # Mask 15% of the tokens
    mask_indices = numpy.random.choice(len(mlm_token_ids), round(len(mlm_token_ids) * 0.15))
    
    for index in mask_indices:
        mask_chance = random.random()

        # Replace tokens with [MASK] 80% of the time
        if mask_chance < 0.8:
            mlm_token_ids[index] = vocab["[MASK]"]

        # Replace tokens with a random token 10% of the time
        elif mask_chance < 0.9:
            mlm_token_ids[index] = random.randint(1000, len(vocab) - 1)

        # Leave it unchanged 10% of the time
        ##
        # Do nothing

    # Each input to BERT starts with [CLS] and ends with a [SEP]
    mlm_token_ids = [vocab["[CLS]"]] + mlm_token_ids + [vocab["[SEP]"]]
    mlm_string = ""
    for i in range(len(mlm_token_ids)):
        if i == 0:
            mlm_string += str(mlm_token_ids[i])
        else:
            mlm_string += " " + str(mlm_token_ids[i])
    mlm_file.write(mlm_string + "\n")

    ###
    # NSP task
    ###

    if i != len(all_sentences) - 2:
        real_next_sentence_chance = random.random()

        if real_next_sentence_chance < 0.5:
            # Append the tokens of the sentences
            next_sentence_tokens = tokenizer.tokenize(all_sentences[i + 1])
            next_sentence_token_ids = tokenization.convert_tokens_to_ids(vocab, next_sentence_tokens)
            nsp_token_ids = nsp_token_ids + [vocab["[SEP]"]] + next_sentence_token_ids
            nsp_file.write("R: {} AND {} \n".format(sentence, all_sentences[i + 1]))
        else:
            # Append a random sentence
            random_sentence_index = random.randint(0, len(all_sentences) - 1)
            random_sentence_tokens = tokenizer.tokenize(all_sentences[random_sentence_index])
            random_sentence_token_ids = tokenization.convert_tokens_to_ids(vocab, random_sentence_tokens)
            nsp_token_ids = nsp_token_ids + [vocab["[SEP]"]] + random_sentence_token_ids
            nsp_file.write("F: {} AND {} \n".format(sentence, all_sentences[random_sentence_index]))

        # Each input to BERT starts with [CLS] and ends with a [SEP]
        nsp_token_ids = [vocab["[CLS]"]] + nsp_token_ids + [vocab["[SEP]"]]
        nsp_string = ""
        for i in range(len(nsp_token_ids)):
            if i == 0:
                nsp_string += str(nsp_token_ids[i])
            else:
                nsp_string += " " + str(nsp_token_ids[i])
        nsp_file.write(nsp_string + "\n")