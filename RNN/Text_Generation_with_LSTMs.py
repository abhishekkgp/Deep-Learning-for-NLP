#!/usr/bin/env python
# coding: utf-8

# ## text generation with LSTMswith keras

# In[1]:


def read_file(filepath):
    with open(filepath) as f:
        str_text=f.read()
        
    return str_text   



# tokenize and clean the text using spacey
import spacy
nlp=spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner']) # load en i.e., english ; alse disbling some word which do not need recogination like parsing,tagger etc
#spacy.load('')



nlp.max_length= 1198623


''' defining a function and it takes some document text as a string 
and then it's going grab the text tokens. if they are not some type of punctuation
or if they are not any line. '''
def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_{|}~\t\n ']

d= read_file('E:/study material/Deep learning/NLP/moby_dick_four_chapters.txt')


tokens= separate_punc(d) #calling the function

print(len(tokens))

# we are passinh here 25 word and let the NN predict 26 word
train_len=25+1
text_sequences=[]

for i in range(train_len, len(tokens)):
    seq=tokens[i-train_len:i] # it contains sequence of 26 value in it 
                              #so that for given 25 value it can predics 26th
    
    text_sequences.append(seq)


print(type(text_sequences))

print(text_sequences[0]) # it will print 0 to 25

print(text_sequences[1]) # it will print 1 to 26

# to see entire in 1 sequence together
print(' '.join(text_sequences[0]))


print(' '.join(text_sequences[1]))


print(' '.join(text_sequences[2]))


# koros use to tokenization into numerical system
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer() # let it has it's value
tokenizer.fit_on_texts(text_sequences) #  coverting text into numbers

sequences= tokenizer.texts_to_sequences(text_sequences)

print(sequences[0]) # we can say that each of the numver is ID of particular word


print(sequences[1]) # we can say that each of the numver is ID of particular word

# to see relationship between ID and word
#tokenizer.index_word--> this will give result for all words

# for one sequences
for i in sequences[0]:
    print(f"{i}:{tokenizer.index_word[i]}") # for same word will have same numberical ID 

#tokenizer.word_counts--> given no of time a word came


# to get size of vocabulary
vocabulary_size=len(tokenizer.word_counts)
print(vocabulary_size)


type(sequences)

#to convert sequences into numpy
import numpy as np
sequences=np.array(sequences)
print(sequences) 


'''we are going to do train-test split on sequence.
assuming last column as target and rest of the column as features'''
from keras.utils import to_categorical
X=sequences[:,:-1]
y=sequences[:,-1]
y=to_categorical(y,num_classes=vocabulary_size+1)
seq_len=X.shape[1] # storing number of features i.e., 25
#seq_len

# Now making tha model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


def create_model(vocabulary_size,seq_len):
    model=Sequential()
    model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len)) # taking input dim=vocabulary_size and output dimension=seq_len
    model.add(LSTM(50,return_sequences=True)) # randomy choosing number of neuron inside LSTM equal to 50. we can also take 150 value
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    
    model.add(Dense(vocabulary_size,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    model.summary()
    
    return model
                                                                             


model=create_model(vocabulary_size+1,seq_len) # we added 1 in vocabulary_size for cosidering 0


from pickle import dump,load
model.fit(X,y,batch_size=512,epochs=500,verbose=1)
#batch size is how  many sequences we want to passin so that NN can handle and that chosen arbitarly
#  verbose give information about accuracy
# our accuracy may be very low as we have taken value of epocha very low.    


# to save this model
model.save('E:/study material/Deep learning/NLP/LSTM.h5')


#to save tokenizer
dump(tokenizer,open('E:/study material/Deep learning/NLP/simpletokeniser','wb'))


# ## To generate new text using seed text

from keras.preprocessing.sequence import pad_sequences

# seed text means some text to just start off
def generate_text(model,tokenizer, seq_len, seed_text, num_gen_words): 
    #num_gen_word is number of word we want to generate after the seed_text
    # tokenzer has idea about what ID number goes to which word
    output_text=[]
    
    input_text=seed_text
    for i in range(num_gen_words):
        encoded_text=tokenizer.texts_to_sequences([input_text])[0] # [0] is use to grab first item here
        # transformed the the text data into sequences of number
        
        pad_encoded=pad_sequences([encoded_text],maxlen=seq_len,truncating='pre') 
        '''
        ->if seed_text is too long or too short then we can pad it up to 25. 
        ->for bext result seed_text lenghth should be equal to seq_len.
        ->for truncating we can either go begining 
        or after the string here it is at begning i.e, "pre"  '''

        # pred_word_ind--> predicted word index
        pred_word_ind= model.predict_classes(pad_encoded, verbose=0) [0]
        # this predicted class probability for each word
        
        predicted_word=tokenizer.index_word[pred_word_ind]
        
        input_text+=' '+predicted_word
        
        output_text.append(predicted_word)
        
        
    return ' '.join(output_text)   

# now to indiatate seed text, and we can do it either directly indiating 
# or choosing randomly from text_sequences we found earlier.
print('text_sequences[0]:-',text_sequences[0])

# to chood seed text randomly-
import random
random.seed(101)
random_pick= random.randint(0,len(text_sequences))
random_seed_text=text_sequences[random_pick]
print('random_seed_text:-',random_seed_text)

seed_text=' '.join(random_seed_text) # here seed_text lenghth is 25 so it will give better result
print('seed_text:-',seed_text)

generated_text=generate_text(model,tokenizer,seq_len,seed_text=seed_text, num_gen_words=10) # this will give result as we only did training on very less epochs
print('generated_text:-',generated_text)

print('' '.join(text_sequences[0]):-',' '.join(text_sequences[0]))

print('' '.join(text_sequences[15]):-',' '.join(text_sequences[15]))


seed_text=' '.join(text_sequences[0]) #second example

print('seed_text2:-',seed_text)


generated_text=generate_text(model,tokenizer,seq_len,seed_text=seed_text, num_gen_words=10) # this will give result as we only did training on very less epochs

print('generated_text2:-',generated_text)

seed_text=' '.join(text_sequences[3]) # Third example

print('seed_text3:-',seed_text)

generated_text=generate_text(model,tokenizer,seq_len,seed_text=seed_text, num_gen_words=10) # this will give result as we only did training on very less epochs
print('generated_text3:-',generated_text)






