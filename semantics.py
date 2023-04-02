import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(f"{word1} - {word2}")
print(word1.similarity(word2))
print(f"{word3} - {word2}")
print(word3.similarity(word2))
print(f"{word3} - {word1}")
print(word3.similarity(word1))

# It is interesting to see that cat and monkey have a higher 
# degree of similarity of 0.59, this makes sense as they are 
# both animals - in addition the banana and monkey have a 
# similarity of 0.40 which can be attributed to the fact 
# that monkeys are associated with eating bananas
# Lastly banana and cat only have a similarity of 0.22 which 
# makes sense as we wouldn't really associate bananas with cats

# An example that would be interesting to test could be 
# "Dog, Cat, Wool"

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
     
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
      similarity = nlp(sentence).similarity(model_sentence)
      print(sentence + " - ", similarity)


# Tested on example.py
# ====== Differences between ‘en_core_web_sm’ 
# and 'en_core_web_md'.======
# The en_core_web_sm has no word vectors loaded therefore the 
# similarity method uses tagger, parser and NER. While these can 
# assess the components and grammar of a sentence is not necessarily 
# useful in assessing the similarity between two sentences 