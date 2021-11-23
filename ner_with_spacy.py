import spacy
from spacy import displacy

user_utterance = "Steve, do you know weather Michael Jordan who worked for the NBA was born in LA (USA) or not?"


nlp = spacy.load("en_core_web_md")

doc = nlp(user_utterance)

print(spacy.explain("ORG"))
print(spacy.explain("GPE"))  # Geopolitical Entities

print(doc.ents)
print(doc.ents[0].label_, type(doc.ents[0].label_))
print(doc.ents[0].text, type(doc.ents[0].text))

print("\nExtracting entities from the text")
for entity in doc.ents:
    print(f"{entity.label_}: {entity.text}")
