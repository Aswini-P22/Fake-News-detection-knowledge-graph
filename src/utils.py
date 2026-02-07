import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_relations(text):
    doc = nlp(text)
    relations = []

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl"):
            subject = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w.text for w in token.rights if w.dep_ in ("dobj", "pobj")]
            if subject and obj:
                relations.append((subject[0], token.lemma_, obj[0]))

    return relations