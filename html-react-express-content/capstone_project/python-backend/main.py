import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spacy

app = FastAPI()

# allow React to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("ja_ginza")

# mapping of pos meanings
POS_NOTES = {
    "NOUN": "thing / object",
    "VERB": "action / state change",
    "ADJ": "describes nouns (quality)",
    "ADV": "modifies action",
    "AUX": "helper verb",
    "PART": "grammar particle",
    "PUNCT": "punctuation",
    "ADP": "connects nouns to rest of sentence",
    "SCONJ": "connects clauses"
}

# other option for notes:
# POS_NOTES = {
#     "NOUN": "thing / object (can be subject or object)",
#     "VERB": "action / state change",
#     "ADJ": "describes nouns (quality)",
#     "PART": "connects grammar (e.g. は, が, を)",
#     "ADP": "particle (connects words / marks grammar role)",
#     "SCONJ": "clause connector (because / if / although)"
# }

# helper function
def get_note(pos):
    return POS_NOTES.get(pos, "")

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/pos")
def pos(data: dict):
    doc = nlp(data["text"])

    return {
        "tokens": [
            {
                "word": token.text,
                "reading": token.lemma_,
                "pos": token.pos_,
                "notes":get_note(token.pos_)
            }
            for token in doc
        ]
    }
