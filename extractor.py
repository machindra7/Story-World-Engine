import spacy


class NLPTripleExtractor:
    """Extracts structured (Subject, Predicate, Object) triples from narrative prose."""

    POSSESS_VERBS = {
        "take", "grab", "pick", "find", "obtain", "hold", "carry",
        "equip", "receive", "get", "keep", "have", "acquire", "discover"
    }
    LOSE_VERBS = {
        "drop", "lose", "discard", "break", "destroy", "leave",
        "give", "trade", "abandon", "throw"
    }
    LOCATION_VERBS = {
        "enter", "reach", "arrive", "walk", "travel", "go",
        "stand", "stay", "visit", "step", "run"
    }
    INTERACT_VERBS = {
        "meet", "fight", "attack", "speak", "talk", "help",
        "follow", "confront", "greet"
    }
    PRONOUNS = {"he", "she", "they", "him", "her"}

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. Run: python -m spacy download {model_name}"
            ) from exc

    def extract_entities(self, text: str) -> list[str]:
        """Extracts named entities and capitalized nouns from input text."""
        doc = self.nlp(text)
        entities = [ent.text.strip().title() for ent in doc.ents]
        for token in doc:
            if token.pos_ == "PROPN" and len(token.text) > 1:
                entities.append(token.text.strip().title())
        return list(dict.fromkeys(entities))

    def _clean_phrase(self, token) -> str:
        """Expands a noun token into its clean compound/adjective phrase."""
        subtree_tokens = [
            t.text for t in token.subtree
            if t.dep_ in ("compound", "amod", "det") and t.pos_ != "DET"
        ]
        subtree_tokens.append(token.text)
        return " ".join(subtree_tokens).strip().title()

    def extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """Parses sentences into state-tracking (Subject, Predicate, Object) triples."""
        doc = self.nlp(text)
        triples = []
        last_active_character = None

        for sent in doc.sents:
            # Track named people for pronoun resolution across sentences
            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    last_active_character = ent.text.strip().title()

            for token in sent:
                if token.pos_ != "VERB" and token.lemma_ != "be":
                    continue

                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                if not subjects:
                    continue

                subj_token = subjects[0]
                if subj_token.text.lower() in self.PRONOUNS:
                    if not last_active_character:
                        continue
                    subject_str = last_active_character
                elif subj_token.pos_ in ("PROPN", "NOUN"):
                    subject_str = self._clean_phrase(subj_token)
                    if subj_token.pos_ == "PROPN":
                        last_active_character = subject_str
                else:
                    continue

                lemma = token.lemma_.lower()

                # Direct objects
                direct_objs = [child for child in token.children if child.dep_ in ("dobj", "attr")]
                for obj_token in direct_objs:
                    if obj_token.pos_ not in ("NOUN", "PROPN"):
                        continue
                    obj_str = self._clean_phrase(obj_token)

                    if lemma == "be":
                        triples.append((subject_str, "IS_A", obj_str))
                    elif lemma in self.POSSESS_VERBS:
                        triples.append((subject_str, "POSSESSES", obj_str))
                    elif lemma in self.LOSE_VERBS:
                        triples.append((subject_str, "LOST", obj_str))
                    elif lemma in self.LOCATION_VERBS:
                        triples.append((subject_str, "LOCATION", obj_str))
                    elif lemma in self.INTERACT_VERBS:
                        triples.append((subject_str, "MET", obj_str))

                # Prepositional objects (e.g., "walked into the Ruined Tower")
                for prep in [child for child in token.children if child.dep_ == "prep"]:
                    for pobj in [child for child in prep.children if child.dep_ == "pobj"]:
                        if pobj.pos_ not in ("NOUN", "PROPN"):
                            continue
                        obj_str = self._clean_phrase(pobj)
                        if lemma in self.LOCATION_VERBS and prep.text.lower() in ("in", "into", "at", "to", "inside"):
                            triples.append((subject_str, "LOCATION", obj_str))
                        elif lemma in self.INTERACT_VERBS and prep.text.lower() in ("with", "to"):
                            triples.append((subject_str, "MET", obj_str))

        return list(dict.fromkeys(triples))