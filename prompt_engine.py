import re
import uuid
import chromadb


class PromptEngine:
    """Combines vector memory (ChromaDB) and graph rules for scene suggestions, summaries, and canon checks."""

    def __init__(self, collection_name: str = "story_memory"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_story_chunk(self, text: str, chapter_num: int) -> None:
        """Stores a published story paragraph tagged with its chapter number."""
        if not text.strip():
            return
        self.collection.add(
            documents=[text],
            metadatas=[{"chapter": chapter_num}],
            ids=[str(uuid.uuid4())],
        )

    def retrieve_similar_context(self, query: str, top_k: int = 2) -> list[str]:
        """Retrieves semantically relevant past passages across all chapters."""
        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        results = self.collection.query(
            query_texts=[query or "recent story events"],
            n_results=min(top_k, total_docs),
        )
        documents = results.get("documents", [[]])
        return documents[0] if documents else []

    def summarize_chapter(
        self,
        chapter_title: str,
        passages: list[dict],
        chapter_facts: list[str],
        generate_fn=None,
        model_id: str = "",
    ) -> str:
        """Generates a concise summary of a chapter from its passages and extracted relations."""
        if not passages:
            return "No passages written in this chapter yet."

        full_text = "\n\n".join(p.get("prose", "").strip() for p in passages if p.get("prose", "").strip())

        if model_id.strip() and generate_fn is not None:
            try:
                summary_prompt = (
                    f"Summarize '{chapter_title}' in 2 to 3 concise sentences capturing the key plot events "
                    f"and character actions:\n\n{full_text[:1800]}\n\nSummary:"
                )
                ai_sum = generate_fn(summary_prompt, model_id.strip(), max_tokens=110)
                if ai_sum.strip():
                    return ai_sum.strip()
            except Exception:
                pass

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]
        if len(sentences) <= 2:
            narrative_digest = " ".join(sentences)
        else:
            narrative_digest = f"{sentences[0]} ... {sentences[-1]}"

        if chapter_facts:
            clean_facts = "; ".join(f.replace("--[", "(").replace("]-->", ")") for f in chapter_facts[:4])
            return f"{narrative_digest}\n\nKey chapter developments: {clean_facts}."
        return narrative_digest

    def check_canon_conflicts(self, text: str, character_states: dict) -> list[str]:
        """Checks across all chapters if a passage uses an item marked as LOST."""
        warnings = []
        lower_text = text.lower()
        for char_name, info in character_states.items():
            if char_name.lower() in lower_text:
                for lost_item in info.get("lost_items", []):
                    if lost_item.lower() in lower_text:
                        warnings.append(
                            f"Canon check: '{char_name}' previously lost '{lost_item}', which is referenced in this passage."
                        )
        return warnings

    def build_scene_suggestion_prompt(
        self,
        active_facts: list[str],
        context_chunks: list[str],
        writer_note: str = "",
    ) -> str:
        """Builds a prompt asking the AI to suggest 3 concise, canon-consistent next-scene ideas."""
        facts_block = (
            "\n".join(f"- {fact}" for fact in active_facts)
            if active_facts
            else "- Blank slate: introduce a compelling protagonist, setting, and central mystery."
        )
        context_block = (
            "\n\n".join(f'"{chunk}"' for chunk in context_chunks)
            if context_chunks
            else "Opening scene (no prior passages yet)."
        )
        focus_line = f"\nWriter's direction/theme hint: {writer_note}" if writer_note.strip() else ""

        return (
            "You are a creative story consultant assisting a human author. "
            "Based on the current world state and recent story events, suggest exactly 3 distinct, concise next-scene ideas "
            "(2 sentences each). Respect all character locations and inventories across all chapters.\n\n"
            f"[CUMULATIVE WORLD STATE]\n{facts_block}\n\n"
            f"[RECENT STORY CONTEXT]\n{context_block}{focus_line}\n\n"
            "Format your response strictly as:\n"
            "1. [Scene Title]: <2-sentence scene suggestion>\n"
            "2. [Scene Title]: <2-sentence scene suggestion>\n"
            "3. [Scene Title]: <2-sentence scene suggestion>\n"
        )

    def parse_scene_suggestions(self, raw_output: str) -> list[dict]:
        """Parses numbered LLM output into clean scene suggestion cards."""
        suggestions = []
        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        pattern = re.compile(r"^(?:Option\s*)?[1-3][\.\)\:\-]\s*(?:\[([^\]]+)\]|([^:]+)):?\s*(.+)$", re.IGNORECASE)

        for line in lines:
            match = pattern.match(line)
            if match:
                title = (match.group(1) or match.group(2) or "Scene Idea").strip(" []*")
                body = match.group(3).strip()
                suggestions.append({"title": title, "beat": body})

        if not suggestions and raw_output.strip():
            chunks = [c.strip() for c in raw_output.strip().split("\n\n") if c.strip()][:3]
            for idx, chunk in enumerate(chunks, start=1):
                suggestions.append({"title": f"Scene Direction 0{idx}", "beat": chunk})

        return suggestions[:3]

    def clear(self) -> None:
        """Deletes and recreates the vector store collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)