import html
import os
import re
import torch
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoModelForCausalLM, AutoTokenizer

from exporter import generate_pdf_bytes, generate_word_bytes
from extractor import NLPTripleExtractor
from graph_manager import KnowledgeGraphManager
from prompt_engine import PromptEngine


st.set_page_config(
    page_title="Story World Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_extractor():
    return NLPTripleExtractor()


@st.cache_resource
def load_language_model(model_id: str):
    """Loads any user-specified Hugging Face causal language model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return tokenizer, model


def generate_text(prompt: str, model_id: str, max_tokens: int = 220) -> str:
    """Runs inference using the configured model ID."""
    tokenizer, model = load_language_model(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def build_dry_run_suggestions(character_states: dict, batch_index: int = 0) -> list[dict]:
    """Generates varied 2-sentence canon-aware scene ideas across refresh clicks when no model ID is set."""
    if not character_states:
        blank_sets = [
            [
                {
                    "title": "The Inciting Discovery",
                    "beat": "Introduce your protagonist arriving at an atmospheric landmark and uncovering an unusual object that sets the plot in motion.",
                },
                {
                    "title": "An Unexpected Ally",
                    "beat": "Have a second character cross paths with your protagonist, bringing conflicting motives and a warning about what lies ahead.",
                },
                {
                    "title": "A Locked Threshold",
                    "beat": "Place a physical or moral obstacle in front of the protagonist that forces them to use what they currently carry.",
                },
            ],
            [
                {
                    "title": "The Interrupted Ritual",
                    "beat": "Open mid-action as your protagonist witnesses a secretive exchange in the rain and accidentally draws attention to themselves.",
                },
                {
                    "title": "A Coded Message",
                    "beat": "Have a courier or stranger slip an unsigned note into your protagonist's coat, naming a meeting place before midnight.",
                },
                {
                    "title": "The Unreliable Map",
                    "beat": "Show your protagonist realizing a landmark they trusted has been deliberately altered, forcing an immediate detour.",
                },
            ],
            [
                {
                    "title": "Echoes of an Old Debt",
                    "beat": "Introduce an acquaintance from the protagonist's past who demands a favor in exchange for safe passage.",
                },
                {
                    "title": "The Vanishing Witness",
                    "beat": "Have your protagonist arrive just moments after a key informant disappears, leaving behind a single telling clue.",
                },
                {
                    "title": "Storm on the Horizon",
                    "beat": "Use an approaching weather front or curfew bell to trap the protagonist in unfamiliar territory with a rival.",
                },
            ],
        ]
        return blank_sets[batch_index % len(blank_sets)]

    primary_char = next(iter(character_states))
    info = character_states[primary_char]
    loc = info["location"] if info["location"] != "Unknown" else "the current setting"
    item = info["inventory"][0] if info["inventory"] else "their wits"

    canon_sets = [
        [
            {
                "title": "Canon Clue Discovery",
                "beat": f"Inside {loc}, {primary_char} notices a hidden detail that reacts directly to {item}, revealing a deeper secret.",
            },
            {
                "title": "Rising Confrontation",
                "beat": f"Someone challenges {primary_char}'s presence at {loc}, forcing a tense exchange before {primary_char} can move forward.",
            },
            {
                "title": "Shift in Setting",
                "beat": f"A sudden disturbance drives {primary_char} out of {loc} toward an unexplored location with new stakes.",
            },
        ],
        [
            {
                "title": "The Hidden Compartment",
                "beat": f"While searching {loc}, {primary_char} uncovers a concealed recess that can only be opened or understood using {item}.",
            },
            {
                "title": "An Unspoken Truce",
                "beat": f"A reluctant rival corners {primary_char} in {loc} and proposes a temporary alliance against a shared threat.",
            },
            {
                "title": "Signs of Pursuit",
                "beat": f"Fresh tracks and a discarded token at {loc} prove that someone has been shadowing {primary_char} all along.",
            },
        ],
        [
            {
                "title": "A Costly Bargain",
                "beat": f"An informant in {loc} offers vital directions, but demands {item}—or a dangerous promise—in return from {primary_char}.",
            },
            {
                "title": "The Quiet Interrogation",
                "beat": f"{primary_char} questions a guarded bystander at {loc}, catching a contradiction that exposes a larger deception.",
            },
            {
                "title": "Closing Exits",
                "beat": f"The main passage out of {loc} is sealed off, forcing {primary_char} to risk an unstable upper or subterranean route.",
            },
        ],
    ]
    return canon_sets[batch_index % len(canon_sets)]


def _slugify_filename(title: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (title or "story").strip()).strip("_").lower()
    return cleaned or "story_manuscript"


def _resize_pyvis_html(raw_html: str, height_px: int) -> str:
    """Adjusts PyVis network container height for split-pane or widescreen rendering."""
    updated = re.sub(r"height:\s*\d+px", f"height: {height_px}px", raw_html)
    override_css = f"<style>#mynetwork {{ height: {height_px}px !important; width: 100% !important; }}</style>"
    if "</head>" in updated:
        return updated.replace("</head>", f"{override_css}\n</head>", 1)
    return override_css + updated


def _refresh_chapter_summary(chap_num: int) -> None:
    """Updates the stored summary for a chapter from its passages and chapter-wise facts."""
    chap = st.session_state.chapters[chap_num]
    chap_facts = st.session_state.graph_manager.get_chapter_facts(chap_num)
    chap["summary"] = st.session_state.prompt_engine.summarize_chapter(
        chapter_title=chap["title"],
        passages=chap["passages"],
        chapter_facts=chap_facts,
        generate_fn=generate_text,
        model_id=st.session_state.model_id,
    )


def _record_passage_in_active_chapter(
    prose_text: str,
    extractor: NLPTripleExtractor,
) -> None:
    """Appends the writer's prose strictly to the active chapter and updates chapter + global graphs."""
    active_chap = st.session_state.active_chapter
    chap_obj = st.session_state.chapters[active_chap]

    prior_states = st.session_state.graph_manager.get_character_states()
    warnings = st.session_state.prompt_engine.check_canon_conflicts(prose_text, prior_states)

    extracted_triples = extractor.extract_triples(prose_text)
    for subj, pred, obj in extracted_triples:
        st.session_state.graph_manager.add_fact(subj, pred, obj, chapter_num=active_chap)
        if pred == "LOCATION" and st.session_state.story_title == "Untitled Chronicle":
            st.session_state.story_title = f"The {obj}"

    st.session_state.prompt_engine.add_story_chunk(prose_text, active_chap)
    chap_obj["passages"].append(
        {
            "author_type": "AUTHOR",
            "prose": prose_text,
            "warnings": warnings,
        }
    )
    _refresh_chapter_summary(active_chap)


def _render_knowledge_graph_panel(active_chap_num: int, is_widescreen: bool = False) -> None:
    """Renders the Knowledge Graph in either the right-hand split column or full widescreen mode."""
    graph_height = 640 if is_widescreen else 375

    if st.session_state.story_completed:
        view_mode = "story_complete"
        selected_graph_chap = active_chap_num
        if is_widescreen:
            back_col, info_col = st.columns([1.4, 5.6])
            with back_col:
                if st.button("← Back to Writing Studio", key="exit_wide_complete", use_container_width=True):
                    st.session_state.graph_widescreen = False
                    st.rerun()
            with info_col:
                st.markdown(
                    '<div style="padding-top:6px;font-size:12.5px;color:#9ba39b;">Widescreen Inter-Chapter Knowledge Graph — showing connected characters and relics across all completed chapters.</div>',
                    unsafe_allow_html=True,
                )
        else:
            if st.button("⤢ Open Widescreen Graph", key="enter_wide_complete", use_container_width=True):
                st.session_state.graph_widescreen = True
                st.rerun()

        display_graph = st.session_state.graph_manager.build_inter_chapter_graph()
        fact_count = display_graph.number_of_edges()
        fact_label = "entity links"
    else:
        chap_options = [
            f"Chapter {n:02d}: {st.session_state.chapters[n]['title']}"
            for n in sorted(st.session_state.chapters.keys())
        ]
        scope_options = [f"Active ({chap_options[-1]})"] + chap_options[:-1] + ["Inter-Chapter Map"]

        if (
            "graph_scope_select" not in st.session_state
            or st.session_state.graph_scope_select not in scope_options
        ):
            st.session_state.graph_scope_select = scope_options[0]

        if is_widescreen:
            back_col, scope_col, spacer_col = st.columns([1.65, 2.4, 3.0])
            with back_col:
                if st.button("← Back to Writing Studio", key="exit_wide_btn", use_container_width=True):
                    st.session_state.graph_widescreen = False
                    st.rerun()
            with scope_col:
                selected_scope = st.selectbox(
                    "Knowledge Graph Scope",
                    options=scope_options,
                    key="graph_scope_select",
                    label_visibility="collapsed",
                )
        else:
            scope_col, wide_col = st.columns([3.0, 1.45])
            with scope_col:
                selected_scope = st.selectbox(
                    "Knowledge Graph Scope",
                    options=scope_options,
                    key="graph_scope_select",
                    label_visibility="collapsed",
                )
            with wide_col:
                if st.button("⤢ Widescreen", key="enter_wide_btn", use_container_width=True):
                    st.session_state.graph_widescreen = True
                    st.rerun()

        if selected_scope == "Inter-Chapter Map":
            view_mode = "story_complete"
            selected_graph_chap = active_chap_num
        elif selected_scope.startswith("Active ("):
            view_mode = "chapter"
            selected_graph_chap = active_chap_num
        else:
            view_mode = "chapter"
            selected_graph_chap = int(selected_scope.split(":")[0].replace("Chapter", "").strip())

        display_graph = st.session_state.graph_manager.get_display_graph(
            view_mode=view_mode,
            chapter_num=selected_graph_chap,
        )
        fact_count = display_graph.number_of_edges()
        fact_label = "entity links" if view_mode == "story_complete" else f"ch {selected_graph_chap:02d} facts"

    eyebrow_label = (
        "02 &nbsp;/&nbsp; KNOWLEDGE GRAPH &nbsp;•&nbsp; WIDESCREEN VIEW"
        if is_widescreen
        else "02 &nbsp;/&nbsp; KNOWLEDGE GRAPH"
    )
    st.markdown(
        f"""
        <div class="section-header">
            <div>
                <div class="section-eyebrow">{eyebrow_label}</div>
                <div class="section-title">{"Connected cast & relics" if view_mode == "story_complete" else f"Chapter {selected_graph_chap:02d} state"}</div>
            </div>
            <div class="facts-counter">
                <div class="facts-num">{fact_count}</div>
                <div class="facts-label">{fact_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if display_graph.number_of_nodes() > 0:
        st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
        raw_graph_html = st.session_state.graph_manager.to_pyvis_html(
            view_mode=view_mode,
            chapter_num=selected_graph_chap,
        )
        sized_graph_html = _resize_pyvis_html(raw_graph_html, graph_height)
        components.html(sized_graph_html, height=graph_height, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_msg = (
            "Characters and items that appear across multiple chapters will connect your chapters here."
            if view_mode == "story_complete"
            else f"Chapter {selected_graph_chap:02d} graph is fresh. Publish a passage to map this chapter's characters and objects."
        )
        st.markdown(
            f"""
            <div class="graph-frame" style="height: {graph_height}px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #6c756c; text-align: center; padding: 2rem; background-image: linear-gradient(to right, rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.035) 1px, transparent 1px); background-size: 90px 90px;">
                <div style="font-family: 'Instrument Serif', serif; font-size: 22px; color: #9ba39b; margin-bottom: 6px;">Fresh chapter canvas</div>
                <div style="font-size: 13px; max-width: 340px;">{empty_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Legend Bar
    if view_mode == "story_complete":
        st.markdown(
            """
            <div class="legend-bar">
                <span class="legend-item"><span class="legend-dot" style="background:#c8a86b;"></span>Story Chapter</span>
                <span class="legend-item"><span class="legend-dot" style="background:#7aa2f7;"></span>Connected Character</span>
                <span class="legend-item"><span class="legend-dot" style="background:#68d391;"></span>Connected Object / Item</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="legend-bar">
                <span class="legend-item"><span class="legend-dot" style="background:#7aa2f7;"></span>Character</span>
                <span class="legend-item"><span class="legend-dot" style="background:#68d391;"></span>Object / Relic</span>
                <span class="legend-item"><span class="legend-dot" style="background:#f08070;"></span>Location</span>
                <span class="legend-item"><span class="legend-dot" style="background:#e2b059;"></span>Trait</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Bottom Details Panel (2 columns in Widescreen, stacked in Split view)
    if view_mode == "story_complete":
        inter_links = st.session_state.graph_manager.get_inter_chapter_relations_list()
        st.markdown(
            f"""
            <div class="inventory-header">
                <span class="inventory-title">Connected characters & items</span>
                <span class="inventory-count">{len(inter_links)} bridges</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if inter_links:
            if is_widescreen and len(inter_links) > 1:
                mid = (len(inter_links) + 1) // 2
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    for link in inter_links[:mid]:
                        st.markdown(f'<div class="fact-pill">{html.escape(link)}</div>', unsafe_allow_html=True)
                with c2:
                    for link in inter_links[mid:]:
                        st.markdown(f'<div class="fact-pill">{html.escape(link)}</div>', unsafe_allow_html=True)
            else:
                for link in inter_links:
                    st.markdown(f'<div class="fact-pill">{html.escape(link)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="padding: 8px 0; font-size: 12.5px; color: #6c756c;">No recurring characters or items bridge multiple chapters yet.</div>',
                unsafe_allow_html=True,
            )
    else:
        chap_facts = st.session_state.graph_manager.get_chapter_facts(selected_graph_chap)
        states = st.session_state.graph_manager.get_character_states()

        def _render_relations():
            st.markdown(
                f"""
                <div class="inventory-header">
                    <span class="inventory-title">Chapter {selected_graph_chap:02d} relations</span>
                    <span class="inventory-count">{len(chap_facts)} triples</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if chap_facts:
                for fact in chap_facts:
                    st.markdown(f'<div class="fact-pill">{html.escape(fact)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="padding: 8px 0; font-size: 12.5px; color: #6c756c;">No relations extracted in Chapter {selected_graph_chap:02d} yet.</div>',
                    unsafe_allow_html=True,
                )

        def _render_inventory():
            border_css = "" if is_widescreen else "margin-top: 6px; border-top: 1px solid #1f241f;"
            st.markdown(
                f"""
                <div class="inventory-header" style="{border_css}">
                    <span class="inventory-title">Cumulative world inventory</span>
                    <span class="inventory-count">{len(states)} characters</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if states:
                rows_html = []
                for char_name, info in states.items():
                    avatar_uri = KnowledgeGraphManager._build_avatar_svg(char_name, "Character")
                    loc_label = (
                        f"At {html.escape(info['location'])}"
                        if info["location"] != "Unknown"
                        else "Location unknown"
                    )
                    if info["inventory"]:
                        items_str = ", ".join(html.escape(item) for item in info["inventory"])
                        right_badge = f'<span class="char-items">🔗 {items_str}</span>'
                    else:
                        right_badge = '<span class="char-empty">Empty</span>'

                    rows_html.append(
                        f"""
                        <div class="char-row">
                            <div class="char-left">
                                <img class="char-avatar-img" src="{avatar_uri}" alt="{html.escape(char_name)}" />
                                <div>
                                    <div class="char-name">{html.escape(char_name)}</div>
                                    <div class="char-loc">{loc_label}</div>
                                </div>
                            </div>
                            <div>{right_badge}</div>
                        </div>
                        """
                    )
                st.markdown("".join(rows_html), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="padding: 8px 0; font-size: 12.5px; color: #6c756c;">No characters tracked yet.</div>',
                    unsafe_allow_html=True,
                )

        if is_widescreen:
            rel_col, inv_col = st.columns(2, gap="large")
            with rel_col:
                _render_relations()
            with inv_col:
                _render_inventory()
        else:
            _render_relations()
            _render_inventory()

    st.markdown(
        """
        <div class="graph-footer">
            <span><span class="dot-green" style="margin-right:6px;"></span>Cross-chapter memory active</span>
            <span class="font-mono" style="letter-spacing:0.1em;">CHAPTER-SCOPED GRAPH</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Initialize Session State
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = KnowledgeGraphManager()
if "prompt_engine" not in st.session_state:
    st.session_state.prompt_engine = PromptEngine()
if "chapters" not in st.session_state:
    st.session_state.chapters = {
        1: {
            "number": 1,
            "title": "The Beginning",
            "status": "active",
            "summary": "",
            "passages": [],
        }
    }
if "active_chapter" not in st.session_state:
    st.session_state.active_chapter = 1
if "reading_chapter" not in st.session_state:
    st.session_state.reading_chapter = None
if "graph_widescreen" not in st.session_state:
    st.session_state.graph_widescreen = False
if "story_completed" not in st.session_state:
    st.session_state.story_completed = False
if "scene_suggestions" not in st.session_state:
    st.session_state.scene_suggestions = []
if "suggestion_history" not in st.session_state:
    st.session_state.suggestion_history = []
if "model_id" not in st.session_state:
    st.session_state.model_id = os.getenv("STORY_MODEL_ID", "")
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 220
if "story_title" not in st.session_state:
    st.session_state.story_title = "Untitled Chronicle"
if "story_logline" not in st.session_state:
    st.session_state.story_logline = "Multi-chapter fiction written by you, linked in canon."
if "writer_editor_area" not in st.session_state:
    st.session_state.writer_editor_area = ""
if "clear_editor" not in st.session_state:
    st.session_state.clear_editor = False

# Clear editor BEFORE widget instantiation when requested
if st.session_state.clear_editor:
    st.session_state.writer_editor_area = ""
    st.session_state.clear_editor = False

extractor = load_extractor()

# Inject Custom Editorial Dark-Olive CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #121512;
        --bg-surface: #171b17;
        --bg-card: #191f19;
        --border-subtle: #262c26;
        --border-green: #263326;
        --text-primary: #f0ede6;
        --text-secondary: #9ba39b;
        --text-muted: #6c756c;
        --accent-gold: #c8a86b;
        --accent-gold-btn: #8f7445;
        --accent-green: #68b684;
        --accent-blue: #7aa2f7;
        --accent-coral: #e07a6b;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
        display: none !important;
    }

    .block-container {
        max-width: 1420px !important;
        padding: 1.25rem 2.25rem 1.5rem 2.25rem !important;
    }

    .font-serif { font-family: 'Instrument Serif', Georgia, serif !important; }
    .font-mono { font-family: 'JetBrains Mono', monospace !important; }

    /* Top Header */
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 9px;
        background: rgba(200, 168, 107, 0.10);
        border: 1px solid #4e422b;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--accent-gold);
        flex-shrink: 0;
    }
    .brand-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 21px;
        line-height: 1.1;
        color: var(--text-primary);
    }
    .brand-sub {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 2px;
    }

    .top-meta-bar {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 18px;
        height: 38px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 12px;
        color: var(--text-secondary);
    }
    .dot-amber {
        width: 7px; height: 7px; border-radius: 50%;
        background: #d4a359; display: inline-block;
    }
    .dot-green {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--accent-green); display: inline-block;
    }
    .v-divider {
        width: 1px;
        height: 24px;
        background: var(--border-subtle);
    }
    .model-meta {
        display: flex;
        flex-direction: column;
        line-height: 1.2;
    }
    .model-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        text-transform: uppercase;
    }
    .model-name {
        font-size: 12.5px;
        color: var(--text-primary);
        font-weight: 500;
        max-width: 180px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Hero Banner */
    .hero-section {
        border-top: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        padding: 1.35rem 0 1.2rem 0;
        margin-top: 0.85rem;
        margin-bottom: 1.35rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 24px;
    }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 0.14em;
        color: #8f9e86;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 42px;
        line-height: 1.05;
        color: var(--text-primary);
        margin: 0;
        font-weight: 400;
    }
    .hero-logline {
        font-family: 'Instrument Serif', Georgia, serif;
        font-style: italic;
        font-size: 17px;
        color: var(--text-muted);
        margin-top: 6px;
    }
    .hero-guardrail-note {
        display: flex;
        align-items: center;
        gap: 12px;
        max-width: 290px;
        font-size: 12px;
        line-height: 1.45;
        color: var(--text-muted);
        padding-bottom: 6px;
    }
    .hero-rule {
        width: 28px;
        height: 1px;
        background: #6b5b3e;
        flex-shrink: 0;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 0.85rem;
    }
    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.14em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .section-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 26px;
        line-height: 1.1;
        color: var(--text-primary);
    }
    .passages-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--text-secondary);
        padding-bottom: 4px;
    }
    .facts-counter {
        text-align: right;
        line-height: 1.1;
    }
    .facts-num {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 32px;
        color: var(--text-primary);
    }
    .facts-label {
        font-size: 11px;
        color: var(--text-muted);
    }

    /* Manuscript Passages */
    .passage-block {
        margin: 0.35rem 0 1.45rem 0;
        padding-right: 0.75rem;
    }
    .passage-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .doc-badge {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        border: 1px solid #594b31;
        background: rgba(200, 168, 107, 0.08);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--accent-gold);
        font-size: 10px;
    }
    .passage-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.13em;
        color: var(--accent-gold);
        text-transform: uppercase;
    }
    .passage-prose {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 21px;
        line-height: 1.62;
        color: #ebe8e0;
    }
    .canon-warning {
        margin-top: 8px;
        padding: 7px 12px;
        border-radius: 6px;
        background: rgba(224, 122, 107, 0.10);
        border: 1px solid rgba(224, 122, 107, 0.35);
        color: #e89b8f;
        font-size: 12px;
    }

    /* Archive Summary Box (Summary Only in Dropdown) */
    .summary-callout {
        background: #161b16;
        border-left: 2px solid var(--accent-gold);
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 10px;
        font-size: 13px;
        line-height: 1.5;
        color: #cfd6cf;
    }
    .summary-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9.5px;
        letter-spacing: 0.12em;
        color: var(--accent-gold);
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Dedicated Single-Chapter Reader Block */
    .reader-sheet {
        background: #161a16;
        border: 1px solid #283028;
        border-radius: 10px;
        padding: 28px 34px;
        margin-top: 0.5rem;
    }
    .reader-chapter-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 0.14em;
        color: var(--accent-gold);
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .reader-chapter-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 32px;
        color: var(--text-primary);
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #252c25;
    }
    .reader-paragraph {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 21px;
        line-height: 1.72;
        color: #ece9e1;
        margin-bottom: 18px;
    }

    /* Scene Suggestion Cards */
    .suggestion-card {
        background: #161b16;
        border: 1px solid #283328;
        border-radius: 8px;
        padding: 13px 15px;
        height: 100%;
        margin-bottom: 10px;
    }
    .suggestion-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9.5px;
        letter-spacing: 0.12em;
        color: var(--accent-gold);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .suggestion-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 18px;
        color: var(--text-primary);
        margin-bottom: 5px;
    }
    .suggestion-body {
        font-size: 12.5px;
        line-height: 1.5;
        color: var(--text-secondary);
    }

    /* Previous Suggestion History Items */
    .history-batch-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.12em;
        color: var(--accent-gold);
        text-transform: uppercase;
        margin: 8px 0 6px 0;
    }
    .history-idea-row {
        background: #141814;
        border: 1px solid #232a23;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .history-idea-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 16px;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .history-idea-beat {
        font-size: 12px;
        line-height: 1.45;
        color: var(--text-secondary);
    }

    /* Graph Container Frame */
    .graph-frame {
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        overflow: hidden;
        background: #141814;
    }
    .legend-bar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        padding: 12px 2px 14px 2px;
        border-bottom: 1px solid var(--border-subtle);
        font-size: 12px;
        color: var(--text-secondary);
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .legend-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Active Inventory & Chapter Relations */
    .inventory-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0 8px 0;
    }
    .inventory-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 21px;
        color: var(--text-primary);
    }
    .inventory-count {
        font-size: 12px;
        color: var(--text-muted);
    }
    .char-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #1f241f;
    }
    .char-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .char-avatar-img {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .char-name {
        font-size: 13.5px;
        font-weight: 500;
        color: var(--text-primary);
        line-height: 1.25;
    }
    .char-loc {
        font-size: 11.5px;
        color: var(--text-muted);
        margin-top: 2px;
    }
    .char-items {
        font-size: 12.5px;
        color: var(--accent-green);
        font-weight: 500;
    }
    .char-empty {
        font-size: 12px;
        color: var(--text-muted);
    }
    .fact-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #b7c2b7;
        background: #171c17;
        border: 1px solid #252d25;
        border-radius: 6px;
        padding: 6px 10px;
        margin-bottom: 6px;
    }

    .graph-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 12px;
        font-size: 11px;
        color: var(--text-muted);
    }

    /* Buttons, Expander, Selectbox, Download Buttons & Inputs */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background-color: #181d18 !important;
        color: var(--text-primary) !important;
        border: 1px solid #2d342d !important;
        border-radius: 7px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--accent-gold) !important;
        color: var(--accent-gold) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: var(--accent-gold-btn) !important;
        border-color: #a68852 !important;
        color: #121412 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #a88953 !important;
        color: #0e100e !important;
    }
    div[data-testid="stPopover"] > button {
        background-color: #181d18 !important;
        color: var(--text-secondary) !important;
        border: 1px solid #2d342d !important;
        border-radius: 7px !important;
        font-size: 12px !important;
        padding: 0.35rem 0.7rem !important;
    }
    div[data-testid="stExpander"] {
        background-color: #151915 !important;
        border: 1px solid #252b25 !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background-color: #181d18 !important;
        border: 1px solid #2d342d !important;
        border-radius: 7px !important;
        color: var(--text-secondary) !important;
        font-size: 12px !important;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: #171b17 !important;
        color: var(--text-primary) !important;
        border: 1px solid #272d27 !important;
        border-radius: 9px !important;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 19px !important;
        line-height: 1.5 !important;
        padding: 14px 16px !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #4f4634 !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextArea"] label {
        display: none !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border-subtle) !important;
    }

    .page-footer {
        border-top: 1px solid var(--border-subtle);
        margin-top: 1.4rem;
        padding-top: 0.85rem;
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.12em;
        color: #555e55;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP NAVIGATION BAR
# ---------------------------------------------------------------------------
top_left, top_right = st.columns([5, 5])

with top_left:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                </svg>
            </div>
            <div>
                <div class="brand-title">Story World Engine</div>
                <div class="brand-sub">Multi-chapter fiction, kept in canon</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    ctrl_spacer, ctrl_status, ctrl_model, ctrl_cfg, ctrl_reset = st.columns([1.2, 2.2, 2.3, 1.3, 1.8])
    with ctrl_status:
        if st.session_state.model_id.strip():
            st.markdown(
                '<div class="top-meta-bar"><span class="status-pill"><span class="dot-green"></span>Advisor active</span><div class="v-divider"></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="top-meta-bar"><span class="status-pill"><span class="dot-amber"></span>Dry-run preview</span><div class="v-divider"></div></div>',
                unsafe_allow_html=True,
            )

    with ctrl_model:
        display_model = (
            st.session_state.model_id.split("/")[-1]
            if st.session_state.model_id.strip()
            else "Unassigned"
        )
        st.markdown(
            f"""
            <div class="model-meta" style="padding-top:3px;">
                <span class="model-eyebrow">SCENE ADVISOR</span>
                <span class="model-name" title="{html.escape(st.session_state.model_id or 'No model configured')}">{html.escape(display_model)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ctrl_cfg:
        with st.popover("Setup"):
            st.caption("Configure your Hugging Face model ID and story metadata.")
            new_model = st.text_input(
                "Model ID",
                value=st.session_state.model_id,
                placeholder="Paste model ID here...",
            )
            new_title = st.text_input("Story Title", value=st.session_state.story_title)
            new_logline = st.text_input("Logline", value=st.session_state.story_logline)
            new_tokens = st.slider("Max Tokens", 80, 512, st.session_state.max_tokens, 20)

            if st.button("Apply Settings", use_container_width=True):
                st.session_state.model_id = new_model
                st.session_state.story_title = new_title or "Untitled Chronicle"
                st.session_state.story_logline = new_logline
                st.session_state.max_tokens = new_tokens
                st.rerun()

            st.download_button(
                label="Export Graph JSON",
                data=st.session_state.graph_manager.export_json(),
                file_name="story_world_graph.json",
                mime="application/json",
                use_container_width=True,
            )

    with ctrl_reset:
        if st.button("↻  Reset world", use_container_width=True):
            st.session_state.graph_manager.clear()
            st.session_state.prompt_engine.clear()
            st.session_state.chapters = {
                1: {
                    "number": 1,
                    "title": "The Beginning",
                    "status": "active",
                    "summary": "",
                    "passages": [],
                }
            }
            st.session_state.active_chapter = 1
            st.session_state.reading_chapter = None
            st.session_state.graph_widescreen = False
            st.session_state.story_completed = False
            st.session_state.scene_suggestions = []
            st.session_state.suggestion_history = []
            st.session_state.clear_editor = True
            st.session_state.story_title = "Untitled Chronicle"
            st.rerun()

# ---------------------------------------------------------------------------
# HERO TITLE BANNER
# ---------------------------------------------------------------------------
active_chap_num = st.session_state.active_chapter
active_chap_data = st.session_state.chapters[active_chap_num]
total_chapters = len(st.session_state.chapters)
total_passages = sum(len(c["passages"]) for c in st.session_state.chapters.values())

hero_eyebrow_text = (
    f"STORY COMPLETE &nbsp;/&nbsp; {total_chapters} CHAPTERS ARCHIVED"
    if st.session_state.story_completed
    else f"ACTIVE CHAPTER {active_chap_num:02d} &nbsp;/&nbsp; {html.escape(active_chap_data['title'].upper())}"
)

st.markdown(
    f"""
    <div class="hero-section">
        <div>
            <div class="hero-eyebrow">{hero_eyebrow_text}</div>
            <h1 class="hero-title">{html.escape(st.session_state.story_title)}</h1>
            <div class="hero-logline">{html.escape(st.session_state.story_logline)}</div>
        </div>
        <div class="hero-guardrail-note">
            <span class="hero-rule"></span>
            <span>You write the story; AI only suggests scene ideas and keeps every chapter in canon.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# WORKSPACE LAYOUT: FULL WIDESCREEN GRAPH OR SPLIT STUDIO (MANUSCRIPT + GRAPH)
# ---------------------------------------------------------------------------
if st.session_state.graph_widescreen:
    _render_knowledge_graph_panel(active_chap_num=active_chap_num, is_widescreen=True)
else:
    left_col, right_col = st.columns([3.1, 2.0], gap="large")

    with left_col:
        # Header + Full Book Word / PDF Download Buttons (with Title Page & Index)
        hdr_title_col, hdr_docx_col, hdr_pdf_col = st.columns([2.5, 1.2, 1.2])
        with hdr_title_col:
            st.markdown(
                f"""
                <div class="section-header" style="margin-bottom:0.4rem;">
                    <div>
                        <div class="section-eyebrow">01 &nbsp;/&nbsp; MANUSCRIPT ROOM</div>
                        <div class="section-title">Ch {active_chap_num:02d}: {html.escape(active_chap_data['title'])} &nbsp;<span class="passages-badge" style="font-family:'Inter',sans-serif;"><span class="dot-green"></span>{total_chapters} chapters · {total_passages} passages</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        file_slug = _slugify_filename(st.session_state.story_title)
        with hdr_docx_col:
            st.download_button(
                label="↓ Book (.docx)",
                data=generate_word_bytes(
                    st.session_state.story_title,
                    st.session_state.story_logline,
                    st.session_state.chapters,
                    include_index=True,
                ),
                file_name=f"{file_slug}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                disabled=(total_passages == 0),
                use_container_width=True,
            )
        with hdr_pdf_col:
            st.download_button(
                label="↓ Book (.pdf)",
                data=generate_pdf_bytes(
                    st.session_state.story_title,
                    st.session_state.story_logline,
                    st.session_state.chapters,
                    include_index=True,
                ),
                file_name=f"{file_slug}.pdf",
                mime="application/pdf",
                disabled=(total_passages == 0),
                use_container_width=True,
            )

        # -------------------------------------------------------------------
        # VIEW A: INDIVIDUAL CHAPTER MANUSCRIPT READER + SINGLE-CHAPTER PDF
        # -------------------------------------------------------------------
        if st.session_state.reading_chapter is not None and st.session_state.reading_chapter in st.session_state.chapters:
            read_num = st.session_state.reading_chapter
            read_chap = st.session_state.chapters[read_num]

            nav_back, nav_switch, nav_pdf = st.columns([1.4, 2.1, 1.5])
            with nav_back:
                if st.button("← Back to Writing", use_container_width=True):
                    st.session_state.reading_chapter = None
                    st.rerun()

            with nav_switch:
                chap_keys = sorted(st.session_state.chapters.keys())
                chap_labels = [f"Chapter {n:02d}: {st.session_state.chapters[n]['title']}" for n in chap_keys]
                chosen_idx = st.selectbox(
                    "Select Individual Chapter to Read",
                    options=range(len(chap_keys)),
                    format_func=lambda i: chap_labels[i],
                    index=chap_keys.index(read_num),
                    label_visibility="collapsed",
                )
                if chap_keys[chosen_idx] != read_num:
                    st.session_state.reading_chapter = chap_keys[chosen_idx]
                    st.rerun()

            with nav_pdf:
                single_chap_pdf = generate_pdf_bytes(
                    st.session_state.story_title,
                    st.session_state.story_logline,
                    {read_num: read_chap},
                    include_index=False,
                )
                st.download_button(
                    label=f"↓ Chapter {read_num:02d} PDF",
                    data=single_chap_pdf,
                    file_name=f"{file_slug}_chapter_{read_num:02d}.pdf",
                    mime="application/pdf",
                    disabled=(len(read_chap["passages"]) == 0),
                    key=f"reader_pdf_btn_{read_num}",
                    use_container_width=True,
                )

            if read_chap["passages"]:
                body_html = "".join(
                    f'<div class="reader-paragraph">{html.escape(p["prose"]).replace(chr(10), "<br>")}</div>'
                    for p in read_chap["passages"]
                )
            else:
                body_html = '<div class="reader-paragraph" style="color:#6c756c;font-style:italic;">No passages written in this chapter yet.</div>'

            st.markdown(
                f"""
                <div class="reader-sheet">
                    <div class="reader-chapter-eyebrow">INDIVIDUAL MANUSCRIPT VIEW &nbsp;•&nbsp; CHAPTER {read_num:02d}</div>
                    <div class="reader-chapter-title">{html.escape(read_chap['title'])}</div>
                    {body_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------------
        # VIEW B: ACTIVE CHAPTER WRITING STUDIO + SUMMARY-ONLY CHAPTER DROPDOWNS
        # -------------------------------------------------------------------
        else:
            # Chapter Lifecycle Toolbar
            if not st.session_state.story_completed:
                bar_rename, bar_next_chap, bar_finish_story = st.columns([1.45, 1.65, 1.65])

                with bar_rename:
                    with st.popover("✎ Rename Chapter", use_container_width=True):
                        edited_chap_title = st.text_input(
                            f"Chapter {active_chap_num:02d} Name",
                            value=active_chap_data["title"],
                        )
                        if st.button("Save Chapter Name", use_container_width=True):
                            if edited_chap_title.strip():
                                active_chap_data["title"] = edited_chap_title.strip()
                                st.session_state.graph_manager.set_chapter_title(
                                    active_chap_num,
                                    edited_chap_title.strip(),
                                )
                                st.rerun()

                with bar_next_chap:
                    with st.popover("+ Start Next Chapter", use_container_width=True):
                        next_num = active_chap_num + 1
                        st.caption(
                            f"Completes Chapter {active_chap_num:02d} and opens a fresh workspace & graph for Chapter {next_num:02d}."
                        )
                        next_chap_name = st.text_input(
                            f"Chapter {next_num:02d} Name",
                            placeholder="e.g., The Sunken Vault",
                            key="next_chapter_name_input",
                        )
                        if st.button(f"Begin Chapter {next_num:02d} →", type="primary", use_container_width=True):
                            _refresh_chapter_summary(active_chap_num)
                            active_chap_data["status"] = "completed"

                            chosen_title = next_chap_name.strip() or f"Chapter {next_num:02d}"
                            st.session_state.chapters[next_num] = {
                                "number": next_num,
                                "title": chosen_title,
                                "status": "active",
                                "summary": "",
                                "passages": [],
                            }
                            st.session_state.active_chapter = next_num
                            st.session_state.graph_manager.ensure_chapter(next_num, chosen_title)
                            st.session_state.scene_suggestions = []
                            st.session_state.suggestion_history = []
                            st.session_state.clear_editor = True
                            st.rerun()

                with bar_finish_story:
                    if st.button("✓ Complete All Chapters", use_container_width=True):
                        _refresh_chapter_summary(active_chap_num)
                        active_chap_data["status"] = "completed"
                        st.session_state.story_completed = True
                        st.session_state.scene_suggestions = []
                        st.rerun()
            else:
                done_banner, reopen_col = st.columns([3.4, 1.4])
                with done_banner:
                    st.markdown(
                        '<div style="padding: 8px 12px; background: #172117; border: 1px solid #2b3d2b; border-radius: 7px; font-size: 12.5px; color: #9ec49e;">✓ All chapters completed. Open any chapter below to view its summary, read it, or open its PDF file.</div>',
                        unsafe_allow_html=True,
                    )
                with reopen_col:
                    if st.button("↺ Reopen Writing", use_container_width=True):
                        st.session_state.story_completed = False
                        st.session_state.chapters[active_chap_num]["status"] = "active"
                        st.rerun()

            # Active Chapter Writing Feed (Fresh workspace per chapter)
            if not st.session_state.story_completed:
                feed_box = st.container(height=275, border=False)
                with feed_box:
                    current_passages = active_chap_data["passages"]
                    if not current_passages:
                        st.markdown(
                            f"""
                            <div style="padding: 1.75rem 0.5rem; color: #6c756c; font-family: 'Instrument Serif', serif; font-size: 20px; font-style: italic;">
                                Chapter {active_chap_num:02d} ({html.escape(active_chap_data['title'])}) is ready. Write your scene below or click "✦ Suggest scenes" for canon-aware scene ideas...
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        for p_idx, entry in enumerate(current_passages, start=1):
                            safe_prose = html.escape(entry["prose"]).replace("\n", "<br>")
                            warning_html = "".join(
                                f'<div class="canon-warning">⚠ {html.escape(w)}</div>'
                                for w in entry.get("warnings", [])
                            )
                            st.markdown(
                                f"""
                                <div class="passage-block">
                                    <div class="passage-header">
                                        <span class="doc-badge">▤</span>
                                        <span class="passage-tag">CHAPTER {active_chap_num:02d} &nbsp;•&nbsp; SCENE {p_idx:02d} &nbsp;/&nbsp; AUTHOR</span>
                                    </div>
                                    <div class="passage-prose">{safe_prose}</div>
                                    {warning_html}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Current AI Scene Suggestion Cards
                if st.session_state.scene_suggestions:
                    current_batch_num = len(st.session_state.suggestion_history)
                    st.markdown(
                        f'<div class="section-eyebrow" style="margin-bottom:8px;">AI SCENE SUGGESTIONS &nbsp;/&nbsp; BATCH {current_batch_num:02d}</div>',
                        unsafe_allow_html=True,
                    )
                    sug_cols = st.columns(len(st.session_state.scene_suggestions))
                    for idx, (col, sug) in enumerate(zip(sug_cols, st.session_state.scene_suggestions)):
                        with col:
                            st.markdown(
                                f"""
                                <div class="suggestion-card">
                                    <div class="suggestion-eyebrow">IDEA 0{idx + 1}</div>
                                    <div class="suggestion-title">{html.escape(sug['title'])}</div>
                                    <div class="suggestion-body">{html.escape(sug['beat'])}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Dropdown for Previous Scene Suggestion Batches (stays available even after closing)
                archived_batches = (
                    st.session_state.suggestion_history[:-1]
                    if st.session_state.scene_suggestions
                    else st.session_state.suggestion_history
                )
                if archived_batches:
                    dropdown_label = (
                        f"🕘 Previous scene ideas ({len(archived_batches)} "
                        f"{'batch' if len(archived_batches) == 1 else 'batches'})"
                    )
                    with st.expander(dropdown_label, expanded=False):
                        for batch in archived_batches:
                            b_num = batch["batch_num"]
                            tag = " — First Suggested Ideas" if b_num == 1 else ""
                            st.markdown(
                                f'<div class="history-batch-header">BATCH {b_num:02d}{tag}</div>',
                                unsafe_allow_html=True,
                            )
                            for idx, old_sug in enumerate(batch["ideas"], start=1):
                                st.markdown(
                                    f"""
                                    <div class="history-idea-row">
                                        <div class="history-idea-title">0{idx}. {html.escape(old_sug['title'])}</div>
                                        <div class="history-idea-beat">{html.escape(old_sug['beat'])}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                # Author Composer Card
                with st.container(border=True):
                    writer_text = st.text_area(
                        "Writer Editor",
                        placeholder=f"Write the next scene for Chapter {active_chap_num:02d} ({active_chap_data['title']})...",
                        height=88,
                        label_visibility="collapsed",
                        key="writer_editor_area",
                    )

                    if st.session_state.scene_suggestions:
                        btn_refresh, btn_close, btn_publish = st.columns([1.2, 0.95, 1.75])
                        with btn_refresh:
                            suggest_clicked = st.button("↻ New ideas", use_container_width=True)
                        with btn_close:
                            close_clicked = st.button("✕ Close", use_container_width=True)
                        with btn_publish:
                            publish_clicked = st.button("Publish passage  →", type="primary", use_container_width=True)
                    else:
                        btn_suggest, btn_publish = st.columns([1.4, 2.0])
                        with btn_suggest:
                            suggest_clicked = st.button("✦ Suggest scenes", use_container_width=True)
                        close_clicked = False
                        with btn_publish:
                            publish_clicked = st.button("Publish passage  →", type="primary", use_container_width=True)

                # Action 1: Close active scene suggestions (keeps batches saved in dropdown)
                if close_clicked:
                    st.session_state.scene_suggestions = []
                    st.rerun()

                # Action 2: Publish writer's prose into current chapter
                if publish_clicked and writer_text.strip():
                    _record_passage_in_active_chapter(
                        prose_text=writer_text.strip(),
                        extractor=extractor,
                    )
                    st.session_state.scene_suggestions = []
                    st.session_state.clear_editor = True
                    st.rerun()

                # Action 3: Generate or refresh 3 new scene ideas & preserve previous batches in history
                if suggest_clicked:
                    all_facts = st.session_state.graph_manager.get_all_facts()
                    recent_chunks = st.session_state.prompt_engine.retrieve_similar_context(writer_text, top_k=2)
                    batch_idx = len(st.session_state.suggestion_history)

                    prev_titles = [
                        idea["title"]
                        for batch in st.session_state.suggestion_history
                        for idea in batch["ideas"]
                    ]
                    combined_note = writer_text.strip()
                    if prev_titles:
                        avoid_clause = f"Avoid repeating these earlier ideas: {', '.join(prev_titles[-6:])}."
                        combined_note = f"{combined_note} {avoid_clause}".strip()

                    with st.spinner("Generating fresh scene ideas..."):
                        if st.session_state.model_id.strip():
                            try:
                                sug_prompt = st.session_state.prompt_engine.build_scene_suggestion_prompt(
                                    active_facts=all_facts,
                                    context_chunks=recent_chunks,
                                    writer_note=combined_note,
                                )
                                raw_sug = generate_text(sug_prompt, st.session_state.model_id.strip(), max_tokens=240)
                                parsed = st.session_state.prompt_engine.parse_scene_suggestions(raw_sug)
                                new_ideas = parsed or build_dry_run_suggestions(
                                    st.session_state.graph_manager.get_character_states(),
                                    batch_index=batch_idx,
                                )
                            except Exception as err:
                                st.error(f"Scene suggestion error: {err}")
                                new_ideas = []
                        else:
                            new_ideas = build_dry_run_suggestions(
                                st.session_state.graph_manager.get_character_states(),
                                batch_index=batch_idx,
                            )

                    if new_ideas:
                        st.session_state.scene_suggestions = new_ideas
                        st.session_state.suggestion_history.append(
                            {
                                "batch_num": batch_idx + 1,
                                "ideas": new_ideas,
                            }
                        )
                    st.rerun()

            # ---------------------------------------------------------------
            # ALL CHAPTERS ARCHIVE (DROPDOWN SHOWS SUMMARY ONLY + READER & PDF)
            # ---------------------------------------------------------------
            st.markdown(
                '<div class="section-eyebrow" style="margin-top: 1.1rem; margin-bottom: 0.55rem;">ALL CHAPTERS ARCHIVE &nbsp;/&nbsp; EXPAND FOR SUMMARY, READER VIEW, OR CHAPTER PDF</div>',
                unsafe_allow_html=True,
            )

            for c_num in sorted(st.session_state.chapters.keys()):
                c_data = st.session_state.chapters[c_num]
                status_tag = "ACTIVE" if c_data["status"] == "active" else "COMPLETED"
                p_cnt = len(c_data["passages"])
                expander_label = f"Chapter {c_num:02d}: {c_data['title']}  —  [{status_tag} · {p_cnt} passages]"

                with st.expander(expander_label, expanded=False):
                    summary_text = c_data.get("summary") or "No passages recorded in this chapter yet."
                    st.markdown(
                        f"""
                        <div class="summary-callout">
                            <div class="summary-label">CHAPTER {c_num:02d} SUMMARY</div>
                            <div>{html.escape(summary_text).replace(chr(10), '<br>')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    read_col, pdf_col = st.columns([1.5, 1.5])
                    with read_col:
                        if st.button(
                            f"📖 Read Chapter {c_num:02d}",
                            key=f"read_chap_btn_{c_num}",
                            use_container_width=True,
                        ):
                            st.session_state.reading_chapter = c_num
                            st.rerun()

                    with pdf_col:
                        chap_pdf_bytes = generate_pdf_bytes(
                            st.session_state.story_title,
                            st.session_state.story_logline,
                            {c_num: c_data},
                            include_index=False,
                        )
                        st.download_button(
                            label=f"↓ Chapter {c_num:02d} PDF",
                            data=chap_pdf_bytes,
                            file_name=f"{file_slug}_chapter_{c_num:02d}.pdf",
                            mime="application/pdf",
                            disabled=(p_cnt == 0),
                            key=f"archive_pdf_btn_{c_num}",
                            use_container_width=True,
                        )

    with right_col:
        _render_knowledge_graph_panel(active_chap_num=active_chap_num, is_widescreen=False)

# ---------------------------------------------------------------------------
# PAGE FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="page-footer">
        <span>SCENE ADVISOR &nbsp;•&nbsp; WIDESCREEN KNOWLEDGE GRAPH</span>
        <span>STORY WORLD ENGINE V2.7</span>
    </div>
    """,
    unsafe_allow_html=True,
)