import json
import os
import tempfile
import urllib.parse
import networkx as nx
from pyvis.network import Network


class KnowledgeGraphManager:
    """Manages chapter-scoped graphs, cumulative cross-chapter canon, and inter-chapter entity bridges."""

    CHARACTER_PALETTES = [
        ("#7aa2f7", "#1d2942"),  # Periwinkle Blue
        ("#f6ad55", "#362614"),  # Warm Amber
        ("#68d391", "#163022"),  # Emerald Mint
        ("#f687b3", "#361927"),  # Rose Quartz
        ("#b794f4", "#271d3b"),  # Soft Violet
        ("#4fd1c5", "#14302e"),  # Teal Cyan
    ]

    def __init__(self):
        self.global_graph = nx.DiGraph()
        self.chapter_graphs: dict[int, nx.DiGraph] = {1: nx.DiGraph()}
        self.chapter_titles: dict[int, str] = {1: "The Beginning"}

    def ensure_chapter(self, chapter_num: int, title: str = "") -> None:
        """Initializes a fresh chapter-scoped graph while preserving cumulative world memory."""
        if chapter_num not in self.chapter_graphs:
            self.chapter_graphs[chapter_num] = nx.DiGraph()
        if title.strip():
            self.chapter_titles[chapter_num] = title.strip()
        elif chapter_num not in self.chapter_titles:
            self.chapter_titles[chapter_num] = f"Chapter {chapter_num:02d}"

    def set_chapter_title(self, chapter_num: int, title: str) -> None:
        """Updates the display name of a specific chapter."""
        if title.strip():
            self.chapter_titles[chapter_num] = title.strip()

    def _infer_node_type(self, predicate: str, is_subject: bool) -> str:
        """Assigns a semantic category to a node for avatar styling and sub-labels."""
        if is_subject:
            return "Character"
        if predicate == "LOCATION":
            return "Location"
        if predicate in ("POSSESSES", "LOST"):
            return "Item"
        if predicate == "IS_A":
            return "Trait"
        if predicate == "MET":
            return "Character"
        return "Entity"

    def _apply_fact_to_graph(self, target_graph: nx.DiGraph, subject: str, predicate: str, obj: str) -> None:
        """Applies a directed triple to a specific graph while enforcing state rules."""
        if subject not in target_graph:
            target_graph.add_node(subject, category=self._infer_node_type(predicate, True))
        if obj not in target_graph:
            target_graph.add_node(obj, category=self._infer_node_type(predicate, False))

        if predicate == "LOST":
            if target_graph.has_edge(subject, obj):
                target_graph.remove_edge(subject, obj)
            target_graph.add_edge(subject, obj, relation="LOST")
            return

        if predicate == "POSSESSES":
            if target_graph.has_edge(subject, obj):
                target_graph.remove_edge(subject, obj)
            target_graph.add_edge(subject, obj, relation="POSSESSES")
            return

        if predicate == "LOCATION":
            old_locations = [
                target for _, target, data in target_graph.out_edges(subject, data=True)
                if data.get("relation") == "LOCATION"
            ]
            for loc in old_locations:
                target_graph.remove_edge(subject, loc)

        target_graph.add_edge(subject, obj, relation=predicate)

    def add_fact(self, subject: str, predicate: str, obj: str, chapter_num: int = 1) -> None:
        """Records a triple in both the active chapter's graph and the cumulative world graph."""
        subject = subject.strip().title()
        obj = obj.strip().title()
        predicate = predicate.strip().upper()

        if not subject or not obj or subject == obj:
            return

        self.ensure_chapter(chapter_num)
        self._apply_fact_to_graph(self.global_graph, subject, predicate, obj)
        self._apply_fact_to_graph(self.chapter_graphs[chapter_num], subject, predicate, obj)

    def get_entity_facts(self, entity_name: str) -> list[str]:
        """Retrieves 1-hop and 2-hop facts across all chapters from the cumulative graph."""
        entity_name = entity_name.strip().title()
        if entity_name not in self.global_graph:
            return []

        facts = []
        for _, target, data in self.global_graph.out_edges(entity_name, data=True):
            rel = data.get("relation", "RELATED_TO")
            facts.append(f"{entity_name} {rel} {target}")

            for _, second_target, second_data in self.global_graph.out_edges(target, data=True):
                second_rel = second_data.get("relation", "RELATED_TO")
                facts.append(f"{target} {second_rel} {second_target}")

        for source, _, data in self.global_graph.in_edges(entity_name, data=True):
            rel = data.get("relation", "RELATED_TO")
            facts.append(f"{source} {rel} {entity_name}")

        return list(dict.fromkeys(facts))

    def get_chapter_facts(self, chapter_num: int) -> list[str]:
        """Returns human-readable triples recorded specifically in the given chapter."""
        cg = self.chapter_graphs.get(chapter_num)
        if not cg:
            return []
        return [
            f"{u} --[{data.get('relation', 'RELATED_TO').lower()}]--> {v}"
            for u, v, data in cg.edges(data=True)
        ]

    def get_all_facts(self) -> list[str]:
        """Returns all cumulative active triples across the entire story."""
        return [
            f"{u} --[{data.get('relation', 'RELATED_TO')}]--> {v}"
            for u, v, data in self.global_graph.edges(data=True)
        ]

    def get_character_states(self) -> dict:
        """Summarizes cumulative inventory, lost items, and current location per character."""
        states = {}
        for node, attrs in self.global_graph.nodes(data=True):
            if attrs.get("category") == "Character":
                inventory = []
                lost = []
                location = "Unknown"

                for _, target, data in self.global_graph.out_edges(node, data=True):
                    rel = data.get("relation")
                    if rel == "POSSESSES":
                        inventory.append(target)
                    elif rel == "LOST":
                        lost.append(target)
                    elif rel == "LOCATION":
                        location = target

                states[node] = {
                    "location": location,
                    "inventory": inventory,
                    "lost_items": lost,
                }
        return states

    def build_inter_chapter_graph(self) -> nx.DiGraph:
        """
        Builds the Inter-Chapter Map containing ONLY:
        1. Chapter nodes (Ch 01: <Name>, Ch 02: <Name>, ...)
        2. Connected Characters and Items (entities of category Character/Item that bridge
           across >= 2 chapters, plus items actively linked to those connected characters).
        Locations, traits, and isolated single-chapter background nodes are excluded.
        """
        macro = nx.DiGraph()
        sorted_chaps = sorted(self.chapter_graphs.keys())
        if not sorted_chaps:
            return macro

        # Map each Character / Item to the set of chapters in which it appears
        entity_chapters: dict[str, set[int]] = {}
        entity_category: dict[str, str] = {}

        for c_num in sorted_chaps:
            cg = self.chapter_graphs[c_num]
            for node, attrs in cg.nodes(data=True):
                cat = attrs.get("category", "Entity")
                if cat in ("Character", "Item"):
                    entity_category[node] = cat
                    entity_chapters.setdefault(node, set()).add(c_num)

        # Add Chapter nodes
        chap_labels: dict[int, str] = {}
        for c_num in sorted_chaps:
            title = self.chapter_titles.get(c_num, f"Chapter {c_num:02d}")
            label = f"Ch {c_num:02d}: {title}"
            chap_labels[c_num] = label
            macro.add_node(label, category="Chapter", chapter_num=c_num)

        # Identify Characters & Items connected across >= 2 chapters
        if len(sorted_chaps) >= 2:
            connected_entities = {
                ent for ent, chaps in entity_chapters.items() if len(chaps) >= 2
            }
            # Also include Items actively possessed/lost by connected cross-chapter Characters
            for ent in list(connected_entities):
                if entity_category.get(ent) == "Character" and ent in self.global_graph:
                    for _, target, data in self.global_graph.out_edges(ent, data=True):
                        if data.get("relation") in ("POSSESSES", "LOST") and entity_category.get(target) == "Item":
                            connected_entities.add(target)
        else:
            # Single-chapter fallback: show only Characters & Items with active connections
            connected_entities = {
                ent for ent in entity_chapters
                if ent in self.global_graph and self.global_graph.degree(ent) > 0
            }

        # Add connected Character & Item nodes and link them to their chapters
        for ent in sorted(connected_entities):
            cat = entity_category.get(ent, "Entity")
            macro.add_node(ent, category=cat)
            for c_num in sorted(entity_chapters.get(ent, [])):
                edge_text = "appears in" if cat == "Character" else "featured in"
                macro.add_edge(chap_labels[c_num], ent, relation=edge_text, edge_kind="chapter_link")

        # Preserve direct Character <-> Item / Character <-> Character links between connected entities
        for u, v, data in self.global_graph.edges(data=True):
            if u in connected_entities and v in connected_entities:
                rel = data.get("relation", "RELATED_TO")
                if rel in ("POSSESSES", "LOST", "MET"):
                    macro.add_edge(u, v, relation=rel, edge_kind="entity_link")

        return macro

    def get_inter_chapter_relations_list(self) -> list[str]:
        """Returns readable descriptions of connected characters & items bridging chapters."""
        macro = self.build_inter_chapter_graph()
        links = []
        for node, attrs in macro.nodes(data=True):
            cat = attrs.get("category")
            if cat in ("Character", "Item"):
                chaps = [
                    src for src, _, d in macro.in_edges(node, data=True)
                    if d.get("edge_kind") == "chapter_link"
                ]
                item_rels = [
                    f"{d.get('relation', '').lower()} {tgt}"
                    for _, tgt, d in macro.out_edges(node, data=True)
                    if d.get("edge_kind") == "entity_link"
                ]
                chap_str = " & ".join(chaps) if chaps else "Story arc"
                extra = f" ({', '.join(item_rels)})" if item_rels else ""
                links.append(f"[{cat.upper()}] {node}  →  connects {chap_str}{extra}")
        return links

    @classmethod
    def _build_avatar_svg(cls, label: str, category: str) -> str:
        """Generates an illustrated SVG avatar tailored to Chapter, Character, Item, Location, or Trait."""
        lower = label.lower()

        if category == "Chapter":
            accent, bg = "#c8a86b", "#2a2418"
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="88" height="88" viewBox="0 0 88 88">
                <circle cx="44" cy="44" r="40" fill="{accent}" fill-opacity="0.16" stroke="{accent}" stroke-opacity="0.5" stroke-width="1.5"/>
                <circle cx="44" cy="44" r="31" fill="{bg}" stroke="{accent}" stroke-width="2.4"/>
                <path d="M28 32 H41 C43 32 44 33.5 44 35.5 V56 C44 54.5 42.5 53.5 41 53.5 H28 Z" fill="{accent}" fill-opacity="0.25" stroke="{accent}" stroke-width="2.2" stroke-linejoin="round"/>
                <path d="M60 32 H47 C45 32 44 33.5 44 35.5 V56 C44 54.5 45.5 53.5 47 53.5 H60 Z" fill="{accent}" fill-opacity="0.25" stroke="{accent}" stroke-width="2.2" stroke-linejoin="round"/>
            </svg>"""

        elif category == "Character":
            accent, bg = cls.CHARACTER_PALETTES[sum(ord(c) for c in label) % len(cls.CHARACTER_PALETTES)]
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84">
                <circle cx="42" cy="42" r="39" fill="{accent}" fill-opacity="0.16" stroke="{accent}" stroke-opacity="0.45" stroke-width="1.5"/>
                <circle cx="42" cy="42" r="30" fill="{bg}" stroke="{accent}" stroke-width="2.2"/>
                <circle cx="42" cy="33" r="10.5" fill="{accent}"/>
                <path d="M23 63 C23 50, 61 50, 61 63" fill="{accent}" stroke="{accent}" stroke-width="2" stroke-linecap="round"/>
                <circle cx="38.5" cy="32" r="1.4" fill="{bg}"/>
                <circle cx="45.5" cy="32" r="1.4" fill="{bg}"/>
            </svg>"""

        elif category == "Item":
            accent, bg = "#68d391", "#162b20"
            if any(k in lower for k in ("key", "lock", "cipher")):
                icon = f"""
                    <circle cx="34" cy="36" r="8" fill="none" stroke="{accent}" stroke-width="3"/>
                    <path d="M41 39 L56 54 M51 49 L55 45 M55 53 L59 49" stroke="{accent}" stroke-width="3" stroke-linecap="round"/>
                """
            elif any(k in lower for k in ("map", "book", "scroll", "letter", "journal", "diary", "note", "page")):
                icon = f"""
                    <rect x="28" y="26" width="28" height="32" rx="4" fill="none" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="34" y1="35" x2="50" y2="35" stroke="{accent}" stroke-width="2.2" stroke-linecap="round"/>
                    <line x1="34" y1="42" x2="50" y2="42" stroke="{accent}" stroke-width="2.2" stroke-linecap="round"/>
                    <line x1="34" y1="49" x2="44" y2="49" stroke="{accent}" stroke-width="2.2" stroke-linecap="round"/>
                """
            elif any(k in lower for k in ("sword", "dagger", "blade", "knife", "axe", "spear", "gun", "weapon")):
                icon = f"""
                    <path d="M54 28 L32 50 M28 46 L38 56 M27 57 L33 51" stroke="{accent}" stroke-width="3" stroke-linecap="round"/>
                    <polygon points="54,28 57,33 51,31" fill="{accent}"/>
                """
            elif any(k in lower for k in ("lantern", "lamp", "torch", "orb", "crystal", "gem", "ring", "stone")):
                icon = f"""
                    <polygon points="42,24 55,39 42,58 29,39" fill="{accent}" fill-opacity="0.28" stroke="{accent}" stroke-width="2.6" stroke-linejoin="round"/>
                    <line x1="29" y1="39" x2="55" y2="39" stroke="{accent}" stroke-width="2"/>
                """
            else:
                icon = f"""
                    <rect x="27" y="33" width="30" height="21" rx="4" fill="{accent}" fill-opacity="0.22" stroke="{accent}" stroke-width="2.6"/>
                    <path d="M33 33 V29 C33 25, 51 25, 51 29 V33" fill="none" stroke="{accent}" stroke-width="2.4"/>
                    <circle cx="42" cy="43" r="3" fill="{accent}"/>
                """
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84">
                <circle cx="42" cy="42" r="38" fill="{accent}" fill-opacity="0.14" stroke="{accent}" stroke-opacity="0.4" stroke-width="1.5"/>
                <circle cx="42" cy="42" r="29" fill="{bg}" stroke="{accent}" stroke-width="2.2"/>
                {icon}
            </svg>"""

        elif category == "Location":
            accent, bg = "#f08070", "#331c19"
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84">
                <circle cx="42" cy="42" r="38" fill="{accent}" fill-opacity="0.15" stroke="{accent}" stroke-opacity="0.42" stroke-width="1.5"/>
                <circle cx="42" cy="42" r="29" fill="{bg}" stroke="{accent}" stroke-width="2.2"/>
                <path d="M31 56 V34 L35 30 L39 34 V32 L42 28 L45 32 V34 L49 30 L53 34 V56 Z" fill="{accent}" fill-opacity="0.28" stroke="{accent}" stroke-width="2.2" stroke-linejoin="round"/>
                <path d="M39 56 V47 C39 44.5, 45 44.5, 45 47 V56" fill="{bg}" stroke="{accent}" stroke-width="2"/>
            </svg>"""

        else:
            accent, bg = "#e2b059", "#302614"
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84">
                <circle cx="42" cy="42" r="38" fill="{accent}" fill-opacity="0.15" stroke="{accent}" stroke-opacity="0.42" stroke-width="1.5"/>
                <circle cx="42" cy="42" r="29" fill="{bg}" stroke="{accent}" stroke-width="2.2"/>
                <polygon points="42,25 46.5,36.5 58.5,37.5 49.5,45.5 52,57.5 42,51 32,57.5 34.5,45.5 25.5,37.5 37.5,36.5" fill="{accent}" fill-opacity="0.85"/>
            </svg>"""

        return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

    def get_display_graph(self, view_mode: str = "chapter", chapter_num: int = 1) -> nx.DiGraph:
        """Selects the active chapter graph, cumulative graph, or connected inter-chapter graph."""
        if view_mode == "story_complete":
            return self.build_inter_chapter_graph()
        if view_mode == "global":
            return self.global_graph
        return self.chapter_graphs.get(chapter_num, nx.DiGraph())

    def to_pyvis_html(self, view_mode: str = "chapter", chapter_num: int = 1) -> str:
        """Renders the selected graph with illustrated SVG avatars and architectural grid."""
        target_graph = self.get_display_graph(view_mode=view_mode, chapter_num=chapter_num)

        net = Network(
            height="375px",
            width="100%",
            directed=True,
            bgcolor="#141814",
            font_color="#f2efe9",
        )
        spring_len = 165 if view_mode == "story_complete" else 145
        net.barnes_hut(gravity=-3200, central_gravity=0.32, spring_length=spring_len, damping=0.25)

        for node, attrs in target_graph.nodes(data=True):
            category = attrs.get("category", "Entity")
            sub_tag = "STORY CHAPTER" if category == "Chapter" else category.upper()
            label_html = f"<b>{node}</b>\n<code>{sub_tag}</code>"

            net.add_node(
                node,
                label=label_html,
                title=f"{node} ({category})",
                shape="image",
                image=self._build_avatar_svg(node, category),
                size=34 if category in ("Chapter", "Character") else 28,
            )

        for u, v, data in target_graph.edges(data=True):
            relation = data.get("relation", "").replace("_", " ").lower()
            is_lost = data.get("relation") == "LOST"
            edge_kind = data.get("edge_kind", "")

            edge_color = (
                "#c8a86b" if edge_kind == "chapter_link"
                else "#d96b5b" if is_lost
                else "#68b684" if edge_kind == "entity_link"
                else "#637063"
            )
            net.add_edge(
                u,
                v,
                label=relation,
                title=relation.upper(),
                color=edge_color,
                width=2.0 if edge_kind else 1.8,
                dashes=is_lost,
                arrows={"to": {"enabled": True, "scaleFactor": 0.55}},
            )

        net.set_options("""
        {
          "nodes": {
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.55)",
              "size": 10,
              "x": 0,
              "y": 4
            },
            "font": {
              "multi": "html",
              "color": "#f2efe9",
              "size": 13,
              "face": "Inter, sans-serif",
              "bold": { "color": "#ffffff", "size": 13, "face": "Inter, sans-serif" },
              "mono": { "color": "#8e998e", "size": 9.5, "face": "JetBrains Mono, monospace" }
            }
          },
          "edges": {
            "smooth": { "type": "continuous", "roundness": 0.18 },
            "font": {
              "color": "#c8cfc8",
              "size": 10.5,
              "face": "Inter, sans-serif",
              "strokeWidth": 3,
              "strokeColor": "#141814",
              "align": "top"
            }
          },
          "interaction": {
            "hover": true,
            "zoomView": true,
            "dragNodes": true
          }
        }
        """)

        try:
            raw_html = net.generate_html()
        except Exception:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                tmp_path = tmp.name
            with open(tmp_path, "r", encoding="utf-8") as f:
                raw_html = f.read()
            os.remove(tmp_path)

        badge_text = (
            "CONNECTED CAST & RELICS / INTER-CHAPTER"
            if view_mode == "story_complete"
            else f"CHAPTER {chapter_num:02d} GRAPH / LIVE"
            if view_mode == "chapter"
            else "CUMULATIVE WORLD / LIVE"
        )

        custom_overlay = f"""
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background-color: #141814 !important;
            overflow: hidden;
          }}
          #mynetwork {{
            border: none !important;
            background-color: #141814 !important;
            background-image:
              radial-gradient(circle at 50% 45%, rgba(200, 168, 107, 0.05), transparent 65%),
              linear-gradient(to right, rgba(255, 255, 255, 0.04) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(255, 255, 255, 0.04) 1px, transparent 1px) !important;
            background-size: 100% 100%, 90px 90px, 90px 90px !important;
          }}
          .live-badge {{
            position: fixed;
            bottom: 12px;
            right: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.12em;
            color: #7c877c;
            pointer-events: none;
            z-index: 999;
          }}
        </style>
        <div class="live-badge">{badge_text}</div>
        """
        return raw_html.replace("</body>", f"{custom_overlay}</body>")

    def export_json(self) -> str:
        """Serializes chapter-wise graphs, cumulative state, and inter-chapter links to JSON."""
        payload = {
            "chapter_titles": self.chapter_titles,
            "chapters": {
                str(c_num): nx.node_link_data(cg)
                for c_num, cg in self.chapter_graphs.items()
            },
            "inter_chapter_graph": nx.node_link_data(self.build_inter_chapter_graph()),
            "cumulative_world": nx.node_link_data(self.global_graph),
        }
        return json.dumps(payload, indent=2)

    def clear(self) -> None:
        """Resets all chapter graphs and cumulative world state."""
        self.global_graph.clear()
        self.chapter_graphs = {1: nx.DiGraph()}
        self.chapter_titles = {1: "The Beginning"}