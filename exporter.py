import html
import io
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def _set_cell_borders(cell, bottom_hex: str | None = None) -> None:
    """Removes default table borders and optionally adds a subtle bottom rule."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        if side == "bottom" and bottom_hex:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), bottom_hex)
        else:
            el.set(qn("w:val"), "none")
        tc_borders.append(el)
    tc_pr.append(tc_borders)


def _add_paragraph_bottom_border(paragraph, color_hex: str = "B8A88A", size: str = "6") -> None:
    """Adds a clean horizontal rule underneath a paragraph."""
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), color_hex)
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def _add_xml_field(paragraph, field_text: str, font_name: str = "Georgia", font_size: int = 9) -> None:
    """Inserts a dynamic Word XML field such as PAGE or TOC."""
    run_begin = paragraph.add_run()
    run_begin.font.name = font_name
    run_begin.font.size = Pt(font_size)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run_begin._r.append(fld_begin)

    run_instr = paragraph.add_run()
    run_instr.font.name = font_name
    run_instr.font.size = Pt(font_size)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {field_text} "
    run_instr._r.append(instr)

    run_sep = paragraph.add_run()
    run_sep.font.name = font_name
    run_sep.font.size = Pt(font_size)
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run_sep._r.append(fld_sep)

    run_end = paragraph.add_run()
    run_end.font.name = font_name
    run_end.font.size = Pt(font_size)
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run_end._r.append(fld_end)


def generate_word_bytes(
    story_title: str,
    story_logline: str,
    chapters: dict,
    include_index: bool = True,
) -> bytes:
    """Generates a formatted storybook Word (.docx) manuscript with title page, index, and chapter breaks."""
    doc = Document()
    title_clean = (story_title or "Untitled Chronicle").strip()
    logline_clean = (story_logline or "").strip()
    sorted_nums = sorted(chapters.keys())

    # Configure standard book page margins & running headers/footers
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.2)
    section.right_margin = Inches(1.2)
    section.different_first_page_header_footer = True

    # Running Header (Content pages)
    header_para = section.header.paragraphs[0]
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_run = header_para.add_run(title_clean.upper())
    h_run.font.name = "Georgia"
    h_run.font.size = Pt(8.5)
    h_run.font.color.rgb = RGBColor(125, 120, 110)

    # Running Footer with dynamic Page Number
    footer_para = section.footer.paragraphs[0]
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    f_left = footer_para.add_run("— ")
    f_left.font.name = "Georgia"
    f_left.font.size = Pt(9)
    f_left.font.color.rgb = RGBColor(125, 120, 110)
    _add_xml_field(footer_para, "PAGE", font_name="Georgia", font_size=9)
    f_right = footer_para.add_run(" —")
    f_right.font.name = "Georgia"
    f_right.font.size = Pt(9)
    f_right.font.color.rgb = RGBColor(125, 120, 110)

    # Base Normal Style
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Georgia"
    normal_style.font.size = Pt(11.5)
    normal_style.font.color.rgb = RGBColor(32, 32, 32)

    # -----------------------------------------------------------------------
    # PAGE 1: BOOK TITLE PAGE
    # -----------------------------------------------------------------------
    spacer_top = doc.add_paragraph()
    spacer_top.paragraph_format.space_before = Pt(150)
    spacer_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    orn = spacer_top.add_run("✦")
    orn.font.name = "Georgia"
    orn.font.size = Pt(16)
    orn.font.color.rgb = RGBColor(145, 118, 68)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(14)
    title_p.paragraph_format.space_after = Pt(12)
    t_run = title_p.add_run(title_clean)
    t_run.font.name = "Georgia"
    t_run.font.size = Pt(28)
    t_run.font.bold = True
    t_run.font.color.rgb = RGBColor(24, 24, 24)
    _add_paragraph_bottom_border(title_p, color_hex="917644", size="8")

    if logline_clean:
        sub_p = doc.add_paragraph()
        sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub_p.paragraph_format.space_before = Pt(14)
        sub_p.paragraph_format.space_after = Pt(24)
        sub_p.paragraph_format.left_indent = Inches(0.6)
        sub_p.paragraph_format.right_indent = Inches(0.6)
        s_run = sub_p.add_run(logline_clean)
        s_run.font.name = "Georgia"
        s_run.font.size = Pt(13)
        s_run.font.italic = True
        s_run.font.color.rgb = RGBColor(90, 90, 90)

    total_scenes = sum(len(chapters[n].get("passages", [])) for n in sorted_nums)
    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.paragraph_format.space_before = Pt(170)
    m_run = meta_p.add_run(f"MANUSCRIPT EDITION  •  {len(sorted_nums)} CHAPTERS  •  {total_scenes} SCENES")
    m_run.font.name = "Georgia"
    m_run.font.size = Pt(9)
    m_run.font.color.rgb = RGBColor(120, 115, 105)

    doc.add_page_break()

    # -----------------------------------------------------------------------
    # PAGE 2: TABLE OF CONTENTS / INDEX PAGE
    # -----------------------------------------------------------------------
    if include_index and sorted_nums:
        idx_header = doc.add_paragraph()
        idx_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        idx_header.paragraph_format.space_before = Pt(24)
        idx_header.paragraph_format.space_after = Pt(18)
        ih_run = idx_header.add_run("CONTENTS")
        ih_run.font.name = "Georgia"
        ih_run.font.size = Pt(18)
        ih_run.font.bold = True
        ih_run.font.color.rgb = RGBColor(30, 30, 30)
        _add_paragraph_bottom_border(idx_header, color_hex="917644", size="6")

        toc_table = doc.add_table(rows=len(sorted_nums), cols=3)
        toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        toc_table.autofit = False

        for row_idx, c_num in enumerate(sorted_nums):
            c_info = chapters[c_num]
            c_title = (c_info.get("title") or f"Chapter {c_num:02d}").strip()
            scene_cnt = len(c_info.get("passages", []))

            row = toc_table.rows[row_idx]
            cell_num, cell_title, cell_scenes = row.cells[0], row.cells[1], row.cells[2]
            cell_num.width = Inches(1.35)
            cell_title.width = Inches(3.65)
            cell_scenes.width = Inches(1.1)

            for c in (cell_num, cell_title, cell_scenes):
                _set_cell_borders(c, bottom_hex="E5E0D5")

            p0 = cell_num.paragraphs[0]
            p0.paragraph_format.space_before = Pt(7)
            p0.paragraph_format.space_after = Pt(7)
            r0 = p0.add_run(f"CHAPTER {c_num:02d}")
            r0.font.name = "Georgia"
            r0.font.size = Pt(9.5)
            r0.font.bold = True
            r0.font.color.rgb = RGBColor(145, 118, 68)

            p1 = cell_title.paragraphs[0]
            p1.paragraph_format.space_before = Pt(7)
            p1.paragraph_format.space_after = Pt(7)
            r1 = p1.add_run(c_title)
            r1.font.name = "Georgia"
            r1.font.size = Pt(11.5)
            r1.font.color.rgb = RGBColor(32, 32, 32)

            p2 = cell_scenes.paragraphs[0]
            p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p2.paragraph_format.space_before = Pt(7)
            p2.paragraph_format.space_after = Pt(7)
            r2 = p2.add_run(f"{scene_cnt} {'scene' if scene_cnt == 1 else 'scenes'}")
            r2.font.name = "Georgia"
            r2.font.size = Pt(9.5)
            r2.font.italic = True
            r2.font.color.rgb = RGBColor(115, 115, 115)

        doc.add_page_break()

    # -----------------------------------------------------------------------
    # CHAPTER PAGES (EACH CHAPTER STARTS ON A FRESH PAGE)
    # -----------------------------------------------------------------------
    for idx, c_num in enumerate(sorted_nums):
        if idx > 0:
            doc.add_page_break()

        c_info = chapters[c_num]
        c_title = (c_info.get("title") or f"Chapter {c_num:02d}").strip()
        passages = c_info.get("passages", [])

        eyebrow_p = doc.add_paragraph()
        eyebrow_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        eyebrow_p.paragraph_format.space_before = Pt(18)
        eyebrow_p.paragraph_format.space_after = Pt(4)
        eb_run = eyebrow_p.add_run(f"CHAPTER {c_num:02d}")
        eb_run.font.name = "Georgia"
        eb_run.font.size = Pt(10)
        eb_run.font.bold = True
        eb_run.font.color.rgb = RGBColor(145, 118, 68)

        chap_heading = doc.add_heading(level=1)
        chap_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        chap_heading.paragraph_format.space_before = Pt(2)
        chap_heading.paragraph_format.space_after = Pt(18)
        ch_run = chap_heading.add_run(c_title)
        ch_run.font.name = "Georgia"
        ch_run.font.size = Pt(20)
        ch_run.font.bold = False
        ch_run.font.color.rgb = RGBColor(25, 25, 25)
        _add_paragraph_bottom_border(chap_heading, color_hex="D6CEC0", size="4")

        if not passages:
            empty_p = doc.add_paragraph()
            empty_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            empty_p.paragraph_format.space_before = Pt(24)
            er = empty_p.add_run("No passages written in this chapter yet.")
            er.font.name = "Georgia"
            er.font.italic = True
            er.font.color.rgb = RGBColor(130, 130, 130)
            continue

        for p_idx, entry in enumerate(passages):
            prose = (entry.get("prose") or "").strip()
            if not prose:
                continue

            sub_paragraphs = [seg.strip() for seg in re.split(r"\n\s*\n|\n", prose) if seg.strip()]
            for s_idx, para_text in enumerate(sub_paragraphs):
                body_p = doc.add_paragraph()
                body_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                body_p.paragraph_format.line_spacing = 1.45
                body_p.paragraph_format.space_after = Pt(8)
                # Classic novel indentation after the opening paragraph of a scene
                if s_idx > 0:
                    body_p.paragraph_format.first_line_indent = Inches(0.3)
                b_run = body_p.add_run(para_text)
                b_run.font.name = "Georgia"
                b_run.font.size = Pt(11.5)
                b_run.font.color.rgb = RGBColor(28, 28, 28)

            # Ornamental scene break between distinct passages
            if p_idx < len(passages) - 1:
                sep_p = doc.add_paragraph()
                sep_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                sep_p.paragraph_format.space_before = Pt(8)
                sep_p.paragraph_format.space_after = Pt(10)
                sr = sep_p.add_run("*   *   *")
                sr.font.name = "Georgia"
                sr.font.size = Pt(10)
                sr.font.color.rgb = RGBColor(145, 118, 68)

    out_buf = io.BytesIO()
    doc.save(out_buf)
    return out_buf.getvalue()


def generate_pdf_bytes(
    story_title: str,
    story_logline: str,
    chapters: dict,
    include_index: bool = True,
) -> bytes:
    """Generates a typeset storybook PDF with title page, exact-page-number index, and chapter pages."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    title_clean = (story_title or "Untitled Chronicle").strip()
    logline_clean = (story_logline or "").strip()
    sorted_nums = sorted(chapters.keys())

    class _StoryBookDocTemplate(SimpleDocTemplate):
        def __init__(self, filename, **kw):
            super().__init__(filename, **kw)
            self.chapter_page_map: dict[int, int] = {}

        def afterFlowable(self, flowable):
            chap_num = getattr(flowable, "_chapter_num", None)
            if chap_num is not None:
                self.chapter_page_map[chap_num] = self.page

    gold_accent = colors.HexColor("#8C6F3E")
    ink_dark = colors.HexColor("#1C1C1C")
    ink_body = colors.HexColor("#262626")
    ink_muted = colors.HexColor("#6E6A63")
    rule_light = colors.HexColor("#DCD6CA")

    style_ornament = ParagraphStyle(
        "Ornament",
        fontName="Times-Roman",
        fontSize=15,
        leading=20,
        alignment=TA_CENTER,
        textColor=gold_accent,
    )
    style_book_title = ParagraphStyle(
        "BookTitle",
        fontName="Times-Bold",
        fontSize=28,
        leading=34,
        alignment=TA_CENTER,
        textColor=ink_dark,
        spaceAfter=10,
    )
    style_logline = ParagraphStyle(
        "BookLogline",
        fontName="Times-Italic",
        fontSize=13,
        leading=19,
        alignment=TA_CENTER,
        textColor=ink_muted,
        leftIndent=36,
        rightIndent=36,
    )
    style_edition = ParagraphStyle(
        "EditionMeta",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=ink_muted,
    )
    style_index_heading = ParagraphStyle(
        "IndexHeading",
        fontName="Times-Bold",
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=ink_dark,
        spaceAfter=8,
    )
    style_toc_chap = ParagraphStyle(
        "TocChap",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=13,
        alignment=TA_LEFT,
        textColor=gold_accent,
    )
    style_toc_title = ParagraphStyle(
        "TocTitle",
        fontName="Times-Roman",
        fontSize=11.5,
        leading=15,
        alignment=TA_LEFT,
        textColor=ink_dark,
    )
    style_toc_page = ParagraphStyle(
        "TocPage",
        fontName="Times-Bold",
        fontSize=11,
        leading=15,
        alignment=TA_RIGHT,
        textColor=ink_dark,
    )
    style_chap_eyebrow = ParagraphStyle(
        "ChapEyebrow",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=gold_accent,
        spaceAfter=4,
    )
    style_chap_title = ParagraphStyle(
        "ChapTitle",
        fontName="Times-Roman",
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=ink_dark,
        spaceAfter=10,
    )
    style_body_first = ParagraphStyle(
        "BodyFirst",
        fontName="Times-Roman",
        fontSize=11.8,
        leading=18.2,
        alignment=TA_JUSTIFY,
        textColor=ink_body,
        spaceAfter=8,
    )
    style_body_indent = ParagraphStyle(
        "BodyIndent",
        parent=style_body_first,
        firstLineIndent=22,
    )
    style_scene_sep = ParagraphStyle(
        "SceneSep",
        fontName="Times-Roman",
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        textColor=gold_accent,
        spaceBefore=6,
        spaceAfter=10,
    )

    def _draw_title_page(canvas, doc_obj):
        canvas.saveState()
        w, h = letter
        canvas.setStrokeColor(rule_light)
        canvas.setLineWidth(0.75)
        canvas.rect(0.75 * inch, 0.75 * inch, w - 1.5 * inch, h - 1.5 * inch)
        canvas.restoreState()

    def _draw_content_page(canvas, doc_obj):
        canvas.saveState()
        w, h = letter
        # Running Header
        canvas.setFont("Times-Roman", 8.5)
        canvas.setFillColor(ink_muted)
        canvas.drawCentredString(w / 2.0, h - 0.65 * inch, title_clean.upper())
        canvas.setStrokeColor(rule_light)
        canvas.setLineWidth(0.5)
        canvas.line(1.15 * inch, h - 0.74 * inch, w - 1.15 * inch, h - 0.74 * inch)

        # Running Footer Page Number
        canvas.setFont("Times-Roman", 9.5)
        canvas.setFillColor(ink_muted)
        canvas.drawCentredString(w / 2.0, 0.62 * inch, f"— {canvas.getPageNumber()} —")
        canvas.restoreState()

    def _build_story_elements(page_map: dict[int, int]) -> list:
        story: list = []
        total_scenes = sum(len(chapters[n].get("passages", [])) for n in sorted_nums)

        # 1. Title Page
        story.append(Spacer(1, 2.1 * inch))
        story.append(Paragraph("✦", style_ornament))
        story.append(Spacer(1, 0.16 * inch))
        story.append(Paragraph(html.escape(title_clean), style_book_title))
        story.append(HRFlowable(width="38%", thickness=1, color=gold_accent, spaceBefore=4, spaceAfter=16))
        if logline_clean:
            story.append(Paragraph(html.escape(logline_clean), style_logline))
        story.append(Spacer(1, 2.6 * inch))
        story.append(
            Paragraph(
                f"MANUSCRIPT EDITION &nbsp;•&nbsp; {len(sorted_nums)} CHAPTERS &nbsp;•&nbsp; {total_scenes} SCENES",
                style_edition,
            )
        )
        story.append(PageBreak())

        # 2. Index / Table of Contents Page
        if include_index and sorted_nums:
            story.append(Spacer(1, 0.35 * inch))
            story.append(Paragraph("CONTENTS", style_index_heading))
            story.append(HRFlowable(width="28%", thickness=0.8, color=gold_accent, spaceBefore=2, spaceAfter=22))

            toc_rows = []
            for c_num in sorted_nums:
                c_info = chapters[c_num]
                c_title = (c_info.get("title") or f"Chapter {c_num:02d}").strip()
                pg_num = page_map.get(c_num, "—")
                toc_rows.append(
                    [
                        Paragraph(f"CHAPTER {c_num:02d}", style_toc_chap),
                        Paragraph(html.escape(c_title), style_toc_title),
                        Paragraph(str(pg_num), style_toc_page),
                    ]
                )

            toc_table = Table(toc_rows, colWidths=[1.3 * inch, 4.15 * inch, 0.75 * inch])
            toc_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.4, rule_light),
                    ]
                )
            )
            story.append(toc_table)
            story.append(PageBreak())

        # 3. Chapter Pages
        for idx, c_num in enumerate(sorted_nums):
            if idx > 0:
                story.append(PageBreak())

            c_info = chapters[c_num]
            c_title = (c_info.get("title") or f"Chapter {c_num:02d}").strip()
            passages = c_info.get("passages", [])

            story.append(Spacer(1, 0.25 * inch))
            eyebrow_para = Paragraph(f"CHAPTER {c_num:02d}", style_chap_eyebrow)
            eyebrow_para._chapter_num = c_num
            story.append(eyebrow_para)
            story.append(Paragraph(html.escape(c_title), style_chap_title))
            story.append(HRFlowable(width="24%", thickness=0.6, color=rule_light, spaceBefore=0, spaceAfter=20))

            if not passages:
                story.append(Paragraph("<i>No passages written in this chapter yet.</i>", style_logline))
                continue

            for p_idx, entry in enumerate(passages):
                prose = (entry.get("prose") or "").strip()
                if not prose:
                    continue

                sub_paragraphs = [seg.strip() for seg in re.split(r"\n\s*\n|\n", prose) if seg.strip()]
                for s_idx, para_text in enumerate(sub_paragraphs):
                    p_style = style_body_first if s_idx == 0 else style_body_indent
                    story.append(Paragraph(html.escape(para_text), p_style))

                if p_idx < len(passages) - 1:
                    story.append(Paragraph("* &nbsp; * &nbsp; *", style_scene_sep))

        return story

    # Pass 1: Compute exact starting page numbers for every chapter
    pass1_buf = io.BytesIO()
    doc1 = _StoryBookDocTemplate(
        pass1_buf,
        pagesize=letter,
        topMargin=1.0 * inch,
        bottomMargin=0.95 * inch,
        leftMargin=1.15 * inch,
        rightMargin=1.15 * inch,
    )
    doc1.build(_build_story_elements({}), onFirstPage=_draw_title_page, onLaterPages=_draw_content_page)

    # Pass 2: Build final PDF with exact page numbers in the Contents / Index page
    final_buf = io.BytesIO()
    doc2 = _StoryBookDocTemplate(
        final_buf,
        pagesize=letter,
        topMargin=1.0 * inch,
        bottomMargin=0.95 * inch,
        leftMargin=1.15 * inch,
        rightMargin=1.15 * inch,
        title=title_clean,
    )
    doc2.build(
        _build_story_elements(doc1.chapter_page_map),
        onFirstPage=_draw_title_page,
        onLaterPages=_draw_content_page,
    )
    return final_buf.getvalue()