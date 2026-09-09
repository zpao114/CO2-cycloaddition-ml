"""
Markdown to Word converter for JCIM submission.

Features:
- Three-line table style (JCIM/ACS standard)
- Reference numbering preserved with hanging indent
- Inline markdown formatting (bold, italic) preserved
- Unicode superscript/subscript properly converted
- Proper paragraph spacing
"""

import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement


# Unicode super/subscript translation tables
SUPERSCRIPT_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁽': '(', '⁾': ')', 'ⁿ': 'n',
}
SUBSCRIPT_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₍': '(', '₎': ')',
}


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set borders for a table cell. None=keep default, ''=no border, dict=custom."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    else:
        # Clear existing borders
        for child in list(tcBorders):
            tcBorders.remove(child)

    for edge, val in (('top', top), ('left', left), ('bottom', bottom), ('right', right)):
        if val is None:
            continue
        b = OxmlElement(f'w:{edge}')
        if val == '':
            b.set(qn('w:val'), 'nil')
        else:
            b.set(qn('w:val'), 'single')
            b.set(qn('w:sz'), str(val.get('sz', 6)))
            b.set(qn('w:space'), '0')
            b.set(qn('w:color'), val.get('color', '000000'))
        tcBorders.append(b)


def set_table_three_lines(table):
    """Apply three-line style: top of row 0, bottom of header, bottom of last row."""
    n_rows = len(table.rows)
    n_cols = len(table.columns)
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            # Default: no left/right borders
            left = right = ''
            top = bottom = ''
            # Top border: only on the first row
            if r_idx == 0:
                top = {'sz': 12}  # 1.5pt thick
            # Bottom border of header (first row)
            if r_idx == 0:
                bottom = {'sz': 6}  # 0.75pt thin
            # Bottom border of last row
            if r_idx == n_rows - 1:
                bottom = {'sz': 12}  # 1.5pt thick
            set_cell_borders(cell, top=top, bottom=bottom, left=left, right=right)


def add_runs_with_formatting(para, text, base_size=12, bold=False, italic=False):
    """
    Parse inline markdown formatting (**bold**, *italic*) and add runs to paragraph.
    Also handles Unicode superscript/subscript characters.
    """
    if not text:
        return

    # First, tokenize the text into segments with formatting markers
    # Pattern matches: **bold**, *italic*, Unicode super/sub sequences
    # We need to handle these simultaneously
    pattern = re.compile(
        r'(\*\*[^*]+?\*\*)'           # **bold**
        r'|(\*[^*\n]+?\*)'              # *italic*
    )

    parts = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(('text', text[last:m.start()]))
        chunk = m.group(0)
        if chunk.startswith('**') and chunk.endswith('**'):
            parts.append(('bold', chunk[2:-2]))
        elif chunk.startswith('*') and chunk.endswith('*'):
            parts.append(('italic', chunk[1:-1]))
        last = m.end()
    if last < len(text):
        parts.append(('text', text[last:]))

    # Now process each part, splitting on Unicode super/sub
    for fmt_type, content in parts:
        _emit_runs_with_sub(para, content, base_size, fmt_type)


def _emit_runs_with_sub(para, text, base_size, fmt_type):
    """Emit runs handling Unicode superscript and subscript characters."""
    if not text:
        return
    # Build segments of (text, super/sub flag)
    segments = []
    buf = []
    cur_kind = None  # None=normal, 'sup', 'sub'

    def flush():
        nonlocal buf, cur_kind
        if buf:
            segments.append((''.join(buf), cur_kind))
        buf = []
        cur_kind = None

    for ch in text:
        if ch in SUPERSCRIPT_MAP:
            if cur_kind != 'sup':
                flush()
                cur_kind = 'sup'
            buf.append(ch)
        elif ch in SUBSCRIPT_MAP:
            if cur_kind != 'sub':
                flush()
                cur_kind = 'sub'
            buf.append(ch)
        else:
            if cur_kind is not None:
                flush()
            buf.append(ch)
    flush()

    for seg_text, kind in segments:
        if not seg_text:
            continue
        run = para.add_run()
        run.font.name = 'Times New Roman'
        run.font.size = Pt(base_size)
        if fmt_type == 'bold':
            run.bold = True
        elif fmt_type == 'italic':
            run.italic = True
        if kind == 'sup':
            run.font.superscript = True
            # Convert Unicode super to ASCII
            run.text = ''.join(SUPERSCRIPT_MAP.get(c, c) for c in seg_text)
        elif kind == 'sub':
            run.font.subscript = True
            run.text = ''.join(SUBSCRIPT_MAP.get(c, c) for c in seg_text)
        else:
            run.text = seg_text


def add_para(doc, text='', bold=False, italic=False, align='left', size=12,
             space_before=0, space_after=6, line_spacing=1.15, first_line_indent=None,
             left_indent=None, hanging_indent=None, keep_with_next=False):
    """Add paragraph with JCIM formatting. Supports inline markdown formatting."""
    para = doc.add_paragraph()

    if align == 'center':
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == 'right':
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == 'justify':
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    para.paragraph_format.space_before = Pt(space_before)
    para.paragraph_format.space_after = Pt(space_after)
    if line_spacing:
        para.paragraph_format.line_spacing = line_spacing
    if first_line_indent is not None:
        para.paragraph_format.first_line_indent = Inches(first_line_indent)
    if left_indent is not None:
        para.paragraph_format.left_indent = Inches(left_indent)
    if hanging_indent is not None:
        para.paragraph_format.left_indent = Inches(hanging_indent)
        para.paragraph_format.first_line_indent = Inches(-hanging_indent)
    if keep_with_next:
        para.paragraph_format.keep_with_next = True

    if text:
        add_runs_with_formatting(para, text, base_size=size, bold=bold, italic=italic)

    return para


def add_heading(doc, text, level):
    """Add heading with JCIM formatting."""
    configs = {
        1: {'size': 14, 'bold': True, 'center': True, 'space_before': 18, 'space_after': 12, 'all_caps': True},
        2: {'size': 13, 'bold': True, 'center': False, 'space_before': 14, 'space_after': 8},
        3: {'size': 12, 'bold': True, 'center': False, 'space_before': 8, 'space_after': 4},
        4: {'size': 11, 'bold': True, 'center': False, 'space_before': 6, 'space_after': 3},
    }
    c = configs.get(level, configs[4])
    para = doc.add_paragraph()
    if c['center']:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.space_before = Pt(c['space_before'])
    para.paragraph_format.space_after = Pt(c['space_after'])
    para.paragraph_format.keep_with_next = True

    text_display = text
    if c.get('all_caps'):
        text_display = text.upper()
    run = para.add_run(text_display)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(c['size'])
    run.bold = c['bold']
    return para


def add_horizontal_rule(doc):
    """Add horizontal line."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(6)
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '000000')
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_three_line_table(doc, header, rows, alignments=None):
    """Add a JCIM/ACS-style three-line table."""
    if not header or not rows:
        return False

    n_cols = len(header)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.autofit = True
    # Set table to fixed layout for predictable column widths
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblLayout = OxmlElement('w:tblLayout')
    tblLayout.set(qn('w:type'), 'autofit')
    tblPr.append(tblLayout)

    # Header row
    hdr = table.rows[0]
    for j, cell_text in enumerate(header):
        cell = hdr.cells[j]
        cell.text = ''
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(4)
        para.paragraph_format.keep_with_next = True
        add_runs_with_formatting(para, str(cell_text), base_size=10, bold=True)

    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = ''
            para = cell.paragraphs[0]
            if alignments and ci < len(alignments) and alignments[ci] == 'left':
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif alignments and ci < len(alignments) and alignments[ci] == 'center':
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after = Pt(2)
            # First column: bold (substrate names)
            if ci == 0:
                add_runs_with_formatting(para, str(cell_text), base_size=10, bold=True)
            else:
                add_runs_with_formatting(para, str(cell_text), base_size=10)

    # Apply three-line style
    set_table_three_lines(table)

    # Add spacing after table
    add_para(doc, '', space_before=4, space_after=4)
    return True


def add_image(doc, img_path, caption='', figure_num=None):
    """Add image with caption."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(3)
    para.paragraph_format.keep_with_next = True

    run = para.add_run()
    run.add_picture(str(img_path), width=Inches(5.5))

    if caption:
        cap_para = doc.add_paragraph()
        cap_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cap_para.paragraph_format.space_before = Pt(2)
        cap_para.paragraph_format.space_after = Pt(8)
        cap_para.paragraph_format.left_indent = Inches(0.2)
        cap_para.paragraph_format.right_indent = Inches(0.2)
        # Format: "Figure N. caption text"
        if figure_num is not None:
            label_run = cap_para.add_run(f"Figure {figure_num}. ")
            label_run.bold = True
            label_run.font.name = 'Times New Roman'
            label_run.font.size = Pt(10)
            add_runs_with_formatting(cap_para, caption, base_size=10)
        else:
            add_runs_with_formatting(cap_para, caption, base_size=10)


def parse_table_alignments(header):
    """Heuristic: if header has many short tokens, default to left; numeric data = left/center."""
    return ['left'] * len(header)


def convert(md_path, docx_path, include_images=True):
    """Convert markdown to Word document."""
    print(f"Reading: {md_path}")

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    doc = Document()

    # Page setup - JCIM standard letter
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)

    # Default style
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)

    # Project root for image paths
    project_root = Path("D:/machine-learning/CO2-cycloaddition")
    md_path_obj = Path(md_path)

    lines = content.split('\n')
    print(f"Total lines: {len(lines)}")

    i = 0
    fig_count = 0
    tbl_count = 0
    prev_was_heading = False
    in_references = False
    reference_count = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line - skip
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped == '---' or stripped == '***':
            add_horizontal_rule(doc)
            prev_was_heading = False
            i += 1
            continue

        # Headings
        if stripped.startswith('# '):
            add_heading(doc, stripped[2:], 1)
            prev_was_heading = True
            i += 1
            continue
        elif stripped.startswith('## '):
            heading_text = stripped[3:]
            if heading_text in ('References', '参考文献'):
                add_heading(doc, 'References', 2)
                in_references = True
            else:
                add_heading(doc, heading_text, 2)
                in_references = False
            prev_was_heading = True
            i += 1
            continue
        elif stripped.startswith('### '):
            add_heading(doc, stripped[4:], 3)
            prev_was_heading = True
            i += 1
            continue
        elif stripped.startswith('#### '):
            add_heading(doc, stripped[5:], 4)
            prev_was_heading = True
            i += 1
            continue

        # Table
        if stripped.startswith('|'):
            header = [c.strip() for c in stripped.strip('|').split('|')]
            i += 1
            # Skip separator
            if i < len(lines) and re.match(r'^\|[\s\-\|:]+\|$', lines[i].strip()):
                i += 1
            # Get rows
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(row)
                i += 1
            alignments = parse_table_alignments(header)
            if add_three_line_table(doc, header, rows, alignments):
                tbl_count += 1
                print(f"  Table {tbl_count}: {len(header)} cols x {len(rows)} rows")
            prev_was_heading = False
            continue

        # Image
        if stripped.startswith('![') and include_images:
            img_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
            if img_match:
                caption, path = img_match.groups()
                img_path = project_root / path
                if img_path.exists():
                    fig_count += 1
                    add_image(doc, img_path, caption, figure_num=fig_count)
                    print(f"  Image {fig_count}: {path}")
                else:
                    add_para(doc, f"[Image not found: {path}]", italic=True, size=10)
            i += 1
            prev_was_heading = False
            continue

        # Standalone Figure caption (for figures referenced as **Figure N.** in text)
        if stripped.startswith('**Figure'):
            fig_match = re.match(r'\*\*Figure (\d+)\.\*\*\s*(.*)', stripped)
            if fig_match:
                num, caption = fig_match.groups()
                # Caption paragraph: bold "Figure N." + normal caption text
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                para.paragraph_format.space_before = Pt(2)
                para.paragraph_format.space_after = Pt(8)
                para.paragraph_format.left_indent = Inches(0.2)
                para.paragraph_format.right_indent = Inches(0.2)
                label_run = para.add_run(f"Figure {num}. ")
                label_run.bold = True
                label_run.font.name = 'Times New Roman'
                label_run.font.size = Pt(10)
                add_runs_with_formatting(para, caption, base_size=10)
            else:
                add_para(doc, stripped.replace('**', ''), bold=True, align='center', size=10, space_before=6, space_after=3)
            prev_was_heading = False
            i += 1
            continue

        # Keywords line
        if stripped.startswith('**Keywords'):
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(12)
            add_runs_with_formatting(para, stripped, base_size=11, italic=True)
            prev_was_heading = False
            i += 1
            continue

        # Reference items - keep the number, format with italic/bold
        if re.match(r'^\(\d+\)', stripped):
            reference_count += 1
            # Add hanging indent: 0.3 inch
            add_para(
                doc,
                text=stripped,  # KEEP the number
                align='justify',
                size=10,
                space_before=0,
                space_after=4,
                line_spacing=1.15,
                hanging_indent=0.3,
            )
            prev_was_heading = False
            i += 1
            continue

        # Regular paragraph - preserve inline markdown formatting
        text = stripped
        space_before = 6 if prev_was_heading else 0
        add_para(
            doc,
            text=text,
            align='justify',
            size=12,
            space_before=space_before,
            space_after=6,
            line_spacing=1.15,
        )
        prev_was_heading = False
        i += 1

        if i % 50 == 0:
            print(f"Progress: {i}/{len(lines)}")

    print(f"Saving: {docx_path}")
    doc.save(str(docx_path))
    print("Done!")
    return fig_count, tbl_count


if __name__ == "__main__":
    md_file = Path("D:/machine-learning/CO2-cycloaddition/docs/paper/paper_final_English.md")
    docx_file = Path("D:/machine-learning/CO2-cycloaddition/docs/paper/paper_for_JCIM_submission.docx")

    fig_count, tbl_count = convert(md_file, docx_file, include_images=True)

    print(f"\n=== Summary ===")
    print(f"Figures: {fig_count}")
    print(f"Tables: {tbl_count}")
    print(f"Output: {docx_file}")
