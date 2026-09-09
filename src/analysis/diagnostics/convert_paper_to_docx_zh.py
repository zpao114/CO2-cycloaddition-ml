"""
Markdown to Word converter for Chinese paper submission.

与 convert_paper_to_docx.py 关系：
- 结构相同（三线表、上下标、图片、参考文献）
- 关键差异：通过 OOXML eastAsia 设置中文字体（宋体/黑体）
- 西文（数字、英文名、化学式）仍保留 Times New Roman
- 标题用黑体，正文/表格/图注用宋体

字体策略（必须通过 w:rFonts 的 eastAsia 设置，font.name 不支持中文）：
- 正文 run:ascii='Times New Roman',  eastAsia='宋体'
- 标题 run:ascii='Times New Roman',  eastAsia='黑体'
"""

import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# Unicode 上标/下标映射
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

# 字体常量
FONT_CN_BODY = '宋体'    # 正文/表格/参考文献
FONT_CN_HEADING = '黑体'  # 标题
FONT_EN = 'Times New Roman'


def _set_run_font(run, size_pt, bold=False, italic=False, heading=False):
    """设置 run 字体：西文用 Times New Roman，中文用宋体/黑体。"""
    rPr = run._element.get_or_add_rPr()
    # 清除已存在的 rFonts
    existing = rPr.find(qn('w:rFonts'))
    if existing is not None:
        rPr.remove(existing)
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), FONT_EN)
    rFonts.set(qn('w:hAnsi'), FONT_EN)
    rFonts.set(qn('w:eastAsia'), FONT_CN_HEADING if heading else FONT_CN_BODY)
    rFonts.set(qn('w:cs'), FONT_EN)
    rPr.append(rFonts)

    run.font.size = Pt(size_pt)
    if bold:
        run.bold = True
    if italic:
        run.italic = True


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """设置表格单元格边框。None=保持默认，''=无边框，dict=自定义。"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    else:
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
    """三线表样式：第一行上下边框粗，其他行只有底线（最后一行加粗底线）。"""
    n_rows = len(table.rows)
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            left = right = ''
            top = bottom = ''
            if r_idx == 0:
                top = {'sz': 12}
                bottom = {'sz': 6}
            if r_idx == n_rows - 1:
                bottom = {'sz': 12}
            set_cell_borders(cell, top=top, bottom=bottom, left=left, right=right)


def _emit_runs_with_sub(para, text, base_size, fmt_type, heading=False):
    """发出 run 段，处理 Unicode 上下标。"""
    if not text:
        return
    segments = []
    buf = []
    cur_kind = None

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

    bold = (fmt_type == 'bold')
    italic = (fmt_type == 'italic')

    for seg_text, kind in segments:
        if not seg_text:
            continue
        run = para.add_run()
        _set_run_font(run, base_size, bold=bold, italic=italic, heading=heading)
        if kind == 'sup':
            run.font.superscript = True
            run.text = ''.join(SUPERSCRIPT_MAP.get(c, c) for c in seg_text)
        elif kind == 'sub':
            run.font.subscript = True
            run.text = ''.join(SUBSCRIPT_MAP.get(c, c) for c in seg_text)
        else:
            run.text = seg_text


def add_runs_with_formatting(para, text, base_size=12, bold=False, italic=False, heading=False):
    """解析内联 markdown 格式（**bold**, *italic*），分别处理并发出 run。"""
    if not text:
        return
    pattern = re.compile(
        r'(\*\*[^*]+?\*\*)'
        r'|(\*[^*\n]+?\*)'
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

    base_bold = bold
    base_italic = italic
    for fmt_type, content in parts:
        if not content:
            continue
        if fmt_type == 'bold':
            _emit_runs_with_sub(para, content, base_size, 'bold', heading=heading)
        elif fmt_type == 'italic':
            _emit_runs_with_sub(para, content, base_size, 'italic', heading=heading)
        else:
            _emit_runs_with_sub(para, content, base_size, 'text', heading=heading)


def add_para(doc, text='', bold=False, italic=False, align='left', size=12,
             space_before=0, space_after=6, line_spacing=1.5, first_line_indent=None,
             left_indent=None, hanging_indent=None, keep_with_next=False,
             heading=False):
    """添加段落（中文版默认 1.5 倍行距，符合中文期刊习惯）。"""
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
        add_runs_with_formatting(para, text, base_size=size, bold=bold, italic=italic, heading=heading)

    return para


def add_heading(doc, text, level):
    """添加标题，黑体（中文）+ Times New Roman（西文）。"""
    configs = {
        # 中文期刊常用字号：二号=22pt, 三号=16pt, 四号=14pt, 小四=12pt
        1: {'size': 22, 'bold': True, 'center': True,  'space_before': 24, 'space_after': 18},
        2: {'size': 16, 'bold': True, 'center': False, 'space_before': 18, 'space_after': 12},
        3: {'size': 14, 'bold': True, 'center': False, 'space_before': 12, 'space_after': 8},
        4: {'size': 12, 'bold': True, 'center': False, 'space_before': 8,  'space_after': 4},
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
    add_runs_with_formatting(para, text, base_size=c['size'], bold=c['bold'], heading=True)
    return para


def add_horizontal_rule(doc):
    """添加水平分隔线。"""
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
    """添加三线表。"""
    if not header or not rows:
        return False

    n_cols = len(header)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.autofit = True
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblLayout = OxmlElement('w:tblLayout')
    tblLayout.set(qn('w:type'), 'autofit')
    tblPr.append(tblLayout)

    # 表头行
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

    # 数据行
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = ''
            para = cell.paragraphs[0]
            if alignments and ci < len(alignments) and alignments[ci] == 'center':
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif alignments and ci < len(alignments) and alignments[ci] == 'left':
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                # 默认：第一列左对齐，其他列居中（适合数字表）
                if ci == 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                else:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after = Pt(2)
            if ci == 0:
                add_runs_with_formatting(para, str(cell_text), base_size=10, bold=True)
            else:
                add_runs_with_formatting(para, str(cell_text), base_size=10)

    # 应用三线表样式
    set_table_three_lines(table)
    add_para(doc, '', space_before=4, space_after=4)
    return True


def add_image(doc, img_path, caption='', figure_num=None):
    """添加图片及图注。"""
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
        if figure_num is not None:
            label_run = cap_para.add_run(f"Figure {figure_num}. ")
            label_run.bold = True
            _set_run_font(label_run, 10, bold=True)
            add_runs_with_formatting(cap_para, caption, base_size=10)
        else:
            add_runs_with_formatting(cap_para, caption, base_size=10)


def parse_table_alignments(header):
    """解析表头中的对齐标记（:---, :---:, ---:）。无标记默认居中（数字）。"""
    alignments = []
    for h in header:
        h_strip = h.strip()
        if h_strip.startswith(':') and h_strip.endswith(':'):
            alignments.append('center')
        elif h_strip.endswith(':'):
            alignments.append('right')
        elif h_strip.startswith(':'):
            alignments.append('left')
        else:
            alignments.append('center')
    return alignments


def convert(md_path, docx_path, include_images=True):
    """把 markdown 文件转换为 docx。"""
    print(f"Reading: {md_path}")

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    doc = Document()

    # 页面设置（A4，符合中文期刊习惯；比 letter 略高）
    section = doc.sections[0]
    section.page_width = Inches(8.27)   # A4 宽
    section.page_height = Inches(11.69) # A4 高
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    # 默认 Normal 样式（小四 = 12pt，宋体/Times New Roman）
    style = doc.styles['Normal']
    style.font.size = Pt(12)
    rPr = style.element.get_or_add_rPr()
    existing = rPr.find(qn('w:rFonts'))
    if existing is not None:
        rPr.remove(existing)
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), FONT_EN)
    rFonts.set(qn('w:hAnsi'), FONT_EN)
    rFonts.set(qn('w:eastAsia'), FONT_CN_BODY)
    rFonts.set(qn('w:cs'), FONT_EN)
    rPr.append(rFonts)

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
        # 去除 Read 工具残留的行号标记（如 "  60|"）
        cleaned = re.sub(r'^\s*\d+\|\s?', '', line)
        stripped = cleaned.strip()
        original_stripped = line.strip()

        # 空行 - 跳过
        if not stripped:
            i += 1
            continue

        # 水平分隔线
        if stripped == '---' or stripped == '***':
            add_horizontal_rule(doc)
            prev_was_heading = False
            i += 1
            continue

        # 标题
        if stripped.startswith('# '):
            add_heading(doc, stripped[2:], 1)
            prev_was_heading = True
            i += 1
            continue
        elif stripped.startswith('## '):
            heading_text = stripped[3:]
            if heading_text in ('References', '参考文献'):
                add_heading(doc, '参考文献', 2)
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

        # 表格
        if stripped.startswith('|'):
            header = [c.strip() for c in stripped.strip('|').split('|')]
            i += 1
            def _clean_line(j):
                return re.sub(r'^\s*\d+\|\s?', '', lines[j]).strip()
            if i < len(lines) and re.match(r'^\|[\s\-\|:]+\|$', _clean_line(i)):
                i += 1
            rows = []
            while i < len(lines) and _clean_line(i).startswith('|'):
                row = [c.strip() for c in _clean_line(i).strip('|').split('|')]
                rows.append(row)
                i += 1
            alignments = parse_table_alignments(header)
            if add_three_line_table(doc, header, rows, alignments):
                tbl_count += 1
                print(f"  Table {tbl_count}: {len(header)} cols x {len(rows)} rows")
            prev_was_heading = False
            continue

        # 图片
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

        # 独立图注段（**Figure N.** ...）
        if stripped.startswith('**Figure'):
            fig_match = re.match(r'\*\*Figure (\d+)\.\*\*\s*(.*)', stripped)
            if fig_match:
                num, caption = fig_match.groups()
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                para.paragraph_format.space_before = Pt(2)
                para.paragraph_format.space_after = Pt(8)
                para.paragraph_format.left_indent = Inches(0.2)
                para.paragraph_format.right_indent = Inches(0.2)
                label_run = para.add_run(f"Figure {num}. ")
                _set_run_font(label_run, 10, bold=True)
                add_runs_with_formatting(para, caption, base_size=10)
            else:
                add_para(doc, stripped.replace('**', ''), bold=True, align='center',
                         size=10, space_before=6, space_after=3)
            prev_was_heading = False
            i += 1
            continue

        # 关键词行（中英文都可以）
        if stripped.startswith('**Keywords') or stripped.startswith('**关键词'):
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(12)
            add_runs_with_formatting(para, stripped, base_size=11)
            prev_was_heading = False
            i += 1
            continue

        # 参考文献条目
        if re.match(r'^\(\d+\)', stripped):
            reference_count += 1
            add_para(
                doc,
                text=stripped,
                align='justify',
                size=10,
                space_before=0,
                space_after=4,
                line_spacing=1.3,
                hanging_indent=0.3,
            )
            prev_was_heading = False
            i += 1
            continue

        # 普通段落
        text = stripped
        space_before = 6 if prev_was_heading else 0
        # 中文段落首行缩进 2 个字符（约 0.32 英寸）
        first_indent = 0.32 if not prev_was_heading else None
        add_para(
            doc,
            text=text,
            align='justify',
            size=12,           # 小四
            space_before=space_before,
            space_after=6,
            line_spacing=1.5,  # 1.5 倍行距（中文期刊标准）
            first_line_indent=first_indent,
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
    md_file = Path("D:/machine-learning/CO2-cycloaddition/docs/paper/paper_final_Chinese.md")
    docx_file = Path("D:/machine-learning/CO2-cycloaddition/docs/paper/paper_for_中文期刊_submission.docx")

    fig_count, tbl_count = convert(md_file, docx_file, include_images=True)

    print(f"\n=== Summary ===")
    print(f"Figures: {fig_count}")
    print(f"Tables: {tbl_count}")
    print(f"Output: {docx_file}")
