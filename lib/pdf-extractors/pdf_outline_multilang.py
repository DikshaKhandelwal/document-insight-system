import fitz
import re

class PDFOutlineMultiLangExtractor:
    def __init__(self):
        pass

    def extract_outline(self, pdf_path, section_limit=20):
        import fitz
        doc = fitz.open(pdf_path)
        title, outline = self.extract_multilang_outline(doc, section_limit=section_limit)
        doc.close()
        return {"title": title, "outline": outline}

    def extract_multilang_outline(self, doc, section_limit=20):
        """
        Generic: Extract main title and headings for non-English PDFs using font size, boldness, and position.
        Returns (title, outline)
        """
        # --- Improved Title Extraction ---
        page = doc[0]
        blocks = page.get_text("dict")['blocks']
        font_sizes = {}
        bold_sizes = {}
        max_size = 0
        for block in blocks:
            if block.get('type') != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    font = span.get('font', '').lower()
                    flags = span.get('flags', 0)
                    bold_indicators = ['bold', 'black', 'heavy', 'demi', 'semi']
                    font_is_bold = any(indicator in font for indicator in bold_indicators)
                    flags_bold = bool(flags & 2**4)
                    font_sizes[size] = font_sizes.get(size, 0) + 1
                    if font_is_bold or flags_bold:
                        bold_sizes[size] = bold_sizes.get(size, 0) + 1
                    if font_is_bold or flags_bold:
                        if size > max_size:
                            max_size = size
        # Title: select largest text (regardless of boldness) in top 40% of first page
        largest_size = 0
        for block in blocks:
            if block.get('type') != 0:
                continue
            for line in block.get("lines", []):
                if line["bbox"][1] > page.rect.height * 0.4:
                    continue
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    if size > largest_size:
                        largest_size = size
        # Collect all consecutive lines with largest_size
        title_lines = []
        for block in blocks:
            if block.get('type') != 0:
                continue
            for line in block.get("lines", []):
                if line["bbox"][1] > page.rect.height * 0.4:
                    continue
                line_text = ""
                line_has_largest = False
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], 1)
                    if text and size == largest_size:
                        line_text += text + " "
                        line_has_largest = True
                line_text = line_text.strip()
                if line_text and line_has_largest:
                    title_lines.append(line_text)
        title_lines = [t for t in title_lines if t]
        if title_lines:
            title = ' '.join(title_lines)
            title = re.sub(r'\s+', ' ', title).strip()
        else:
            title = ""

        # --- Improved Heading Extraction ---
        # Analyze font sizes for adaptive thresholds
        all_sizes = sorted(font_sizes.keys(), reverse=True)
        heading_sizes = all_sizes[:4] if len(all_sizes) >= 4 else all_sizes
        body_size = all_sizes[-1] if all_sizes else max_size - 2

        import difflib
        # Regex to clean TOC dot leaders and page numbers at end
        toc_cleanup_re = re.compile(r'[\s　]*[.．・…‥⋯]+[\s　]*\d{1,3}$')
        # Regex for section numbering (e.g., 1.2, 2.3.4, 1.2.3.4, 1-2, 1:2, etc.)
        section_num_pattern = re.compile(r'^(\d+[.．:：-])+(\d+)?')
        # Map: cleaned heading text -> (level, page)
        heading_map = {}
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")['blocks']
            for block in blocks:
                if block.get('type') != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = ""
                    line_spans = []
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        size = round(span["size"], 1)
                        font = span.get('font', '').lower()
                        flags = span.get('flags', 0)
                        bold_indicators = ['bold', 'black', 'heavy', 'demi', 'semi']
                        font_is_bold = any(indicator in font for indicator in bold_indicators)
                        flags_bold = bool(flags & 2**4)
                        if text:
                            line_text += text + " "
                            line_spans.append(span)
                    line_text = line_text.strip()
                    if not line_text or len(line_text) < 3:
                        continue
                    word_count = len(line_text.split())
                    if word_count < 1 or word_count > 12:
                        continue
                    # Exclude lines with only numbers or punctuation
                    if re.match(r'^[^\w]*$', line_text):
                        continue
                    # Exclude headings that look like quotes
                    if line_text.startswith('"') or line_text.startswith('“') or line_text.endswith('"') or line_text.endswith('”'):
                        continue
                    # Exclude headings that end with a hyphen, comma, or are likely fragments
                    if line_text.endswith('-') or line_text.endswith(','):
                        continue
                    # Exclude all-uppercase headings (likely table headers)
                    if line_text.isupper():
                        continue
                    # Require headings to start with a capital letter
                    if not line_text[0].isupper():
                        continue
                    # Clean TOC dot leaders and page numbers
                    cleaned_text = toc_cleanup_re.sub('', line_text).strip()
                    # Section numbering detection
                    heading_level = None
                    m = section_num_pattern.match(cleaned_text)
                    if m:
                        # Count number of section levels (number of dots, dashes, colons, etc.)
                        num_separators = len(re.findall(r'[.．:：-]', cleaned_text))
                        if num_separators == 0:
                            heading_level = "H1"
                        elif num_separators == 1:
                            heading_level = "H2"
                        elif num_separators == 2:
                            heading_level = "H3"
                        else:
                            heading_level = "H4"
                    else:
                        # Fallback to font size and boldness
                        main_span = max(line_spans, key=lambda s: len(s["text"])) if line_spans else None
                        size = round(main_span["size"], 1) if main_span else body_size
                        is_bold = False
                        if main_span:
                            font = main_span.get('font', '').lower()
                            flags = main_span.get('flags', 0)
                            bold_indicators = ['bold', 'black', 'heavy', 'demi', 'semi']
                            font_is_bold = any(indicator in font for indicator in bold_indicators)
                            flags_bold = bool(flags & 2**4)
                            is_bold = font_is_bold or flags_bold
                        if size == heading_sizes[0]:
                            heading_level = "H1"
                        elif len(heading_sizes) > 1 and size == heading_sizes[1]:
                            heading_level = "H2"
                        elif len(heading_sizes) > 2 and size == heading_sizes[2]:
                            heading_level = "H3"
                        elif len(heading_sizes) > 3 and size == heading_sizes[3]:
                            heading_level = "H4"
                        # Boldness boost for H2/H3
                        if is_bold and not heading_level and size >= body_size + 1:
                            heading_level = "H2"
                        # Indentation-based H4
                        if not heading_level and line["bbox"][0] > page.rect.width * 0.25 and size <= body_size + 1:
                            heading_level = "H4"
                    # Only accept if heading_level assigned
                    if heading_level:
                        # Stricter filtering for H4 headings
                        if heading_level == "H4":
                            # Require at least 3 words, exclude quotes/fragments
                            if word_count < 3:
                                continue
                            if line_text.startswith('"') or line_text.startswith('“') or line_text.endswith('"') or line_text.endswith('”'):
                                continue
                            if line_text.endswith('-') or line_text.endswith(','):
                                continue
                        # Deduplicate: keep only the highest page number for each heading
                        if cleaned_text:
                            prev = heading_map.get(cleaned_text)
                            if not prev or page_num > prev[1]:
                                heading_map[cleaned_text] = (heading_level, page_num)
        # Build outline from deduped headings, sorted by page and section order
        outline = [
            {"level": level, "text": text, "page": page}
            for text, (level, page) in heading_map.items()
        ]
        outline.sort(key=lambda h: (h["page"], h["text"]))
        # Final selection: prioritize H1/H2, limit to section_limit
        priority_sections = [h for h in outline if h["level"] in ["H1", "H2"]]
        other_sections = [h for h in outline if h["level"] not in ["H1", "H2"]]
        final_outline = priority_sections
        for heading in other_sections:
            if len(final_outline) < section_limit:
                final_outline.append(heading)
            else:
                break
        return title, final_outline
