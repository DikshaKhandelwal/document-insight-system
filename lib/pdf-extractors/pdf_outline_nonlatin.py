import fitz
import re
import unicodedata

class PDFOutlineNonLatinExtractor:
    def __init__(self):
        pass

    def extract_outline(self, pdf_path, section_limit=20):
        doc = fitz.open(pdf_path)
        title, outline = self.extract_nonlatin_outline(doc, section_limit=section_limit)
        doc.close()
        return {"title": title, "outline": outline}

    def extract_nonlatin_outline(self, doc, section_limit=20):
        """
        Extract main title and headings for non-Latin PDFs using font size, boldness, position, and section numbering patterns.
        Returns (title, outline)
        """
        import re
        page = doc[0]
        blocks = page.get_text("dict")['blocks']
        font_sizes = {}
        max_size = 0
        for block in blocks:
            if block.get('type') != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    font_sizes[size] = font_sizes.get(size, 0) + 1
                    if size > max_size:
                        max_size = size
        # Title: select largest text in top 40% of first page
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
        # --- Heading Extraction ---
        all_sizes = sorted(font_sizes.keys(), reverse=True)
        heading_sizes = all_sizes[:4] if len(all_sizes) >= 4 else all_sizes
        body_size = all_sizes[-1] if all_sizes else max_size - 2
        outline_candidates = []
        section_num_pattern = re.compile(r'^(\d+([．.]))(\d+([．.]))?(\d+([．.]))?')
        # Regex to clean TOC dot leaders and page numbers at end
        # Remove dot leaders (with or without trailing page numbers)
        toc_cleanup_re = re.compile(r'[.．・…‥⋯]+[ 　]*\d{0,3}$')
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
                        if text:
                            line_text += text + " "
                            line_spans.append(span)
                    line_text = line_text.strip()
                    if not line_text or len(line_text) < 2:
                        continue
                    # Non-Latin: skip Latin-specific checks, but skip lines with only punctuation/numbers
                    if all(unicodedata.category(c).startswith(('P', 'N', 'Z')) for c in line_text):
                        continue
                    # Clean TOC dot leaders and page numbers
                    cleaned_text = toc_cleanup_re.sub('', line_text).strip()
                    # Section numbering detection (Japanese/Chinese)
                    heading_level = None
                    m = section_num_pattern.match(cleaned_text)
                    if m:
                        # Count number of section levels
                        num_dots = cleaned_text.count('．') + cleaned_text.count('.')
                        if num_dots == 0:
                            heading_level = "H1"
                        elif num_dots == 1:
                            heading_level = "H2"
                        elif num_dots == 2:
                            heading_level = "H3"
                        else:
                            heading_level = "H4"
                    else:
                        # Fallback to font size
                        main_span = max(line_spans, key=lambda s: len(s["text"])) if line_spans else None
                        size = round(main_span["size"], 1) if main_span else body_size
                        if size == heading_sizes[0]:
                            heading_level = "H1"
                        elif len(heading_sizes) > 1 and size == heading_sizes[1]:
                            heading_level = "H2"
                        elif len(heading_sizes) > 2 and size == heading_sizes[2]:
                            heading_level = "H3"
                        elif len(heading_sizes) > 3 and size == heading_sizes[3]:
                            heading_level = "H4"
                    # Deduplicate: keep only the highest page number for each heading
                    if heading_level and cleaned_text:
                        if cleaned_text not in heading_map or page_num > heading_map[cleaned_text][1]:
                            heading_map[cleaned_text] = (heading_level, page_num)
        # Build outline from deduped headings, sorted by page and section order
        outline = [
            {"level": level, "text": text, "page": page}
            for text, (level, page) in heading_map.items()
        ]
        outline.sort(key=lambda h: (h["page"], h["text"]))
        # Prioritize H1/H2, limit to section_limit
        priority_sections = [h for h in outline if h["level"] in ["H1", "H2"]]
        other_sections = [h for h in outline if h["level"] not in ["H1", "H2"]]
        final_outline = priority_sections
        for heading in other_sections:
            if len(final_outline) < section_limit:
                final_outline.append(heading)
            else:
                break
        return title, final_outline
