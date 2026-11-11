"""Load documents (PDF, DOCX, XLSX, CSV, JSON, TXT, MD) and split into overlapping chunks."""
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pypdf import PdfReader


def extract_docx(data: bytes) -> str:
    """Extract paragraphs from Word .docx file using built-in zip & xml parsing."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            tree = ET.fromstring(z.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paras = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    paras.append("".join(texts))
            return "\n\n".join(paras)
    except Exception as e:
        return f"Error reading DOCX: {e}"


def extract_xlsx(data: bytes) -> str:
    """Extract tabular text from an Excel .xlsx file using built-in zipfile and XML."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            shared_strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in tree.iterfind(".//main:si", ns):
                    t = si.find("main:t", ns)
                    shared_strings.append(t.text if t is not None and t.text else "")

            sheet_names = [
                n
                for n in z.namelist()
                if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            ]
            if not sheet_names:
                return "Empty spreadsheet."

            tree = ET.fromstring(z.read(sheet_names[0]))
            ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            rows = []
            for row in tree.iterfind(".//main:row", ns):
                cells = []
                for c in row.iterfind("main:c", ns):
                    t_attr = c.get("t")
                    v = c.find("main:v", ns)
                    val = v.text if v is not None and v.text else ""
                    if t_attr == "s" and val.isdigit():
                        idx = int(val)
                        val = shared_strings[idx] if idx < len(shared_strings) else val
                    cells.append(val.strip())
                if any(cells):
                    rows.append(cells)

            if not rows:
                return "No data found in spreadsheet."

            header = rows[0]
            lines = []
            for i, r in enumerate(rows[1:], 1):
                fields = [
                    f"{header[j] if j < len(header) else f'Col{j+1}'}: {val}"
                    for j, val in enumerate(r)
                    if val
                ]
                if fields:
                    lines.append(f"Row {i}: " + ", ".join(fields))
            return "\n".join(lines) if lines else "\n".join([", ".join(r) for r in rows])
    except Exception as e:
        return f"Error reading XLSX: {e}"


def extract_csv(data: bytes) -> str:
    """Format CSV tabular data into structured row records for semantic search."""
    try:
        text = data.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return ""
        header = rows[0]
        lines = []
        for i, row in enumerate(rows[1:], 1):
            fields = [
                f"{header[j] if j < len(header) else f'Col{j+1}'}: {val.strip()}"
                for j, val in enumerate(row)
                if val.strip()
            ]
            if fields:
                lines.append(f"Row {i}: " + ", ".join(fields))
        return "\n".join(lines) if lines else text
    except Exception:
        return data.decode("utf-8", errors="ignore")


def extract_json(data: bytes) -> str:
    """Format JSON content nicely."""
    try:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        return json.dumps(obj, indent=2)
    except Exception:
        return data.decode("utf-8", errors="ignore")


def load_bytes(name: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for PDF, DOCX, XLSX, CSV, JSON, TXT or MD files."""
    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        except Exception as e:
            return [(1, f"Error reading PDF {name}: {e}")]
    elif ext == ".docx":
        return [(1, extract_docx(data))]
    elif ext == ".xlsx":
        return [(1, extract_xlsx(data))]
    elif ext == ".csv":
        return [(1, extract_csv(data))]
    elif ext == ".json":
        return [(1, extract_json(data))]
    else:
        return [(1, data.decode("utf-8", errors="ignore"))]


def load_path(path: Path) -> list[tuple[int, str]]:
    return load_bytes(path.name, path.read_bytes())


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Paragraph-aware splitter with character overlap between chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # hard-split very long paragraphs
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size - overlap:]
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            chunks.append(cur)
            cur = f"{cur[-overlap:]}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]
