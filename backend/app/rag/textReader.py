from __future__ import annotations
from pathlib import Path
from typing import Union
import pypdf


class PDFReader:

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"No file found at {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError("Provided file must be a .pdf document.")

    def extract_text(self) -> str:
        text = ""

        with open(self.file_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n{page_text}"

        return text

class MarkdownReader:

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"No file found at {self.file_path}")
        if self.file_path.suffix.lower() != ".md":
            raise ValueError("Provided file must be a .md document.")

    def extract_text(self) -> str:
        return self.file_path.read_text(encoding="utf-8")
