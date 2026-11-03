"""Builds small fixture files for the tests into tests/fixtures/."""

from pathlib import Path

from openpyxl import Workbook
from reportlab.pdfgen import canvas


def _pdf(path: Path, pages: list[str | None]) -> None:
    """Write a PDF. A page whose text is None gets only a drawn rectangle (no text)."""
    pdf = canvas.Canvas(str(path))
    for text in pages:
        if text is None:
            pdf.rect(100, 100, 200, 100, fill=1)
        else:
            pdf.drawString(100, 750, text)
        pdf.showPage()
    pdf.save()


def build_fixtures(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    _pdf(out_dir / "sample.pdf", ["Welcome to Acme. Our support team is friendly.",
                                  "Refund window is 30 days."])
    _pdf(out_dir / "scanned.pdf", [None])

    (out_dir / "cafe_cp1252.csv").write_bytes("name,city\nZoë,Café\n".encode("cp1252"))
    (out_dir / "header_only.csv").write_text("name,city\n", encoding="utf-8")

    workbook = Workbook()
    sales = workbook.active
    sales.title = "Sales"
    sales.append(["region", "amount"])
    for row in [("East", 100), ("West", 200), ("North", 300)]:
        sales.append(row)
    staff = workbook.create_sheet("Staff")
    staff.append(["name", "department"])
    for row in [("Asha", "Eng"), ("Ben", "Sales"), ("Chen", "HR")]:
        staff.append(row)
    workbook.save(out_dir / "multi.xlsx")
    return out_dir


if __name__ == "__main__":
    build_fixtures(Path(__file__).parent / "fixtures")
