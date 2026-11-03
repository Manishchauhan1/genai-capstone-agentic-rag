"""Writes the four demo files used by the evaluation and the live demo.

Run:  python samples/make_samples.py
The content is fixed so the evaluation's expected answers are known.
"""

from pathlib import Path

from openpyxl import Workbook
from reportlab.pdfgen import canvas

POLICY = (
    "Customers can request a refund within 30 days of purchase. "
    "Support is available 9am to 6pm, Monday to Friday. "
    "Staff may work remotely up to 3 days a week.\n"
)

SALES = (
    "region,product,amount,units\n"
    "East,Laptop,1200,2\n"
    "East,Mouse,50,10\n"
    "West,Laptop,1800,3\n"
    "West,Monitor,600,2\n"
    "North,Laptop,600,1\n"
    "North,Mouse,25,5\n"
    "South,Monitor,900,3\n"
    "South,Mouse,75,15\n"
)


def _handbook(path: Path) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.drawString(50, 750, "The probation period is 90 days.")
    pdf.showPage()
    pdf.drawString(
        50, 750, "Employees get 24 days of annual leave. Expense claims must be filed within 14 days."
    )
    pdf.showPage()
    pdf.save()


def _staff(path: Path) -> None:
    workbook = Workbook()
    staff = workbook.active
    staff.title = "Staff"
    staff.append(["name", "department", "salary"])
    for row in [
        ("Asha", "Engineering", 90000),
        ("Ben", "Engineering", 80000),
        ("Chen", "Sales", 60000),
        ("Dana", "Sales", 70000),
        ("Eli", "HR", 50000),
    ]:
        staff.append(row)
    offices = workbook.create_sheet("Offices")
    offices.append(["city", "seats"])
    offices.append(["Pune", 120])
    offices.append(["Austin", 80])
    workbook.save(path)


def make_samples(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "company_policy.txt").write_text(POLICY, encoding="utf-8")
    (out_dir / "sales.csv").write_text(SALES, encoding="utf-8")
    _handbook(out_dir / "handbook.pdf")
    _staff(out_dir / "staff.xlsx")
    return out_dir


if __name__ == "__main__":
    print("Wrote", make_samples(Path(__file__).parent))
