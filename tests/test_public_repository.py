from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("overleaf", "google drive", "phase iv", "phase v", "phase vi", "phase vii")


def test_public_text_has_no_internal_workflow_language():
    suffixes = {".py", ".md", ".toml", ".json", ".tex", ".cff"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part.startswith(".") or part in {"build", "dist"} for part in path.relative_to(ROOT).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN:
            assert term not in text, f"Internal term {term!r} found in {path.relative_to(ROOT)}"
