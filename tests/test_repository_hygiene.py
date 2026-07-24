from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "THIRD_PARTY_NOTICE.md",
    "pyproject.toml",
    "src/physicshead/__init__.py",
    "src/physicshead/head.py",
    "src/physicshead/losses.py",
    "examples/tau_integration.py",
    "results/checkpoint_audit.csv",
    "results/provenance.json",
    "tests/test_head.py",
    "tests/test_losses.py",
    "tests/test_public_api.py",
    "tests/test_repository_hygiene.py",
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".csv", ".cff", ""}


def release_files():
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") and part not in {".gitattributes", ".gitignore"} for part in path.relative_to(ROOT).parts)
        and not any(part in {"build", "dist", "__pycache__"} or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
    }


def release_text():
    chunks = []
    for relative in sorted(ALLOWED - {"tests/test_repository_hygiene.py"}):
        path = ROOT / relative
        if path.suffix.lower() in TEXT_SUFFIXES:
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_allowlist_and_large_files():
    assert release_files() == ALLOWED
    assert all((ROOT / path).stat().st_size < 1_000_000 for path in ALLOWED)


def test_no_data_or_model_artifacts():
    forbidden_extensions = {
        ".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth", ".ckpt",
        ".onnx", ".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg",
        ".log", ".pdf",
    }
    assert not [path for path in ROOT.rglob("*") if path.suffix.lower() in forbidden_extensions]


def test_no_absolute_paths_secrets_internal_names_or_pde_classes():
    text = release_text()
    assert not re.search(r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/mnt/|/Users/)", text)
    secret_patterns = [
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"AKIA[0-9A-Z]{16}",
        r"gh[pousr]_[A-Za-z0-9]{20,}",
    ]
    assert not any(re.search(pattern, text) for pattern in secret_patterns)
    internal_patterns = [
        r"\bV[1-7]\b", r"\bv5v2\b", r"\bAbl\b", r"TAU-PINN",
        r"PhysicsHeadV[1-7]", r"AdvectionDiffusionLoss", r"CombinedPINN",
    ]
    assert not any(re.search(pattern, text, re.IGNORECASE) for pattern in internal_patterns)
