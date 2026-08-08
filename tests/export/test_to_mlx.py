# tests/export/test_to_mlx.py
from pathlib import Path

from llm_internal.export.to_mlx import render_mlx_readme, write_readme


def test_render_mlx_readme_references_output_dir_and_enable_thinking_false():
    content = render_mlx_readme("export")

    assert "mlx_lm.generate --model export" in content
    assert '"enable_thinking": false' in content
    assert "mlx_lm.server --model export" in content


def test_write_readme_creates_file_with_content(tmp_path: Path):
    path = tmp_path / "README.md"

    write_readme(path, "# hi\n")

    assert path.read_text() == "# hi\n"
