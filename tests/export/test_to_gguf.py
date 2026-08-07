# tests/export/test_to_gguf.py
from pathlib import Path

from llm_internal.export.to_gguf import render_modelfile, write_modelfile


def test_render_modelfile_references_gguf_file_and_system_prompt():
    content = render_modelfile("model-q4_k_m.gguf", system_prompt="You are a helpful assistant with tools.")

    assert "FROM ./model-q4_k_m.gguf" in content
    assert 'SYSTEM """You are a helpful assistant with tools."""' in content


def test_write_modelfile_creates_file_with_content(tmp_path: Path):
    path = tmp_path / "Modelfile"

    write_modelfile(path, "FROM ./x.gguf\n")

    assert path.read_text() == "FROM ./x.gguf\n"
