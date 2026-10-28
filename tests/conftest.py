import pytest


@pytest.fixture(autouse=True)
def _node_id_file(tmp_path, monkeypatch):
    """Keep tests from writing a node_id into the real home directory."""
    from zilver import node_types
    monkeypatch.setattr(node_types, "_NODE_ID_FILE", tmp_path / "node_id")
