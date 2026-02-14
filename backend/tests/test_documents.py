"""Tests for document processing."""
import pytest
def test_text_parser():
    """Test plain text parsing."""
    from modules.documents.parsers import DocumentParser
    result = DocumentParser.parse("test.txt", b"Hello World")
    assert result == "Hello World"
def test_json_parser():
    """Test JSON parsing."""
    from modules.documents.parsers import DocumentParser
    result = DocumentParser.parse("test.json", b'{"key": "value"}')
    assert "key" in result
    assert "value" in result
def test_xml_parser():
    """Test XML parsing."""
    from modules.documents.parsers import DocumentParser
    xml = b'<root><item>test</item></root>'
    result = DocumentParser.parse("test.xml", xml)
    assert "test" in result
def test_export_txt():
    """Test text export."""
    import tempfile
    import os
    from modules.documents.export import DocumentExporter
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {'exports_path': tmpdir}
        exporter = DocumentExporter(config)
        chat_data = {
            'character': 'TestBot',
            'messages': [
                {'role': 'user', 'name': 'User', 'content': 'Hello'},
                {'role': 'assistant', 'name': 'Bot', 'content': 'Hi there!'}
            ]
        }
        result = exporter.export('txt', chat_data)
        assert result['status'] == 'ok'
        assert os.path.exists(os.path.join(tmpdir, result['filename']))
def test_summary():
    """Test summary generation."""
    import tempfile
    from modules.documents.export import DocumentExporter
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {'exports_path': tmpdir}
        exporter = DocumentExporter(config)
        chat_data = {
            'character': 'TestBot',
            'messages': [
                {'role': 'user', 'name': 'User', 'content': 'Tell me about machine learning'},
                {'role': 'assistant', 'name': 'Bot', 'content': 'Machine learning is a subset of AI that enables systems to learn from data.'},
            ]
        }
        summary = exporter.generate_summary(chat_data)
        assert 'TestBot' in summary
        assert 'Total messages' in summary
