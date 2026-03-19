"""Tests for Pydantic model parsing: SolveRequest, SolveResponse, and helpers."""

from __future__ import annotations

import base64

import pytest
from ai_accounting_agent.models import (
    FileAttachment,
    SolveResponse,
    TripletexCredentials,
    build_request,
    parse_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CREDS_DICT: dict[str, str] = {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "test-token-abc",
}

_CREDS = TripletexCredentials(
    base_url="https://tx-proxy.ainm.no/v2",
    session_token="test-token-abc",
)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseMinimalRequest:
    def test_parse_minimal_request(self) -> None:
        """Prompt + credentials only, no files."""
        data: dict[str, object] = {
            "prompt": "Create an employee",
            "tripletex_credentials": _CREDS_DICT,
        }
        req = parse_request(data)
        assert req.prompt == "Create an employee"
        assert req.tripletex_credentials.session_token == "test-token-abc"
        assert req.files == []


class TestParseFullRequest:
    def test_parse_full_request(self) -> None:
        """Prompt + credentials + files."""
        data: dict[str, object] = {
            "prompt": "Process this invoice",
            "tripletex_credentials": _CREDS_DICT,
            "files": [
                {
                    "filename": "invoice.pdf",
                    "content_base64": _b64("fake-pdf-content"),
                }
            ],
        }
        req = parse_request(data)
        assert len(req.files) == 1
        assert req.files[0].filename == "invoice.pdf"


class TestParseMissingOptionalFiles:
    def test_parse_request_missing_optional_files(self) -> None:
        """Files field is optional and defaults to empty list."""
        data: dict[str, object] = {
            "prompt": "Do something",
            "tripletex_credentials": _CREDS_DICT,
        }
        req = parse_request(data)
        assert req.files == []


class TestParseExtraFieldsIgnored:
    def test_parse_request_extra_fields_ignored(self) -> None:
        """Extra fields should not cause validation errors."""
        data: dict[str, object] = {
            "prompt": "Do something",
            "tripletex_credentials": _CREDS_DICT,
            "unexpected_field": "should be ignored",
        }
        # Pydantic v2 ignores extra fields by default
        req = parse_request(data)
        assert req.prompt == "Do something"


class TestCredentialsAuthTuple:
    def test_credentials_auth_tuple(self) -> None:
        """Auth tuple should be ('0', session_token)."""
        creds = TripletexCredentials(
            base_url="https://example.com",
            session_token="my-secret",
        )
        assert creds.auth_tuple == ("0", "my-secret")


class TestFileAttachmentDecode:
    def test_file_attachment_decode(self) -> None:
        """Decode should return the original bytes."""
        content = "Hello, world!"
        attachment = FileAttachment(
            filename="test.txt",
            content_base64=_b64(content),
        )
        assert attachment.decode() == content.encode()


class TestUnicodePromptsPreserved:
    @pytest.mark.parametrize(
        "prompt",
        [
            "Opprett ansatt med navn \u00c6rlig \u00d8stby",
            "Erstellen Sie einen Mitarbeiter namens \u00c4\u00d6\u00dc",
            "\u00c9l\u00e8ve fran\u00e7ais avec accent \u00ea",
            "Kari \u00c5se fra \u00d8stfold",
            "Stra\u00dfe und Gr\u00fc\u00dfe",
            "R\u00e9sum\u00e9 du projet",
        ],
    )
    def test_unicode_prompts_preserved(self, prompt: str) -> None:
        """Norwegian, German, and French characters survive round-trip."""
        data: dict[str, object] = {
            "prompt": prompt,
            "tripletex_credentials": _CREDS_DICT,
        }
        req = parse_request(data)
        assert req.prompt == prompt


class TestBuildRequestRoundtrip:
    def test_build_request_roundtrip(self) -> None:
        """build_request -> parse_request should round-trip cleanly."""
        files = [FileAttachment(filename="f.txt", content_base64=_b64("data"))]
        built = build_request(prompt="Test prompt", credentials=_CREDS, files=files)
        parsed = parse_request(built)
        assert parsed.prompt == "Test prompt"
        assert len(parsed.files) == 1
        assert parsed.files[0].filename == "f.txt"
        assert parsed.tripletex_credentials.base_url == _CREDS.base_url


class TestSolveResponseSerialization:
    def test_solve_response_serialization(self) -> None:
        """SolveResponse should serialize to dict with 'status' key."""
        resp = SolveResponse(status="completed")
        dumped = resp.model_dump()
        assert dumped == {"status": "completed"}

    def test_solve_response_json_roundtrip(self) -> None:
        resp = SolveResponse(status="error")
        json_str = resp.model_dump_json()
        restored = SolveResponse.model_validate_json(json_str)
        assert restored.status == "error"


class TestFuzzedRequestVariations:
    @pytest.mark.parametrize(
        "prompt,has_files,token",
        [
            ("Simple prompt", False, "tok-1"),
            ("", False, "tok-2"),
            ("A" * 5000, False, "tok-3"),
            ("With files", True, "tok-4"),
            ("Short", False, "tok-5"),
            ("Opprett ansatt \u00e6\u00f8\u00e5", False, "tok-6"),
            ("Numbers 12345", False, "tok-7"),
            ("Special chars !@#$%", False, "tok-8"),
            ("Newline\nin\nprompt", False, "tok-9"),
            ("Tabs\there\ttoo", False, "tok-10"),
            ("Mixed \u00e9\u00fc\u00f1 intl", True, "tok-11"),
        ],
    )
    def test_fuzzed_request_variations(
        self, prompt: str, has_files: bool, token: str
    ) -> None:
        """Various prompt shapes and field combos all parse correctly."""
        data: dict[str, object] = {
            "prompt": prompt,
            "tripletex_credentials": {
                "base_url": "https://example.com/v2",
                "session_token": token,
            },
        }
        if has_files:
            data["files"] = [{"filename": "doc.pdf", "content_base64": _b64("content")}]
        req = parse_request(data)
        assert req.prompt == prompt
        assert req.tripletex_credentials.session_token == token
        if has_files:
            assert len(req.files) == 1
        else:
            assert req.files == []
