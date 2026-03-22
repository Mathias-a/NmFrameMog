"""Tests for API client — retry logic, auth, payload format, error handling.

All tests mock httpx to avoid real network calls.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from astar_island.api import (
    API_BASE,
    _request,
    _resolve_token,
    get_active_round,
    get_analysis,
    get_round_detail,
    get_rounds,
    make_client,
    submit_prediction,
)


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


class TestResolveToken:
    def test_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTAR_API_TOKEN", "test-jwt-token")
        assert _resolve_token() == "test-jwt-token"

    def test_missing_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("ASTAR_API_TOKEN", raising=False)
        # Patch the .env lookup path to a non-existent location
        with patch("astar_island.api.Path") as mock_path:
            mock_file = MagicMock()
            mock_file.exists.return_value = False
            mock_path.return_value.resolve.return_value.parents.__getitem__ = (
                lambda self, idx: tmp_path
            )
            with pytest.raises(ValueError, match="ASTAR_API_TOKEN"):
                _resolve_token()


# ---------------------------------------------------------------------------
# make_client
# ---------------------------------------------------------------------------


class TestMakeClient:
    def test_sets_bearer_header(self) -> None:
        client = make_client(token="my-token")
        assert client.headers["Authorization"] == "Bearer my-token"
        client.close()


# ---------------------------------------------------------------------------
# _request — retry logic
# ---------------------------------------------------------------------------


class TestRequestRetry:
    def test_429_retries_and_succeeds(self) -> None:
        """Should retry on 429 and succeed when server eventually returns 200."""
        responses = [
            httpx.Response(429, request=httpx.Request("GET", "http://test")),
            httpx.Response(429, request=httpx.Request("GET", "http://test")),
            httpx.Response(
                200, json={"ok": True}, request=httpx.Request("GET", "http://test")
            ),
        ]
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(side_effect=responses)

        with patch("astar_island.api.time.sleep"):  # skip actual waiting
            resp = _request(client, "GET", "/test", max_retries=3)

        assert resp.status_code == 200
        assert client.get.call_count == 3

    def test_429_exhausted_raises(self) -> None:
        """Should raise RuntimeError after exhausting retries on 429."""
        response_429 = httpx.Response(429, request=httpx.Request("GET", "http://test"))
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=response_429)

        with patch("astar_island.api.time.sleep"):
            with pytest.raises(RuntimeError, match="Rate limited"):
                _request(client, "GET", "/test", max_retries=2)

        # Should have tried 3 times (initial + 2 retries)
        assert client.get.call_count == 3

    def test_non_429_error_raises_immediately(self) -> None:
        """Non-429 errors should raise immediately without retry."""
        response_500 = httpx.Response(500, request=httpx.Request("GET", "http://test"))
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=response_500)

        with pytest.raises(httpx.HTTPStatusError):
            _request(client, "GET", "/test", max_retries=3)

        assert client.get.call_count == 1

    def test_post_method(self) -> None:
        """POST requests should use client.post."""
        response_200 = httpx.Response(
            200, json={}, request=httpx.Request("POST", "http://test")
        )
        client = MagicMock(spec=httpx.Client)
        client.post = MagicMock(return_value=response_200)

        _request(client, "POST", "/test", json_body={"key": "val"})
        client.post.assert_called_once()


# ---------------------------------------------------------------------------
# Round discovery
# ---------------------------------------------------------------------------


class TestGetRounds:
    def test_parses_round_list(self, mock_round_list: list[dict[str, object]]) -> None:
        resp = httpx.Response(
            200, json=mock_round_list, request=httpx.Request("GET", "http://test")
        )
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=resp)

        rounds = get_rounds(client)
        assert len(rounds) == 2
        assert rounds[0]["round_number"] == 18
        assert rounds[1]["status"] == "active"


class TestGetActiveRound:
    def test_finds_active(self, mock_round_list: list[dict[str, object]]) -> None:
        resp = httpx.Response(
            200, json=mock_round_list, request=httpx.Request("GET", "http://test")
        )
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=resp)

        active = get_active_round(client)
        assert active is not None
        assert active["round_number"] == 19

    def test_no_active(self) -> None:
        resp = httpx.Response(
            200,
            json=[{"id": "x", "round_number": 1, "status": "completed"}],
            request=httpx.Request("GET", "http://test"),
        )
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=resp)

        assert get_active_round(client) is None


# ---------------------------------------------------------------------------
# Submit prediction
# ---------------------------------------------------------------------------


class TestSubmitPrediction:
    def test_payload_format(self) -> None:
        """Submit should POST with correct body keys."""
        resp = httpx.Response(
            200,
            json={"status": "accepted"},
            request=httpx.Request("POST", "http://test"),
        )
        client = MagicMock(spec=httpx.Client)
        client.post = MagicMock(return_value=resp)

        prediction = [[[1.0 / 6] * 6] * 40] * 40  # 40x40x6

        with patch("astar_island.api.time.sleep"):
            result = submit_prediction(client, "round-id-123", 0, prediction)

        assert result["status"] == "accepted"
        # Verify the POST body structure
        call_kwargs = client.post.call_args
        posted_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert posted_json["round_id"] == "round-id-123"
        assert posted_json["seed_index"] == 0
        assert "prediction" in posted_json


# ---------------------------------------------------------------------------
# Analysis (404/409 handling)
# ---------------------------------------------------------------------------


class TestGetAnalysis:
    def test_404_returns_none(self) -> None:
        """Analysis for unavailable round should return None, not raise."""
        response_404 = httpx.Response(404, request=httpx.Request("GET", "http://test"))
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=response_404)

        # get_analysis catches HTTPStatusError for 404/409
        result = get_analysis(client, "nonexistent-round", 0)
        assert result is None

    def test_409_returns_none(self) -> None:
        """Analysis conflict (409) should return None."""
        response_409 = httpx.Response(409, request=httpx.Request("GET", "http://test"))
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=response_409)

        result = get_analysis(client, "round-id", 0)
        assert result is None

    def test_success_returns_dict(self) -> None:
        """Successful analysis response should be returned as dict."""
        resp = httpx.Response(
            200,
            json={"ground_truth": [], "score": 75.5},
            request=httpx.Request("GET", "http://test"),
        )
        client = MagicMock(spec=httpx.Client)
        client.get = MagicMock(return_value=resp)

        result = get_analysis(client, "round-id", 0)
        assert result is not None
        assert result["score"] == 75.5
