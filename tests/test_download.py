"""The downloader against a local HTTP server: checks, resumes and cancels.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import http.server
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parakeet_nemotron_addon import download  # noqa: E402

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
REPO = 'nvidia/parakeet-tdt-0.6b-v3'
REVISION = 'a' * 40


class _Handler(http.server.BaseHTTPRequestHandler):
    requests: list[tuple[str, str]] = []
    honour_ranges = True

    def do_GET(self):  # noqa: N802
        wanted = self.headers.get('Range') or ''
        type(self).requests.append((self.path, wanted))
        start, end = 0, len(PAYLOAD) - 1
        if wanted and self.honour_ranges:
            low, _, high = wanted.split('=')[1].partition('-')
            start, end = int(low), int(high) if high else len(PAYLOAD) - 1
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:end + 1]
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture()
def server(monkeypatch, tmp_path):
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('HF_ENDPOINT', f'http://127.0.0.1:{httpd.server_address[1]}')
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    _Handler.requests = []
    _Handler.honour_ranges = True
    yield httpd
    httpd.shutdown()


def _pin(sha256: str | None = None) -> dict:
    return {'repo': REPO, 'revision': REVISION, 'path': 'parakeet-tdt-0.6b-v3.q8_0.gguf',
            'size': len(PAYLOAD), 'sha256': sha256 or hashlib.sha256(PAYLOAD).hexdigest()}


def _ensure(pins, cancel=lambda: False, progress=lambda *_: None):
    download.ensure(pins, progress, cancel)


def test_downloads_checks_and_stamps(server, tmp_path):
    pin = _pin()
    seen = []
    _ensure([pin], progress=lambda done, total: seen.append((done, total)))
    path = download.local_path(pin)
    assert path == tmp_path / 'parakeet-nemotron' / 'parakeet-tdt-0.6b-v3' / REVISION / 'parakeet-tdt-0.6b-v3.q8_0.gguf'
    assert path.read_bytes() == PAYLOAD
    assert download.is_present(pin)
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert _Handler.requests == [(f'/{REPO}/resolve/{REVISION}/parakeet-tdt-0.6b-v3.q8_0.gguf', '')]
    _ensure([pin])  # present now: nothing is fetched again
    assert len(_Handler.requests) == 1


def test_a_server_that_ignores_the_resume_starts_over(server):
    _Handler.honour_ranges = False
    pin = _pin()
    part = download.local_path(pin).with_name('parakeet-tdt-0.6b-v3.q8_0.gguf.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(b'x' * 1000)
    _ensure([pin])
    assert download.local_path(pin).read_bytes() == PAYLOAD


def test_resumes_from_the_part_file(server):
    pin = _pin()
    part = download.local_path(pin).with_name('parakeet-tdt-0.6b-v3.q8_0.gguf.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    _ensure([pin])
    assert _Handler.requests[0][1] == f'bytes=1000-{len(PAYLOAD) - 1}'
    assert download.local_path(pin).read_bytes() == PAYLOAD


def test_a_wrong_sha256_discards_the_file(server):
    pin = _pin(sha256='0' * 64)
    with pytest.raises(download.DownloadError, match='sha256'):
        _ensure([pin])
    path = download.local_path(pin)
    assert not path.exists() and not path.with_name(path.name + '.part').exists()


def test_a_cancel_stops_the_download(server):
    pin = _pin()
    with pytest.raises(download.Cancelled):
        _ensure([pin], cancel=lambda: True)
    assert not download.is_present(pin)


def test_a_file_without_its_stamp_is_checked_not_fetched(server):
    pin = _pin()
    path = download.local_path(pin)
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD)
    assert not download.is_present(pin)
    _ensure([pin])
    assert download.is_present(pin)
    assert _Handler.requests == []


def test_an_unreachable_server_is_a_download_error(monkeypatch, tmp_path):
    monkeypatch.setenv('HF_ENDPOINT', 'http://127.0.0.1:9')  # discard port: nothing listens
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    with pytest.raises(download.DownloadError):
        _ensure([_pin()])
