"""Job manager, dynamic batcher, batch sizing and result cache."""

from __future__ import annotations

import threading
import time

import pytest

from textlens.core.result import Page, Result
from textlens.errors import CapacityError, InputError, JobNotFoundError
from textlens.runtime.batching import DynamicBatcher, suggest_batch_size
from textlens.runtime.cache import ResultCache, config_hash
from textlens.runtime.jobs import JobManager


def _wait(jm, job_id, timeout=5.0):
    return jm.wait(job_id, timeout=timeout)


def test_job_success_failure_and_retry():
    jm = JobManager(workers=2, max_queue=4, retries=1)
    ok = jm.submit(lambda job: 42)
    assert _wait(jm, ok.id).result == 42 and jm.get(ok.id).status == "succeeded"

    calls = {"n": 0}

    def flaky(job):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "second time"

    j = jm.submit(flaky)
    assert _wait(jm, j.id).status == "succeeded" and j.attempts == 2

    bad = jm.submit(lambda job: (_ for _ in ()).throw(InputError("bad input")))
    done = _wait(jm, bad.id)
    assert done.status == "failed" and done.attempts == 1  # input errors are not retried
    assert done.error["error"] == "input_error"
    with pytest.raises(JobNotFoundError):
        jm.get("missing")
    jm.shutdown()


def test_backpressure_and_cancel():
    gate = threading.Event()
    jm = JobManager(workers=1, max_queue=1)
    running = jm.submit(lambda job: gate.wait(5))
    queued = jm.submit(lambda job: "never")
    with pytest.raises(CapacityError):
        jm.submit(lambda job: "overflow")
    assert jm.cancel(queued.id).status == "cancelled"
    gate.set()
    assert _wait(jm, running.id).status == "succeeded"
    assert jm.stats()["pending"] == 0
    jm.shutdown()


def test_cooperative_cancellation_of_running_job():
    jm = JobManager(workers=1)
    started = threading.Event()

    def work(job):
        started.set()
        for _ in range(100):
            job.check_cancelled()
            time.sleep(0.01)
        return "finished"

    j = jm.submit(work)
    started.wait(2)
    jm.cancel(j.id)
    assert _wait(jm, j.id).status == "cancelled"
    jm.shutdown()


def test_dynamic_batcher_groups_concurrent_requests():
    seen = []

    def process(items):
        seen.append(len(items))
        return [x * 2 for x in items]

    b = DynamicBatcher(process, max_batch=4, window_ms=100)
    futures = [b.submit(i) for i in range(6)]
    assert [f.result(timeout=5) for f in futures] == [0, 2, 4, 6, 8, 10]
    assert max(seen) <= 4 and sum(seen) == 6 and len(seen) <= 3
    b.close()


def test_dynamic_batcher_propagates_errors():
    b = DynamicBatcher(lambda items: (_ for _ in ()).throw(ValueError("boom")), max_batch=2, window_ms=10)
    with pytest.raises(ValueError):
        b(1, timeout=5)
    b.close()


def test_suggest_batch_size(fake_models):
    from conftest import fake_system

    from textlens.models.specs import get_spec

    size, why = suggest_batch_size(get_spec("fake-fast"), fake_system(ram=8))
    assert 1 <= size <= 32 and "RAM" in why


def _result(text="hello") -> Result:
    from textlens.core.result import Block

    return Result(pages=[Page(number=1, blocks=[Block(text=text)])])


def test_cache_modes(tmp_path):
    key = ResultCache.key("doc", config_hash({"model": "x"}))
    assert key != ResultCache.key("doc", config_hash({"model": "y"}))
    off = ResultCache("off")
    off.put(key, _result())
    assert off.get(key) is None
    mem = ResultCache("memory", max_items=1)
    mem.put(key, _result())
    assert mem.get(key).text == "hello"
    mem.put("other", _result("x"))
    assert mem.get(key) is None  # LRU evicted
    disk = ResultCache("disk", directory=tmp_path)
    disk.put(key, _result("persisted"))
    fresh = ResultCache("disk", directory=tmp_path)
    assert fresh.get(key).text == "persisted"
    assert fresh.clear() >= 1 and fresh.get(key) is None


def test_no_persist_disables_disk_cache(monkeypatch, tmp_path):
    from textlens.config import reload_settings

    monkeypatch.setenv("TEXTLENS_NO_PERSIST", "1")
    reload_settings()
    assert ResultCache("disk", directory=tmp_path).mode == "memory"
