# src/llmcore/media/jobs.py
"""Lifecycle management for long-running media jobs.

Video generation, queue-based image vendors and aggregator predictions are all
asynchronous: submit, then wait.  :class:`MediaJobManager` owns that waiting —
the backoff schedule, the timeout policy, cancellation and the job registry — so
that no provider adapter implements its own polling loop.  Adapters only supply
:meth:`~llmcore.media.protocols.MediaJobPoller.poll_media_job`.

Polling is always available.  Webhooks (a later phase) are an optimization that
short-circuits the wait; they never become a requirement, because the common
development case has no public ingress.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable, Iterable
from typing import Any

from ..exceptions import MediaJobError, MediaJobTimeoutError
from .models import MediaJob, MediaJobStatus
from .protocols import MediaJobPoller

logger = logging.getLogger(__name__)

__all__ = ["JobPolicy", "MediaJobManager"]

#: Defaults chosen so a short image job feels responsive while a 10-minute video
#: job does not generate hundreds of requests.
DEFAULT_POLL_INITIAL_SECONDS = 2.0
DEFAULT_POLL_MAX_SECONDS = 30.0
DEFAULT_JOB_TIMEOUT_SECONDS = 1800.0


class JobPolicy:
    """Backoff and timeout policy for job polling.

    Attributes:
        poll_initial_seconds: First sleep after submission.
        poll_max_seconds: Ceiling for the exponential backoff.
        job_timeout_seconds: Default wall-clock budget for :meth:`MediaJobManager.wait`.
        jitter: Fractional jitter applied to each sleep, to avoid thundering
            herds when many jobs are submitted together.
    """

    __slots__ = ("jitter", "job_timeout_seconds", "poll_initial_seconds", "poll_max_seconds")


    def __init__(
        self,
        poll_initial_seconds: float = DEFAULT_POLL_INITIAL_SECONDS,
        poll_max_seconds: float = DEFAULT_POLL_MAX_SECONDS,
        job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
        jitter: float = 0.1,
    ) -> None:
        self.poll_initial_seconds = max(0.0, float(poll_initial_seconds))
        self.poll_max_seconds = max(self.poll_initial_seconds, float(poll_max_seconds))
        self.job_timeout_seconds = float(job_timeout_seconds)
        self.jitter = max(0.0, min(1.0, float(jitter)))

    #: Exponent ceiling for the backoff doubling. Any realistic
    #: ``poll_initial_seconds`` reaches ``poll_max_seconds`` long before 2**32,
    #: and without the cap a job polled a few thousand times overflows the float
    #: conversion of ``2 ** attempt`` — which a multi-hour video job would hit.
    _MAX_BACKOFF_SHIFT = 32

    def delay_for(self, attempt: int) -> float:
        """Return the sleep before poll *attempt* (1-based), with jitter."""
        shift = min(max(0, attempt - 1), self._MAX_BACKOFF_SHIFT)
        base = min(self.poll_initial_seconds * (2**shift), self.poll_max_seconds)
        if not self.jitter:
            return base
        return base * (1.0 + random.uniform(-self.jitter, self.jitter))

    @classmethod
    def from_config(cls, get: Callable[[str, Any], Any]) -> JobPolicy:
        """Build a policy from an llmcore config accessor.

        Args:
            get: A ``config.get(key, default)``-style callable.

        Returns:
            The configured policy.
        """
        return cls(
            poll_initial_seconds=get("media.jobs.poll_initial_seconds", DEFAULT_POLL_INITIAL_SECONDS),
            poll_max_seconds=get("media.jobs.poll_max_seconds", DEFAULT_POLL_MAX_SECONDS),
            job_timeout_seconds=get("media.jobs.job_timeout_seconds", DEFAULT_JOB_TIMEOUT_SECONDS),
        )


class MediaJobManager:
    """Tracks and drives long-running media jobs.

    The manager keeps every job it has seen in an in-memory registry so callers
    can enumerate outstanding work, and resolves a job back to its provider
    adapter through the resolver supplied at construction (normally
    :class:`~llmcore.media.manager.MediaManager`'s adapter lookup).

    Args:
        resolver: Maps a provider instance name to its media adapter.
        policy: Backoff/timeout policy; defaults are used when omitted.
    """

    def __init__(
        self,
        resolver: Callable[[str], Any],
        policy: JobPolicy | None = None,
    ) -> None:
        self._resolver = resolver
        self._policy = policy or JobPolicy()
        self._jobs: dict[str, MediaJob] = {}

    # --- registry ---

    def track(self, job: MediaJob) -> MediaJob:
        """Record *job* in the registry and return it.

        Called by routers immediately after an adapter returns a job handle, so
        an expensive submission is never lost to a dropped reference.
        """
        self._jobs[job.id] = job
        logger.debug(
            "Tracking media job %s (%s on %s/%s)",
            job.id,
            job.capability,
            job.provider,
            job.model,
        )
        return job

    def get(self, job_id: str) -> MediaJob | None:
        """Return the tracked job with *job_id*, if any."""
        return self._jobs.get(job_id)

    def list(self, *, active_only: bool = False) -> list[MediaJob]:
        """Return tracked jobs, newest first.

        Args:
            active_only: Exclude jobs that have reached a terminal state.
        """
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j for j in jobs if not j.is_terminal] if active_only else jobs

    def forget(self, job_id: str) -> None:
        """Drop *job_id* from the registry."""
        self._jobs.pop(job_id, None)

    # --- driving ---

    def _poller_for(self, job: MediaJob) -> MediaJobPoller:
        """Resolve the adapter that can poll *job*.

        Raises:
            MediaJobError: If the provider is gone or cannot poll.
        """
        adapter = self._resolver(job.provider)
        if adapter is None:
            raise MediaJobError(
                "Provider is no longer configured, so the job cannot be polled.",
                job_id=job.id,
                status=job.status,
                provider_name=job.provider,
                capability=job.capability,
            )
        if not isinstance(adapter, MediaJobPoller):
            raise MediaJobError(
                "Provider does not implement media job polling.",
                job_id=job.id,
                status=job.status,
                provider_name=job.provider,
                capability=job.capability,
            )
        return adapter

    async def poll(self, job: MediaJob) -> MediaJob:
        """Refresh *job* once and return the updated handle.

        Terminal jobs are returned unchanged without contacting the vendor.
        """
        if job.is_terminal:
            return job
        updated = await self._poller_for(job).poll_media_job(job)
        updated.touch()
        self._jobs[updated.id] = updated
        return updated

    async def wait(
        self,
        job: MediaJob,
        *,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> MediaJob:
        """Poll *job* until it reaches a terminal state.

        Args:
            job: The handle returned by a router.
            timeout: Wall-clock budget in seconds; the configured default when
                omitted. ``None`` in config means wait indefinitely.
            raise_on_failure: Raise :class:`MediaJobError` when the job ends
                ``FAILED``/``CANCELED``/``EXPIRED`` instead of returning it.

        Returns:
            The terminal job handle.

        Raises:
            MediaJobTimeoutError: If the budget elapses first. **The job is not
                cancelled** — the handle stays valid and can be waited on again,
                so an expensive generation is never discarded over a client-side
                deadline. This is why the deadline is an explicit ``timeout``
                parameter rather than ``asyncio.timeout``: cancelling the task
                would be exactly the wrong behaviour here.
            MediaJobError: On terminal failure when *raise_on_failure*.
        """
        budget = self._policy.job_timeout_seconds if timeout is None else timeout
        started = time.monotonic()
        attempt = 0

        while not job.is_terminal:
            attempt += 1
            delay = self._policy.delay_for(attempt)
            if budget is not None and budget >= 0:
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0:
                    raise MediaJobTimeoutError(
                        job_id=job.id,
                        status=job.status,
                        timeout_seconds=budget,
                        provider_name=job.provider,
                    )
                delay = min(delay, remaining)
            await asyncio.sleep(delay)
            job = await self.poll(job)
            logger.debug(
                "Polled media job %s: status=%s progress=%s", job.id, job.status, job.progress
            )

        if raise_on_failure and job.status is not MediaJobStatus.SUCCEEDED:
            raise MediaJobError(
                job.error or "Job did not succeed.",
                job_id=job.id,
                status=job.status,
                provider_name=job.provider,
                capability=job.capability,
            )
        return job

    async def cancel(self, job: MediaJob) -> MediaJob:
        """Ask the vendor to cancel *job*.

        Terminal jobs are returned unchanged.
        """
        if job.is_terminal:
            return job
        updated = await self._poller_for(job).cancel_media_job(job)
        updated.touch()
        self._jobs[updated.id] = updated
        return updated

    async def cancel_all(self, jobs: Iterable[MediaJob] | None = None) -> list[MediaJob]:
        """Cancel *jobs* (default: every active tracked job), best effort.

        Failures are logged, not raised: shutdown should not be blocked by a
        vendor that will not answer.
        """
        targets = list(jobs) if jobs is not None else self.list(active_only=True)
        results: list[MediaJob] = []
        for job in targets:
            try:
                results.append(await self.cancel(job))
            except Exception as e:
                logger.warning("Failed to cancel media job %s: %s", job.id, e)
        return results
