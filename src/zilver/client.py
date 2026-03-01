"""HTTP clients."""

from __future__ import annotations

from typing import Any

import httpx

from .node import NodeCapabilities, SimJob, JobResult
from .batch_distributor import BatchResult, BatchSlice

import time


# ---------------------------------------------------------------------------
# NodeClient
# ---------------------------------------------------------------------------

class NodeClient:
    """
    HTTP client for a single remote Zilver node.

    Wraps the REST endpoints exposed by :func:`~zilver.server.make_app`.
    All methods are synchronous and raise ``httpx.HTTPStatusError`` on non-2xx
    responses and ``httpx.ConnectError`` / ``httpx.TimeoutException`` on
    network failures — callers should handle these as appropriate.

    Parameters
    ----------
    url:
        Base URL of the node server, e.g. ``"http://192.168.1.5:7700"``.
        No trailing slash required.
    timeout:
        Per-request timeout in seconds.  Defaults to 30 s, which is generous
        for large statevector jobs on remote hardware.

    Example
    -------
    ::

        client = NodeClient("http://192.168.1.5:7700")
        job = SimJob(circuit_ops=[], n_qubits=4, n_params=0, params=[])
        result = client.execute(job)
        assert result.verify(job)
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def execute(self, job: SimJob) -> JobResult:
        """
        Submit a :class:`~zilver.node.SimJob` to the remote node.

        Parameters
        ----------
        job:
            The simulation job to execute.

        Returns
        -------
        JobResult
            The result including expectation value and SHA-256 proof.

        Raises
        ------
        httpx.HTTPStatusError
            If the node returns a 422 (unsupported backend / over capacity)
            or any other non-2xx status.
        """
        resp = self._client.post(f"{self.url}/execute", json=job.to_dict())
        resp.raise_for_status()
        return JobResult(**resp.json())

    def caps(self) -> NodeCapabilities:
        """
        Retrieve the node's hardware capabilities.

        Returns
        -------
        NodeCapabilities
            Capabilities as advertised by the node at startup.
        """
        resp = self._client.get(f"{self.url}/caps")
        resp.raise_for_status()
        return NodeCapabilities(**resp.json())

    def health(self) -> dict[str, str]:
        """
        Quick health check.

        Returns
        -------
        dict
            ``{"status": "ok", "node_id": "<id>"}`` on success.

        Raises
        ------
        httpx.ConnectError
            If the node is unreachable.
        """
        resp = self._client.get(f"{self.url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "NodeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"NodeClient(url={self.url!r})"


# ---------------------------------------------------------------------------
# RegistryClient
# ---------------------------------------------------------------------------

class RegistryClient:
    """
    HTTP client for the Zilver capability registry.

    Wraps the REST endpoints exposed by
    :func:`~zilver.registry_server.make_registry_app`.
    Used by node daemons for self-registration and by
    :class:`NetworkCoordinator` for node discovery.

    Parameters
    ----------
    url:
        Base URL of the registry server, e.g. ``"http://registry-host:7701"``.
    timeout:
        Per-request timeout in seconds.  Discovery calls are cheap and default
        to 10 s; adjust upward on unreliable networks.

    Example
    -------
    ::

        reg = RegistryClient("http://registry-host:7701")
        reg.register(node.caps, node_url="http://my-mac:7700")
        # ... serve jobs ...
        reg.deregister(node.caps.node_id)
    """

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def register(self, caps: NodeCapabilities, node_url: str) -> bool:
        """
        Register (or re-register) a node with the registry.

        Parameters
        ----------
        caps:
            The node's hardware capabilities.
        node_url:
            The reachable HTTP base URL of the node server,
            e.g. ``"http://192.168.1.10:7700"``.  Stored by the registry so
            coordinators can connect directly without a second lookup.

        Returns
        -------
        bool
            ``True`` on success.

        Raises
        ------
        httpx.HTTPStatusError
            On 4xx / 5xx responses.
        """
        body = {"caps": caps.to_dict(), "url": node_url}
        resp = self._client.post(f"{self.url}/nodes", json=body)
        resp.raise_for_status()
        return resp.json().get("registered", False)

    def deregister(self, node_id: str) -> bool:
        """
        Mark a node offline in the registry.

        Parameters
        ----------
        node_id:
            The node's unique identifier.

        Returns
        -------
        bool
            ``True`` if the node was found and deregistered.
        """
        resp = self._client.delete(f"{self.url}/nodes/{node_id}")
        resp.raise_for_status()
        return resp.json().get("deregistered", False)

    def heartbeat(self, node_id: str) -> bool:
        """
        Refresh the last-seen timestamp for a node.

        Should be called on a regular interval (e.g. every 30 s) by the
        node daemon's background thread to prevent the registry from marking
        the node stale.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the node was not found.
        """
        resp = self._client.post(f"{self.url}/nodes/{node_id}/heartbeat")
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def match(
        self,
        backend:   str,
        n_qubits:  int,
        min_stake: int = 0,
    ) -> str | None:
        """
        Find the best available node for a job and return its URL.

        Parameters
        ----------
        backend:
            Requested simulator backend: ``"sv"``, ``"dm"``, or ``"tn"``.
        n_qubits:
            Number of qubits required.
        min_stake:
            Minimum stake threshold (default 0).

        Returns
        -------
        str | None
            The node's HTTP base URL, e.g. ``"http://192.168.1.5:7700"``,
            or ``None`` if no eligible node is available.
        """
        params: dict[str, Any] = {
            "backend": backend,
            "n_qubits": n_qubits,
            "min_stake": min_stake,
        }
        resp = self._client.get(f"{self.url}/match", params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("url")

    def nodes(self) -> list[dict[str, Any]]:
        """
        Return all currently online nodes.

        Returns
        -------
        list[dict]
            Each element is a ``NodeCapabilities.to_dict()`` extended with
            a ``"url"`` field.
        """
        resp = self._client.get(f"{self.url}/nodes")
        resp.raise_for_status()
        return resp.json()

    def summary(self) -> dict[str, Any]:
        """
        Return aggregate registry statistics.

        Returns
        -------
        dict
            Keys: ``online``, ``total_registered``, ``backends``,
            ``max_sv_qubits``, ``max_dm_qubits``, ``total_stake``.
        """
        resp = self._client.get(f"{self.url}/summary")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "RegistryClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RegistryClient(url={self.url!r})"


# ---------------------------------------------------------------------------
# NetworkCoordinator
# ---------------------------------------------------------------------------

class NetworkCoordinator:
    """
    High-level coordinator for distributed job submission.

    Resolves the best available node via the registry, submits a job to it,
    and returns the result.  For batch evaluation, it distributes parameter
    slices across multiple nodes and reassembles the results.

    Parameters
    ----------
    registry_url:
        Base URL of a running registry server.
    timeout:
        HTTP timeout forwarded to both :class:`RegistryClient` and
        :class:`NodeClient`.

    Example
    -------
    ::

        from zilver.client import NetworkCoordinator
        from zilver.node import SimJob

        coord = NetworkCoordinator("http://registry-host:7701")

        job = SimJob(
            circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
            n_qubits=4, n_params=0, params=[], backend="sv",
        )
        result = coord.submit(job)
        print(result.expectation)
    """

    def __init__(self, registry_url: str, timeout: float = 30.0) -> None:
        self.registry_url = registry_url
        self.timeout = timeout
        self._registry = RegistryClient(registry_url, timeout=min(timeout, 10.0))

    def submit(self, job: SimJob) -> JobResult:
        """
        Submit a single job to the best available node.

        Performs registry lookup → node selection → remote execution in one
        call.  The returned :class:`~zilver.node.JobResult` includes a
        SHA-256 proof that the caller can verify with ``result.verify(job)``.

        Parameters
        ----------
        job:
            The simulation job to execute.

        Returns
        -------
        JobResult

        Raises
        ------
        RuntimeError
            If no eligible node is available in the registry.
        httpx.HTTPStatusError
            On node-side errors (wrong backend, over-capacity, etc.).
        """
        node_url = self._registry.match(job.backend, job.n_qubits)
        if node_url is None:
            raise RuntimeError(
                f"No eligible node for backend={job.backend!r} "
                f"n_qubits={job.n_qubits}"
            )
        with NodeClient(node_url, timeout=self.timeout) as node_client:
            return node_client.execute(job)

    def submit_batch(
        self,
        circuit: Any,
        params_batch: Any,
        backend:    str = "sv",
        observable: str = "sum_z",
    ) -> BatchResult:
        """
        Evaluate a circuit at N parameter sets, distributed across online nodes.

        Queries the registry for all eligible nodes, splits ``params_batch``
        evenly across them, submits each slice as a :class:`~zilver.node.SimJob`
        with a single parameter set (one job per row), and assembles the
        results in original order.

        Parameters
        ----------
        circuit:
            A :class:`~zilver.circuit.Circuit` instance.
        params_batch:
            An ``(N, n_params)`` MLX array of parameter vectors.
        backend:
            Simulator backend: ``"sv"``, ``"dm"``, or ``"tn"``.
        observable:
            Observable to measure: ``"sum_z"`` or ``"z0"``.

        Returns
        -------
        BatchResult
            ``expectations[i]`` is the expectation value for ``params_batch[i]``.

        Raises
        ------
        RuntimeError
            If no eligible node is available.

        Notes
        -----
        Each parameter set is submitted as a separate :class:`~zilver.node.SimJob`.
        For large batches on a high-latency network, using the local
        :func:`~zilver.batch_distributor.run_local_batch` is faster because
        it dispatches all evaluations in a single Metal vmap call.
        This method is intended for genuinely distributed hardware where
        each node runs the job on different physical silicon.
        """
        import numpy as np
        from .node import job_from_circuit
        from .batch_distributor import _split_indices

        if hasattr(params_batch, 'ndim') and params_batch.ndim == 1:
            params_batch = params_batch[None, :]

        n_evals = params_batch.shape[0]

        # Discover all eligible nodes up front
        all_nodes_info = self._registry.nodes()
        eligible = [
            n for n in all_nodes_info
            if backend in n.get("backends", [])
            and n.get(f"{backend}_qubits_max", 0) >= circuit.n_qubits
        ]
        if not eligible:
            raise RuntimeError(
                f"No eligible node for backend={backend!r} "
                f"n_qubits={circuit.n_qubits}"
            )

        k = len(eligible)
        index_slices = _split_indices(n_evals, k)

        t0 = time.perf_counter()
        all_expectations: list[float] = [0.0] * n_evals
        result_slices: list[BatchSlice] = []

        for node_info, (start, end) in zip(eligible, index_slices):
            if start == end:
                continue

            node_url = node_info.get("url", "")
            node_id  = node_info.get("node_id", "unknown")

            ts = time.perf_counter()
            slice_expectations: list[float] = []

            with NodeClient(node_url, timeout=self.timeout) as nc:
                for i in range(start, end):
                    import mlx.core as mx
                    row = params_batch[i]
                    job = job_from_circuit(circuit, row, observable=observable, backend=backend)
                    result = nc.execute(job)
                    slice_expectations.append(result.expectation)

            slice_ms = (time.perf_counter() - ts) * 1000.0
            for i, val in enumerate(slice_expectations):
                all_expectations[start + i] = val

            result_slices.append(BatchSlice(
                node_id=node_id,
                start=start,
                end=end,
                expectations=slice_expectations,
                elapsed_ms=slice_ms,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BatchResult(
            expectations=all_expectations,
            n_evals=n_evals,
            n_nodes_used=len(result_slices),
            slices=result_slices,
            elapsed_ms=elapsed_ms,
        )

    def nodes(self) -> list[dict[str, Any]]:
        """Return all online nodes from the registry."""
        return self._registry.nodes()

    def summary(self) -> dict[str, Any]:
        """Return aggregate registry statistics."""
        return self._registry.summary()

    def __repr__(self) -> str:
        return f"NetworkCoordinator(registry={self.registry_url!r})"
