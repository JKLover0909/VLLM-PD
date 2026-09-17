"""CCTVAI host metrics: standalone read-only hardware monitoring service.

This package is independent of the main Meibook application. It has no
import-time side effects (no I/O, no env reads) so it is safe to import
from any process, including the Meibook app container (which only needs
``tools.host_metrics.schemas``).

Submodules:
    schemas   -- shared Pydantic v2 contract (snapshot envelope + limits).
    collector -- psutil/nvidia-smi/docker sampler that writes snapshot.json.
    server    -- FastAPI+TLS read-only API serving the latest snapshot.
"""
