"""Typed witness failures: an assurance refusal must never become transport fallback."""

ROUTING_VERSION = "strata-v6"


class WitnessRefused(ValueError):
    """Invalid schema, model provenance or confinement. Terminal for the selected route."""


class WitnessUnavailable(RuntimeError):
    """The transport could not answer, e.g. connection, timeout, provider or quota failure."""


class WitnessAnomalous(RuntimeError):
    """The response carried unknown (non-error) envelope shapes. It is recorded, can never verify
    a claim and is never plain availability: the port retries it a bounded number of times."""

    def __init__(self, message: str = "", anomalies: list | None = None) -> None:
        super().__init__(message)
        self.anomalies = list(anomalies or [])
