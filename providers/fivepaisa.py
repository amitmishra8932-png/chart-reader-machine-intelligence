"""5paisa integration boundary.

Authentication and endpoint calls are intentionally not guessed. Implement only
against the current official 5paisa Xstream documentation after broker support
confirms the account's market-data permissions and callback configuration.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderStatus:
    connected: bool
    message: str


def connection_status() -> ProviderStatus:
    return ProviderStatus(
        connected=False,
        message="5paisa authentication is pending configuration and verification.",
    )
