from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import timedelta
from ssl import SSLContext
from typing import Any, Sequence, Union

from .api_jwk import PyJWK
from .api_jwt import decode as decode_token
from .api_jwt import decode_complete
from .exceptions import InvalidIssuerError, PyJWKClientError
from .jwks_client import PyJWKClient
from .types import Options


class MultiIssuerJWKClient:
    """Resolve JWKS signing keys across multiple trusted issuers.

    Wraps one :class:`PyJWKClient` per trusted issuer and routes each
    token to the right client based on its (unverified) ``iss`` claim.
    Issuers not present in the mapping are rejected, giving callers a
    simple issuer whitelist on top of the regular JWKS machinery.

    Key rotation is handled by the underlying clients in both
    directions:

    - **New keys**: an unknown ``kid`` triggers a JWKS refresh (subject
      to the client's cooldown), so keys added by a rotation are picked
      up automatically.
    - **Old keys**: the per-key LRU cache (``cache_keys=True``, enabled
      by default here) retains previously fetched keys with no
      time-based expiry, so tokens signed with a key that has already
      been rotated out of the JWKS keep verifying until the cache
      evicts them.

    :param issuers: Mapping of trusted issuer names to either a JWKS
        endpoint URL or a pre-configured :class:`PyJWKClient` instance.
    :type issuers: dict[str, str | PyJWKClient]

    The remaining keyword arguments are forwarded to every
    :class:`PyJWKClient` constructed from a URL; they are ignored for
    issuers configured with a client instance.
    """

    def __init__(
        self,
        issuers: Mapping[str, Union[str, PyJWKClient]],
        *,
        cache_keys: bool = True,
        max_cached_keys: int = 16,
        cache_jwk_set: bool = True,
        lifespan: float = 300,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,
        ssl_context: SSLContext | None = None,
        cooldown_duration: float = 30,
    ) -> None:
        if not issuers:
            raise PyJWKClientError("At least one issuer must be configured")

        self._clients: dict[str, PyJWKClient] = {}
        for issuer, config in issuers.items():
            if isinstance(config, PyJWKClient):
                self._clients[issuer] = config
            elif isinstance(config, str):
                self._clients[issuer] = PyJWKClient(
                    config,
                    cache_keys=cache_keys,
                    max_cached_keys=max_cached_keys,
                    cache_jwk_set=cache_jwk_set,
                    lifespan=lifespan,
                    headers=headers,
                    timeout=timeout,
                    ssl_context=ssl_context,
                    cooldown_duration=cooldown_duration,
                )
            else:
                raise PyJWKClientError(
                    f'Issuer "{issuer}" must map to a JWKS URI or PyJWKClient, '
                    f'got "{type(config).__name__}"'
                )

    @property
    def issuers(self) -> frozenset[str]:
        """The whitelist of trusted issuer names."""
        return frozenset(self._clients)

    def get_client(self, issuer: str | None) -> PyJWKClient:
        """Return the :class:`PyJWKClient` for a whitelisted issuer.

        :raises InvalidIssuerError: If *issuer* is missing or not trusted.
        """
        try:
            return self._clients[issuer]  # type: ignore[index]
        except KeyError:
            raise InvalidIssuerError(f'Issuer "{issuer}" is not trusted') from None

    def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        """Return the signing key for a JWT, routed by its ``iss`` claim.

        The ``iss`` claim is read from the unverified payload, checked
        against the issuer whitelist, and used to select the JWKS
        client that resolves the token's ``kid``.

        :param token: The encoded JWT.
        :type token: str or bytes
        :returns: The matching signing key.
        :rtype: PyJWK
        :raises InvalidIssuerError: If the token's issuer is not trusted.
        """
        unverified = decode_complete(token, options={"verify_signature": False})
        issuer = unverified["payload"].get("iss")
        client = self.get_client(issuer)
        return client.get_signing_key(unverified["header"].get("kid"))

    def decode(
        self,
        token: str | bytes,
        algorithms: Sequence[str] | None = None,
        options: Options | None = None,
        audience: str | Iterable[str] | None = None,
        subject: str | None = None,
        issuer: str | Iterable[str] | None = None,
        leeway: float | timedelta = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Verify a JWT and return its claims, like :func:`jwt.decode`.

        The verification key is resolved automatically from the token's
        ``iss`` claim (which must be whitelisted) and ``kid`` header.
        Unless an explicit *issuer* is given, the ``iss`` claim is
        additionally validated against the issuer that resolved the
        key, so a token cannot be replayed across trusted issuers.

        All other arguments behave exactly as in :func:`jwt.decode`.
        """
        unverified = decode_complete(token, options={"verify_signature": False})
        token_issuer = unverified["payload"].get("iss")
        client = self.get_client(token_issuer)
        signing_key = client.get_signing_key(unverified["header"].get("kid"))

        if issuer is None:
            issuer = token_issuer

        return decode_token(
            token,
            signing_key.key,
            algorithms=algorithms,
            options=options,
            audience=audience,
            subject=subject,
            issuer=issuer,
            leeway=leeway,
            **kwargs,
        )
