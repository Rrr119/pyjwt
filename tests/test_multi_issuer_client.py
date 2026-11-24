from __future__ import annotations

import time

import pytest

import jwt
from jwt import MultiIssuerJWKClient, PyJWKClient
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    PyJWKClientError,
)

from .utils import crypto_required, utc_timestamp

pytestmark = crypto_required

ISSUERS = [
    "https://issuer-a.example.com",
    "https://issuer-b.example.com",
    "https://issuer-c.example.com",
]


def make_rsa_key_pair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def jwks_payload(public_key, kid: str) -> dict:
    from jwt.algorithms import RSAAlgorithm

    jwk = RSAAlgorithm.to_jwk(public_key, as_dict=True)
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [jwk]}


class RotatingJWKClient(PyJWKClient):
    """A PyJWKClient whose JWKS payload can be swapped to simulate rotation."""

    def __init__(self, jwks: dict, **kwargs):
        kwargs.setdefault("cache_keys", True)
        kwargs.setdefault("cooldown_duration", 0)
        super().__init__("https://unused.example.com/jwks.json", **kwargs)
        self._jwks = jwks
        self.fetch_count = 0

    def fetch_data(self):
        self.fetch_count += 1
        if self.jwk_set_cache is not None:
            self.jwk_set_cache.put(self._jwks)
        self._last_successful_fetch = time.monotonic()
        return self._jwks

    def rotate_to(self, jwks: dict) -> None:
        self._jwks = jwks


def make_token(private_key, issuer: str, kid: str, **claims) -> str:
    payload = {"iss": issuer, "sub": "user-1", "exp": utc_timestamp() + 3600}
    payload.update(claims)
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def issuer_setup():
    """Three issuers, each with a generation-1 signing key."""
    setup = {}
    for index, issuer in enumerate(ISSUERS):
        priv1, pub1 = make_rsa_key_pair()
        priv2, pub2 = make_rsa_key_pair()
        kid1, kid2 = f"issuer-{index}-key-1", f"issuer-{index}-key-2"
        client = RotatingJWKClient(jwks_payload(pub1, kid1))
        setup[issuer] = {
            "client": client,
            "gen1": (priv1, pub1, kid1),
            "gen2": (priv2, pub2, kid2),
        }
    multi = MultiIssuerJWKClient(
        {issuer: cfg["client"] for issuer, cfg in setup.items()}
    )
    return multi, setup


def rotate_all(setup) -> None:
    for cfg in setup.values():
        _, pub2, kid2 = cfg["gen2"]
        cfg["client"].rotate_to(jwks_payload(pub2, kid2))


class TestMultiIssuerJWKClient:
    def test_decode_accepts_all_whitelisted_issuers(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        for issuer, cfg in setup.items():
            priv1, _, kid1 = cfg["gen1"]
            token = make_token(priv1, issuer, kid1)
            payload = multi.decode(token, algorithms=["RS256"])
            assert payload["iss"] == issuer
            assert payload["sub"] == "user-1"

    def test_old_tokens_still_verify_after_all_issuers_rotate(
        self, issuer_setup
    ) -> None:
        multi, setup = issuer_setup

        old_tokens = {}
        for issuer, cfg in setup.items():
            priv1, _, kid1 = cfg["gen1"]
            token = make_token(priv1, issuer, kid1)
            assert multi.decode(token, algorithms=["RS256"])["iss"] == issuer
            old_tokens[issuer] = token

        rotate_all(setup)

        # New keys are picked up automatically (unknown kid triggers refresh).
        for issuer, cfg in setup.items():
            priv2, _, kid2 = cfg["gen2"]
            token = make_token(priv2, issuer, kid2)
            assert multi.decode(token, algorithms=["RS256"])["iss"] == issuer

        # Old keys are gone from every JWKS, but cached keys still verify.
        for issuer, token in old_tokens.items():
            payload = multi.decode(token, algorithms=["RS256"])
            assert payload["iss"] == issuer

    def test_rotation_fetches_new_jwks_on_unknown_kid(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        issuer = ISSUERS[0]
        cfg = setup[issuer]
        priv1, _, kid1 = cfg["gen1"]

        multi.decode(make_token(priv1, issuer, kid1), algorithms=["RS256"])
        fetches_before = cfg["client"].fetch_count

        rotate_all(setup)
        priv2, _, kid2 = cfg["gen2"]
        multi.decode(make_token(priv2, issuer, kid2), algorithms=["RS256"])

        assert cfg["client"].fetch_count > fetches_before

    def test_untrusted_issuer_is_rejected(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        priv1, _, kid1 = setup[ISSUERS[0]]["gen1"]
        token = make_token(priv1, "https://evil.example.com", kid1)

        with pytest.raises(InvalidIssuerError):
            multi.decode(token, algorithms=["RS256"])

    def test_missing_issuer_is_rejected(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        priv1, _, kid1 = setup[ISSUERS[0]]["gen1"]
        token = jwt.encode(
            {"sub": "user-1"}, priv1, algorithm="RS256", headers={"kid": kid1}
        )

        with pytest.raises(InvalidIssuerError):
            multi.decode(token, algorithms=["RS256"])

    def test_token_cannot_be_replayed_across_issuers(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        # Issuer B's key signs a token claiming to be from issuer A, reusing
        # A's kid so the key resolves but the signature must not.
        priv_b, _, _ = setup[ISSUERS[1]]["gen1"]
        _, _, kid_a = setup[ISSUERS[0]]["gen1"]
        token = make_token(priv_b, ISSUERS[0], kid_a)

        with pytest.raises(InvalidSignatureError):
            multi.decode(token, algorithms=["RS256"])

    def test_issuers_are_isolated_per_client(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        # B's kid is unknown to A's JWKS, so a token with iss=A and B's kid
        # cannot resolve a key at all.
        _, _, kid_b = setup[ISSUERS[1]]["gen1"]
        priv_a, _, _ = setup[ISSUERS[0]]["gen1"]
        token = make_token(priv_a, ISSUERS[0], kid_b)

        with pytest.raises(PyJWKClientError):
            multi.decode(token, algorithms=["RS256"])

    def test_decode_passes_through_decode_arguments(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        priv1, _, kid1 = setup[ISSUERS[0]]["gen1"]

        token = make_token(priv1, ISSUERS[0], kid1, aud="my-api")
        payload = multi.decode(token, algorithms=["RS256"], audience="my-api")
        assert payload["aud"] == "my-api"

        with pytest.raises(InvalidAudienceError):
            multi.decode(token, algorithms=["RS256"], audience="other-api")

        expired = make_token(priv1, ISSUERS[0], kid1, exp=utc_timestamp() - 10)
        with pytest.raises(ExpiredSignatureError):
            multi.decode(expired, algorithms=["RS256"])
        payload = multi.decode(
            expired, algorithms=["RS256"], options={"verify_exp": False}
        )
        assert payload["iss"] == ISSUERS[0]

    def test_get_signing_key_from_jwt_routes_by_issuer(self, issuer_setup) -> None:
        multi, setup = issuer_setup
        for issuer, cfg in setup.items():
            priv1, pub1, kid1 = cfg["gen1"]
            token = make_token(priv1, issuer, kid1)
            key = multi.get_signing_key_from_jwt(token)
            assert key.key_id == kid1

    def test_get_signing_key_from_jwt_rejects_untrusted_issuer(
        self, issuer_setup
    ) -> None:
        multi, setup = issuer_setup
        priv1, _, kid1 = setup[ISSUERS[0]]["gen1"]
        token = make_token(priv1, "https://evil.example.com", kid1)

        with pytest.raises(InvalidIssuerError):
            multi.get_signing_key_from_jwt(token)

    def test_issuer_whitelist_is_exposed(self, issuer_setup) -> None:
        multi, _ = issuer_setup
        assert multi.issuers == frozenset(ISSUERS)

    def test_url_configured_issuers_build_cached_clients(self) -> None:
        multi = MultiIssuerJWKClient(
            {issuer: f"{issuer}/jwks.json" for issuer in ISSUERS}
        )
        assert multi.issuers == frozenset(ISSUERS)
        for issuer in ISSUERS:
            client = multi.get_client(issuer)
            assert isinstance(client, PyJWKClient)
            # cache_keys=True by default so rotated-out keys keep verifying
            assert hasattr(client.get_signing_key, "cache_info")

    def test_empty_issuer_mapping_is_rejected(self) -> None:
        with pytest.raises(PyJWKClientError):
            MultiIssuerJWKClient({})

    def test_invalid_issuer_config_is_rejected(self) -> None:
        with pytest.raises(PyJWKClientError):
            MultiIssuerJWKClient({ISSUERS[0]: 12345})
