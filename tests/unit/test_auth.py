"""
Tests for authentication.
"""
import pytest
import os
os.environ.setdefault("DASHBOARD_USERNAME", "testuser")
os.environ.setdefault("DASHBOARD_PASSWORD", "testpass")
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_32_chars_minimum_x")

from app.auth.service import AuthService, create_access_token, decode_token, hash_password, verify_password


class TestAuthService:
    def setup_method(self):
        self.auth = AuthService()

    def test_valid_credentials_return_token(self):
        # AuthService reads from environment
        import app.config.settings as settings_module
        settings_module.get_settings.cache_clear()
        os.environ["DASHBOARD_USERNAME"] = "admin"
        os.environ["DASHBOARD_PASSWORD"] = "secret123"
        self.auth = AuthService()

        token = self.auth.authenticate("admin", "secret123")
        assert token is not None
        assert len(token) > 10

    def test_wrong_password_returns_none(self):
        token = self.auth.authenticate("admin", "wrongpassword")
        assert token is None

    def test_wrong_username_returns_none(self):
        token = self.auth.authenticate("hacker", "secret123")
        assert token is None


class TestJWT:
    def test_token_encode_decode(self):
        token = create_access_token("testuser", expires_minutes=60)
        username = decode_token(token)
        assert username == "testuser"

    def test_invalid_token_returns_none(self):
        username = decode_token("invalid.token.here")
        assert username is None

    def test_empty_token_returns_none(self):
        username = decode_token("")
        assert username is None


class TestPasswordHashing:
    def test_hash_and_verify(self):
        password = "my_secure_password_123"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)

    def test_wrong_password_fails_verify(self):
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        """bcrypt should generate different salts."""
        hashed1 = hash_password("mypassword")
        hashed2 = hash_password("mypassword")
        assert hashed1 != hashed2
        assert verify_password("mypassword", hashed1)
        assert verify_password("mypassword", hashed2)
