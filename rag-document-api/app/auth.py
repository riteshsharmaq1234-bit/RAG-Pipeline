"""Authentication module.

Implements a minimal JWT-based auth flow with a single hardcoded user
(``admin`` / ``admin123``) as specified. Passwords are verified through
``passlib`` (bcrypt) and tokens are signed/verified with ``python-jose``.

The pieces exposed here are reused via FastAPI dependency injection:

* :func:`authenticate_user`   - validate username/password.
* :func:`create_access_token` - mint a signed JWT.
* :func:`verify_token`        - decode and validate a JWT.
* :func:`get_current_user`    - FastAPI dependency guarding routes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

logger = logging.getLogger(__name__)

# Password hashing context (bcrypt).
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# The token URL is relative; it points at the POST /login route.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Hardcoded credentials. The password is hashed once at import time so we
# never compare plaintext directly.
ADMIN_USERNAME: str = "admin"
ADMIN_PASSWORD = "admin123"

# Reusable 401 exception for any credential failure.
CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return ``True`` if ``plain_password`` matches ``hashed_password``."""
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> bool:
    """Validate a username/password pair against the hardcoded admin user.

    Args:
        username: Submitted username.
        password: Submitted plaintext password.

    Returns:
        ``True`` when the credentials are valid, otherwise ``False``.
    """
    if username != ADMIN_USERNAME:
        return False
    return verify_password(password, _ADMIN_PASSWORD_HASH)


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """Create a signed JWT access token.

    Args:
        data: Claims to embed (e.g. ``{"sub": "admin"}``).
        expires_delta: Optional custom lifetime; defaults to the configured
            ``ACCESS_TOKEN_EXPIRE_MINUTES``.

    Returns:
        The encoded JWT as a string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def verify_token(token: str) -> dict:
    """Decode and validate a JWT, returning its payload.

    Raises:
        HTTPException: ``401`` if the token is invalid, expired or malformed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise CREDENTIALS_EXCEPTION from exc
    return payload


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency that resolves the current authenticated user.

    Args:
        token: Bearer token extracted by ``OAuth2PasswordBearer``.

    Returns:
        The username (``sub`` claim) of the authenticated user.

    Raises:
        HTTPException: ``401`` if the token is missing/invalid/expired.
    """
    payload = verify_token(token)
    username: Optional[str] = payload.get("sub")
    if username is None:
        raise CREDENTIALS_EXCEPTION
    return username
