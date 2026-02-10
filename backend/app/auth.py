import os
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from dotenv import load_dotenv

load_dotenv()

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

security = HTTPBearer(auto_error=False)


def verify_supabase_token(token: str) -> dict:
    """Decode and verify Supabase JWT; return payload (sub, email, etc.)."""
    if not SUPABASE_JWT_SECRET:
        # Auth not configured – behave as anonymous.
        raise HTTPException(status_code=503, detail="Auth not configured (SUPABASE_JWT_SECRET)")
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True},
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Strict version – used only where auth is mandatory."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )
    payload = verify_supabase_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: no sub")
    return {"sub": user_id, "email": payload.get("email"), "raw_payload": payload}


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    Lenient version for read‑only endpoints.

    - If auth not configured → return None.
    - If no Authorization header → return None.
    - If token invalid/expired → return None.
    """
    if not SUPABASE_JWT_SECRET:
        return None
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = jwt.decode(
            credentials.credentials,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True},
        )
    except jwt.InvalidTokenError:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None
    return {"sub": user_id, "email": payload.get("email"), "raw_payload": payload}
