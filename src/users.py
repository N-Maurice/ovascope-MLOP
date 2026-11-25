"""User accounts and sign-in sessions (stored in the same SQLite database).

* Passwords are stored only as salted scrypt hashes (Python standard library, no extra dependency).
* Signing in returns a random session token; only its SHA-256 hash is stored, so a copy of the
  database cannot be used to sign in.
* Roles: "admin" (upload, retraining, model and user management) and "user" (signed in, no admin
  rights). The very first account becomes admin so a fresh installation can be set up from the
  web page; every later sign-up is a "user" until an admin promotes it.

Command line (e.g. on a server without a browser):
    python -m src.users create-admin --username alice --email alice@example.org
    python -m src.users reset-password --login alice
    python -m src.users list
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import connect, init_db
from src.registry import utc_now

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ROLES = ("admin", "user")
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
PUBLIC_FIELDS = "id, username, email, role, active, created_at, last_login"


class AccountError(Exception):
    """A user-facing reason why an account action failed (status: HTTP status to answer with)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate.hex(), digest)


_DUMMY_HASH = hash_password(secrets.token_hex(8))   # equalises timing for unknown accounts


def check_new_password(password: str, min_length: int) -> None:
    if len(password) < min_length:
        raise AccountError(f"Password must be at least {min_length} characters long.")
    if len(password) > 256:
        raise AccountError("Password is too long (maximum 256 characters).")
    if password.isdigit() or password.isalpha():
        raise AccountError("Password must contain both letters and digits or symbols.")


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def _public(row) -> dict:
    user = dict(row)
    user["active"] = bool(user["active"])
    return user


def count_users(db_path: Path, role: str | None = None, active_only: bool = False) -> int:
    query, args = "SELECT COUNT(*) FROM users WHERE 1=1", []
    if role:
        query, args = query + " AND role=?", [role]
    if active_only:
        query += " AND active=1"
    with connect(db_path) as db:
        return db.execute(query, args).fetchone()[0]


def create_user(db_path: Path, username: str, email: str, password: str, role: str | None = None,
                min_password_length: int = 8) -> dict:
    """Create an account. role=None: "admin" if no admin exists yet, otherwise "user"."""
    username, email = username.strip(), email.strip().lower()
    if not USERNAME_RE.match(username):
        raise AccountError("Username must be 3–32 characters: letters, digits, '.', '_' or '-'.")
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise AccountError("Please enter a valid email address.")
    check_new_password(password, min_password_length)
    if role is not None and role not in ROLES:
        raise AccountError(f"Unknown role '{role}'.")
    password_hash = hash_password(password)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN IMMEDIATE")   # the "first account becomes admin" check must be atomic
        if role is None:
            has_admin = connection.execute("SELECT 1 FROM users WHERE role='admin' AND active=1").fetchone()
            role = "user" if has_admin else "admin"
        clash = connection.execute("SELECT username, email FROM users WHERE username=? COLLATE NOCASE OR email=?",
                                   (username, email)).fetchone()
        if clash:
            connection.rollback()
            field = "username" if clash["username"].lower() == username.lower() else "email address"
            raise AccountError(f"An account with this {field} already exists.", status=409)
        cursor = connection.execute(
            "INSERT INTO users (username, email, password_hash, role, active, created_at) VALUES (?,?,?,?,1,?)",
            (username, email, password_hash, role, utc_now()))
        connection.commit()
        user_id = cursor.lastrowid
    finally:
        connection.close()
    return get_user(db_path, user_id)


def get_user(db_path: Path, user_id: int) -> dict | None:
    with connect(db_path) as db:
        row = db.execute(f"SELECT {PUBLIC_FIELDS} FROM users WHERE id=?", (user_id,)).fetchone()
    return _public(row) if row else None


def list_users(db_path: Path) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute(f"SELECT {PUBLIC_FIELDS} FROM users ORDER BY id").fetchall()
    return [_public(r) for r in rows]


def find_user(db_path: Path, login: str):
    """Look an account up by username (case-insensitive) or email."""
    login = login.strip()
    with connect(db_path) as db:
        return db.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE OR email=?",
                          (login, login.lower())).fetchone()


def authenticate(db_path: Path, login: str, password: str) -> dict:
    row = find_user(db_path, login)
    if row is None:
        verify_password(password, _DUMMY_HASH)   # same work as a real check: no timing hint
        raise AccountError("Incorrect username/email or password.", status=401)
    if not verify_password(password, row["password_hash"]):
        raise AccountError("Incorrect username/email or password.", status=401)
    if not row["active"]:
        raise AccountError("This account has been deactivated. Contact an administrator.", status=403)
    return get_user(db_path, row["id"])


def _guard_last_admin(db, user_id: int) -> None:
    others = db.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1 AND id<>?", (user_id,)).fetchone()[0]
    if others == 0:
        raise AccountError("This is the only active administrator; promote another account first.", status=409)


def update_user(db_path: Path, user_id: int, role: str | None = None, active: bool | None = None) -> dict:
    if role is not None and role not in ROLES:
        raise AccountError(f"Unknown role '{role}'.")
    with connect(db_path) as db:
        row = db.execute("SELECT role, active FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None:
            raise AccountError("User not found.", status=404)
        losing_admin = row["role"] == "admin" and row["active"] and (role == "user" or active is False)
        if losing_admin:
            _guard_last_admin(db, user_id)
        if role is not None:
            db.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
        if active is not None:
            db.execute("UPDATE users SET active=? WHERE id=?", (int(active), user_id))
            if not active:
                db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    return get_user(db_path, user_id)


def delete_user(db_path: Path, user_id: int) -> None:
    with connect(db_path) as db:
        row = db.execute("SELECT role, active FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None:
            raise AccountError("User not found.", status=404)
        if row["role"] == "admin" and row["active"]:
            _guard_last_admin(db, user_id)
        db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM users WHERE id=?", (user_id,))


def change_password(db_path: Path, user_id: int, current: str, new: str, min_length: int = 8) -> None:
    with connect(db_path) as db:
        row = db.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None or not verify_password(current, row["password_hash"]):
        raise AccountError("The current password is not correct.", status=401)
    set_password(db_path, user_id, new, min_length)


def set_password(db_path: Path, user_id: int, new: str, min_length: int = 8) -> None:
    check_new_password(new, min_length)
    with connect(db_path) as db:
        db.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(new), user_id))
        db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))   # sign out everywhere


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db_path: Path, user_id: int, days: float) -> dict:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(days=days)).isoformat(timespec="seconds")
    with connect(db_path) as db:
        db.execute("DELETE FROM sessions WHERE expires_at < ?", (now.isoformat(timespec="seconds"),))
        db.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                   (_token_hash(token), user_id, now.isoformat(timespec="seconds"), expires))
        db.execute("UPDATE users SET last_login=? WHERE id=?", (now.isoformat(timespec="seconds"), user_id))
    return {"token": token, "expires_at": expires}


def user_from_token(db_path: Path, token: str | None) -> dict | None:
    if not token:
        return None
    with connect(db_path) as db:
        row = db.execute(f"SELECT {', '.join('u.' + f.strip() for f in PUBLIC_FIELDS.split(','))} FROM sessions s"
                         " JOIN users u ON u.id = s.user_id"
                         " WHERE s.token_hash=? AND s.expires_at > ? AND u.active=1",
                         (_token_hash(token), utc_now())).fetchone()
    return _public(row) if row else None


def end_session(db_path: Path, token: str) -> None:
    with connect(db_path) as db:
        db.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))


class LoginThrottle:
    """Slows down password guessing: at most `limit` failures per login name in `window` seconds."""

    def __init__(self, limit: int = 5, window: float = 300):
        self.limit, self.window, self.failures = limit, window, {}

    def check(self, login: str) -> None:
        recent = [t for t in self.failures.get(login.lower(), []) if time.time() - t < self.window]
        self.failures[login.lower()] = recent
        if len(recent) >= self.limit:
            raise AccountError("Too many failed sign-in attempts. Wait a few minutes and try again.", status=429)

    def failed(self, login: str) -> None:
        self.failures.setdefault(login.lower(), []).append(time.time())

    def succeeded(self, login: str) -> None:
        self.failures.pop(login.lower(), None)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    import getpass

    from src.config import load_config
    parser = argparse.ArgumentParser(description="Manage OvaScope accounts")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-admin", help="create an administrator account")
    create.add_argument("--username", required=True)
    create.add_argument("--email", required=True)
    reset = sub.add_parser("reset-password", help="set a new password for an account")
    reset.add_argument("--login", required=True, help="username or email")
    sub.add_parser("list", help="list accounts")
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg["paths"]["database"]
    min_length = cfg.get("auth", {}).get("min_password_length", 8)
    init_db(db_path)
    try:
        if args.command == "list":
            for u in list_users(db_path):
                print(f"{u['id']:>3}  {u['username']:<20} {u['email']:<30} {u['role']:<6} "
                      f"{'active' if u['active'] else 'disabled'}")
        elif args.command == "create-admin":
            password = getpass.getpass("Password: ")
            if password != getpass.getpass("Repeat password: "):
                raise SystemExit("Passwords do not match.")
            user = create_user(db_path, args.username, args.email, password, role="admin", min_password_length=min_length)
            print(f"Created administrator '{user['username']}'.")
        elif args.command == "reset-password":
            row = find_user(db_path, args.login)
            if row is None:
                raise SystemExit("No such account.")
            password = getpass.getpass("New password: ")
            if password != getpass.getpass("Repeat password: "):
                raise SystemExit("Passwords do not match.")
            set_password(db_path, row["id"], password, min_length)
            print(f"Password changed for '{row['username']}'; existing sessions were signed out.")
    except AccountError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
