"""Local SQLite + bcrypt auth."""
import sqlite3
import bcrypt
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "users.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email    TEXT NOT NULL,
            pw_hash  BLOB NOT NULL,
            created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    return c


def sign_up(username: str, full_name: str, email: str, password: str) -> tuple[bool, str]:
    if not username or not password or len(password) < 6:
        return False, "Username required and password must be ≥ 6 chars."
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users(username, full_name, email, pw_hash) VALUES (?,?,?,?)",
                (username.strip().lower(), full_name.strip(), email.strip(), pw_hash),
            )
        return True, "Account created. You can sign in now."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def sign_in(username: str, password: str) -> tuple[bool, str | dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT username, full_name, email, pw_hash FROM users WHERE username=?",
            (username.strip().lower(),),
        ).fetchone()
    if not row:
        return False, "Invalid credentials."
    if not bcrypt.checkpw(password.encode(), row[3]):
        return False, "Invalid credentials."
    return True, {"username": row[0], "full_name": row[1], "email": row[2]}
