#!/usr/bin/env python3
"""SQLite persistence server for vLLM chat conversations.

Provides REST API for conversation CRUD operations with SQLite backend.
Serves static files from ui/ directory for the Web UI.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import urllib.parse

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "conversations.db"


def init_db():
    """Initialize SQLite database with WAL mode for concurrent access."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
    conn.commit()
    conn.close()


def checkpoint_db():
    """Checkpoint WAL to main database for clean sync."""
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()


class ConversationHandler(SimpleHTTPRequestHandler):
    """HTTP handler for conversation API and static file serving."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/conversations":
            self._list_conversations()
        elif parsed.path.startswith("/api/conversations/"):
            conv_id = parsed.path.split("/")[-1]
            self._get_conversation(conv_id)
        elif parsed.path == "/api/checkpoint":
            self._checkpoint()
        else:
            # Serve static files from ui/
            self.directory = "ui"
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/conversations":
            self._create_conversation()
        elif self.path.startswith("/api/conversations/") and self.path.endswith("/messages"):
            conv_id = self.path.split("/")[-2]
            self._add_message(conv_id)

    def do_DELETE(self):
        if self.path.startswith("/api/conversations/"):
            conv_id = self.path.split("/")[-1]
            self._delete_conversation(conv_id)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_cors_headers(self):
        """Add CORS headers for local development."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _list_conversations(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT c.id, c.title, c.model, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """)
        conversations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        self._json_response(conversations)

    def _get_conversation(self, conv_id):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        conv = cursor.fetchone()
        if not conv:
            conn.close()
            self._error_response(404, "Conversation not found")
            return

        cursor = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()

        result = dict(conv)
        result["messages"] = messages
        self._json_response(result)

    def _create_conversation(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        conv_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        title = body.get("title", "New Conversation")
        model = body.get("model")

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, title, model, now, now)
        )
        conn.commit()
        conn.close()

        self._json_response({
            "id": conv_id,
            "title": title,
            "model": model,
            "created_at": now,
            "updated_at": now,
            "messages": []
        }, 201)

    def _add_message(self, conv_id):
        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length))

        msg_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(DB_PATH)

        # Verify conversation exists
        cursor = conn.execute("SELECT id, title FROM conversations WHERE id = ?", (conv_id,))
        conv = cursor.fetchone()
        if not conv:
            conn.close()
            self._error_response(404, "Conversation not found")
            return

        # Insert message
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, conv_id, body["role"], body["content"], now)
        )

        # Auto-title from first user message
        if conv[1] == "New Conversation" and body["role"] == "user":
            new_title = body["content"][:50] + ("..." if len(body["content"]) > 50 else "")
            conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (new_title, now, conv_id))
        else:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

        conn.commit()
        conn.close()

        self._json_response({
            "id": msg_id,
            "conversation_id": conv_id,
            "role": body["role"],
            "content": body["content"],
            "created_at": now
        })

    def _delete_conversation(self, conv_id):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()
        self._json_response({"deleted": conv_id})

    def _checkpoint(self):
        checkpoint_db()
        self._json_response({"status": "checkpointed"})

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _error_response(self, status, message):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


if __name__ == "__main__":
    init_db()
    print(f"Database: {DB_PATH}")
    server = HTTPServer(("0.0.0.0", 8080), ConversationHandler)
    print("Conversation server running on http://0.0.0.0:8080")
    server.serve_forever()
