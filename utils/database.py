"""
Менеджер базы данных для хранения истории диалогов (thread-safe)
С улучшенной обработкой ошибок
"""

import sqlite3
import logging
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ConversationDatabase:
    """Управление историей диалогов в SQLite (thread-safe)"""

    def __init__(self, db_path: str = "data/conversations.db"):
        self.db_path = db_path
        self.lock = threading.Lock()  # Блокировка для thread-safety

        try:
            # Создаем папку data если не существует
            Path("data").mkdir(exist_ok=True)

            self._init_database()
            logger.debug("✓ ConversationDatabase инициализирован")

        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}", exc_info=True)
            raise

    def _get_connection(self):
        """Получить новое соединение (для каждого потока)"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            return conn
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}", exc_info=True)
            raise

    def _init_database(self):
        """Инициализация базы данных"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Таблица сессий
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    name TEXT DEFAULT 'Новая сессия'
                )
            """)

            # Таблица сообщений
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)

            # Таблица для изображений (для будущего)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    image_data BLOB,
                    image_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            """)

            conn.commit()
            conn.close()
            logger.info(f"✓ База данных инициализирована: {self.db_path}")

        except Exception as e:
            logger.error(f"Ошибка инициализации таблиц БД: {e}", exc_info=True)
            raise

    def create_session(self, name: str = None) -> int:
        """Создать новую сессию (thread-safe)"""
        with self.lock:
            try:
                if name is None:
                    name = f"Сессия {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "INSERT INTO sessions (name) VALUES (?)",
                    (name,)
                )
                conn.commit()
                session_id = cursor.lastrowid
                conn.close()

                logger.info(f"✓ Создана новая сессия: {session_id} - {name}")
                return session_id

            except Exception as e:
                logger.error(f"Ошибка создания сессии: {e}", exc_info=True)
                return -1

    def get_latest_session(self) -> Optional[int]:
        """Получить ID последней сессии (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1"
                )
                result = cursor.fetchone()
                conn.close()

                return result[0] if result else None

            except Exception as e:
                logger.error(f"Ошибка получения последней сессии: {e}", exc_info=True)
                return None

    def save_message(self, session_id: int, role: str, content: str, metadata: Dict = None):
        """Сохранить сообщение (thread-safe)"""
        with self.lock:
            try:
                metadata_json = json.dumps(metadata) if metadata else None

                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, metadata_json)
                )
                conn.commit()
                conn.close()

                logger.debug(f"✓ Сохранено сообщение: {role} - {content[:50]}...")

            except Exception as e:
                logger.error(f"Ошибка сохранения сообщения: {e}", exc_info=True)

    def load_session_history(self, session_id: int) -> List[Dict]:
        """Загрузить историю сессии (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT role, content, timestamp, metadata 
                    FROM messages 
                    WHERE session_id = ? 
                    ORDER BY timestamp ASC
                    """,
                    (session_id,)
                )

                messages = []
                for row in cursor.fetchall():
                    try:
                        role, content, timestamp, metadata = row

                        # Формат для Gemini API
                        messages.append({
                            "role": role,
                            "parts": [{"text": content}]
                        })
                    except Exception as e:
                        logger.warning(f"Ошибка обработки сообщения: {e}")
                        continue

                conn.close()
                logger.info(f"✓ Загружено {len(messages)} сообщений из сессии {session_id}")
                return messages

            except Exception as e:
                logger.error(f"Ошибка загрузки истории сессии: {e}", exc_info=True)
                return []

    def get_session_message_count(self, session_id: int) -> int:
        """Получить количество сообщений в сессии (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (session_id,)
                )
                count = cursor.fetchone()[0]
                conn.close()

                return count

            except Exception as e:
                logger.error(f"Ошибка подсчета сообщений: {e}", exc_info=True)
                return 0

    def clear_session(self, session_id: int):
        """Очистить сообщения сессии (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.commit()
                conn.close()

                logger.info(f"✓ Сессия {session_id} очищена")

            except Exception as e:
                logger.error(f"Ошибка очистки сессии: {e}", exc_info=True)

    def delete_session(self, session_id: int):
        """Удалить сессию и все сообщения (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                conn.close()

                logger.info(f"✓ Сессия {session_id} удалена")

            except Exception as e:
                logger.error(f"Ошибка удаления сессии: {e}", exc_info=True)

    def get_all_sessions(self) -> List[Dict]:
        """Получить список всех сессий (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT s.id, s.name, s.created_at, COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN messages m ON s.id = m.session_id
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    """
                )

                sessions = []
                for row in cursor.fetchall():
                    try:
                        sessions.append({
                            'id': row[0],
                            'name': row[1],
                            'created_at': row[2],
                            'message_count': row[3]
                        })
                    except Exception as e:
                        logger.warning(f"Ошибка обработки сессии: {e}")
                        continue

                conn.close()
                return sessions

            except Exception as e:
                logger.error(f"Ошибка получения списка сессий: {e}", exc_info=True)
                return []

    def save_image(self, message_id: int, image_data: bytes, image_type: str = "png"):
        """Сохранить изображение (thread-safe)"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "INSERT INTO images (message_id, image_data, image_type) VALUES (?, ?, ?)",
                    (message_id, image_data, image_type)
                )
                conn.commit()
                conn.close()

                logger.debug(f"✓ Изображение сохранено для сообщения {message_id}")

            except Exception as e:
                logger.error(f"Ошибка сохранения изображения: {e}", exc_info=True)

    def close(self):
        """Закрытие (нет постоянного соединения)"""
        try:
            logger.info("✓ БД использует локальные соединения - закрытие не требуется")
        except Exception as e:
            logger.error(f"Ошибка при закрытии: {e}")