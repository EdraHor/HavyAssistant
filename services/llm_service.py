"""
Сервис работы с Gemini API
С улучшенной обработкой ошибок
"""

import logging
import requests
from typing import Optional
from utils.config_loader import config
from utils.database import ConversationDatabase

logger = logging.getLogger(__name__)

class GeminiService:
    """Клиент для Gemini API"""

    def __init__(self):
        self.api_key = config.get('gemini.api_key')
        self.model = config.get('gemini.model', 'gemini-2.0-flash-exp')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

        # Настройки генерации
        self.temperature = config.get('gemini.temperature', 0.7)
        self.max_tokens = config.get('gemini.max_output_tokens', 200)

        # База данных для истории
        try:
            self.db = ConversationDatabase()
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}", exc_info=True)
            self.db = None

        self.current_session_id = None

        # История диалога (в памяти для быстрого доступа)
        self.conversation_history = []

        # Прокси
        self.proxies = None
        if config.get('gemini.proxy.enabled', False):
            try:
                proxy_type = config.get('gemini.proxy.type', 'socks5')
                proxy_host = config.get('gemini.proxy.host')
                proxy_port = config.get('gemini.proxy.port')

                proxy_url = f"{proxy_type}://{proxy_host}:{proxy_port}"
                self.proxies = {
                    'http': proxy_url,
                    'https': proxy_url
                }
                logger.info(f"Используется прокси: {proxy_url}")
            except Exception as e:
                logger.error(f"Ошибка настройки прокси: {e}")

        # Системный промпт
        self.system_prompt = """Ты голосовой ассистент. 
Отвечай максимально кратко и по существу - 1-2 предложения. 
Без лишних вводных слов. Прямо и понятно. Используй разговорный стиль.
Запоминай контекст разговора и отвечай с учетом предыдущих сообщений."""

        # Callbacks
        self.response_callback = None
        self.error_callback = None

        # Загружаем последнюю сессию или создаем новую
        self._init_session()

        logger.debug("GeminiService инициализирован")

    def _init_session(self):
        """Инициализация или загрузка сессии"""
        try:
            if not self.db:
                logger.warning("БД недоступна, создаю временную сессию")
                self.current_session_id = "temp"
                return

            last_session = self.db.get_latest_session()

            if last_session:
                self.current_session_id = last_session
                self.conversation_history = self.db.load_session_history(last_session)
                msg_count = len(self.conversation_history)
                logger.info(f"✓ Загружена сессия {last_session} с {msg_count} сообщениями")
            else:
                self.current_session_id = self.db.create_session()
                logger.info(f"✓ Создана новая сессия {self.current_session_id}")

        except Exception as e:
            logger.error(f"Ошибка инициализации сессии: {e}", exc_info=True)
            self.current_session_id = "temp"

    def send_query(self, user_message: str) -> Optional[str]:
        """Отправить запрос в Gemini с учетом истории"""
        try:
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

            # Формируем contents с учетом истории
            contents = []

            # Системный промпт только если история пустая
            if not self.conversation_history:
                contents.append({
                    "role": "user",
                    "parts": [{"text": self.system_prompt}]
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": "Понял, буду отвечать кратко и по существу, учитывая контекст разговора."}]
                })

            # Добавляем историю
            for msg in self.conversation_history:
                contents.append(msg)

            # Добавляем текущее сообщение
            contents.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": self.max_tokens
                }
            }

            logger.info(f"Запрос к Gemini: {user_message} (история: {len(self.conversation_history)//2} сообщений)")

            response = requests.post(
                url,
                json=payload,
                proxies=self.proxies,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                if 'candidates' in data and len(data['candidates']) > 0:
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    logger.info(f"✅ Ответ Gemini: {text}")

                    # Сохраняем в БД
                    if self.db:
                        try:
                            self.db.save_message(self.current_session_id, "user", user_message)
                            self.db.save_message(self.current_session_id, "model", text)
                        except Exception as e:
                            logger.error(f"Ошибка сохранения в БД: {e}")

                    # Обновляем кэш в памяти
                    self.conversation_history.append({
                        "role": "user",
                        "parts": [{"text": user_message}]
                    })
                    self.conversation_history.append({
                        "role": "model",
                        "parts": [{"text": text}]
                    })

                    if self.response_callback:
                        try:
                            self.response_callback(text)
                        except Exception as e:
                            logger.error(f"Ошибка в response_callback: {e}")

                    return text
                else:
                    error = "Получен пустой ответ от Gemini"
                    logger.error(error)
                    if self.error_callback:
                        try:
                            self.error_callback(error)
                        except Exception as e:
                            logger.error(f"Ошибка в error_callback: {e}")
                    return None
            else:
                error = f"Ошибка API: {response.status_code} - {response.text}"
                logger.error(error)
                if self.error_callback:
                    try:
                        self.error_callback(error)
                    except Exception as e:
                        logger.error(f"Ошибка в error_callback: {e}")
                return None

        except requests.exceptions.ProxyError as e:
            error = "🔒 Ошибка подключения к прокси-серверу.\n\nПроверьте настройки прокси в config/settings.yaml"
            logger.error(f"ProxyError: {e}", exc_info=True)
            if self.error_callback:
                try:
                    self.error_callback(error)
                except Exception as e:
                    logger.error(f"Ошибка в error_callback: {e}")
            return None

        except requests.exceptions.ConnectionError as e:
            error = "❌ Нет подключения к интернету!\n\nПроверьте:\n• Подключение к сети\n• Настройки прокси\n• Firewall"
            logger.error(f"ConnectionError: {e}", exc_info=True)
            if self.error_callback:
                try:
                    self.error_callback(error)
                except Exception as e:
                    logger.error(f"Ошибка в error_callback: {e}")
            return None

        except requests.exceptions.Timeout as e:
            error = "⏱️ Превышено время ожидания ответа от Gemini.\n\nПопробуйте:\n• Проверить интернет\n• Повторить запрос"
            logger.error(f"Timeout: {e}", exc_info=True)
            if self.error_callback:
                try:
                    self.error_callback(error)
                except Exception as e:
                    logger.error(f"Ошибка в error_callback: {e}")
            return None

        except Exception as e:
            error = f"Ошибка запроса: {str(e)}"
            logger.error(f"Unexpected error: {e}", exc_info=True)
            if self.error_callback:
                try:
                    self.error_callback(error)
                except Exception as e:
                    logger.error(f"Ошибка в error_callback: {e}")
            return None

    def clear_history(self):
        """Очистить текущую сессию и создать новую"""
        try:
            if self.db:
                self.current_session_id = self.db.create_session()
            else:
                self.current_session_id = "temp"

            self.conversation_history.clear()
            logger.info(f"✓ Создана новая сессия {self.current_session_id}")

        except Exception as e:
            logger.error(f"Ошибка очистки истории: {e}", exc_info=True)

    def get_history_length(self) -> int:
        """Получить количество пар сообщений"""
        try:
            return len(self.conversation_history) // 2
        except Exception as e:
            logger.error(f"Ошибка получения длины истории: {e}")
            return 0

    def set_response_callback(self, callback):
        """Установить callback для ответа"""
        self.response_callback = callback

    def set_error_callback(self, callback):
        """Установить callback для ошибки"""
        self.error_callback = callback

    def __del__(self):
        """Закрытие БД при удалении объекта"""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
        except Exception as e:
            logger.error(f"Ошибка в __del__: {e}")