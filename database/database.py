import sys
import os
import sqlite3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class PredictionDB:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    normal_prob REAL NOT NULL,
                    pneumonia_prob REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def log_prediction(self, filename, prediction, confidence, normal_prob, pneumonia_prob, model_used="custom_cnn"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (filename, prediction, confidence, normal_prob, pneumonia_prob, model_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, prediction, confidence, normal_prob, pneumonia_prob, model_used, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

    def fetch_all_predictions(self, limit=100):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, filename, prediction, confidence, normal_prob, pneumonia_prob, model_used, timestamp FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            return rows

    def get_stats(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*), AVG(confidence) FROM predictions')
            total, avg_conf = cursor.fetchone()
            total = total or 0
            avg_conf = avg_conf or 0.0

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='NORMAL'")
            normal_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='PNEUMONIA'")
            pneumonia_count = cursor.fetchone()[0] or 0

            return {
                "total": total,
                "normal": normal_count,
                "pneumonia": pneumonia_count,
                "avg_confidence": round(avg_conf, 2)
            }

    def clear_history(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions')
            conn.commit()
