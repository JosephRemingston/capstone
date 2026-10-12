import json
import os
from pathlib import Path
from unittest import mock
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.settings = Settings.for_testing(Path(self.tmp.name))
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.tmp.cleanup()

    def signup(self, email="user@example.com", password="a-very-strong-password"):
        response = self.client.post("/api/v1/auth/signup", json={"email": email, "password": password})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["data"]["access_token"]

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_auth_memory_isolation_and_standard_envelope(self):
        token = self.signup()
        other = self.signup("other@example.com")
        created = self.client.post(
            "/api/v1/memories", headers=self.auth(token),
            json={"content": "I prefer concise explanations.", "session_id": "s1"},
        )
        self.assertEqual(created.status_code, 201, created.text)
        memory_id = created.json()["data"]["id"]
        self.assertTrue(created.json()["success"])

        mine = self.client.get("/api/v1/memories", headers=self.auth(token))
        self.assertEqual([x["id"] for x in mine.json()["data"]], [memory_id])
        forbidden_by_ownership = self.client.get(f"/api/v1/memories/{memory_id}", headers=self.auth(other))
        self.assertEqual(forbidden_by_ownership.status_code, 404)

        unauthenticated = self.client.get("/api/v1/memories")
        self.assertEqual(unauthenticated.status_code, 401)
        self.assertFalse(unauthenticated.json()["success"])
        self.assertIn("request_id", unauthenticated.json())

    def test_validation_metrics_health_and_demo(self):
        bad = self.client.post("/api/v1/auth/signup", json={"email": "bad", "password": "short"})
        self.assertEqual(bad.status_code, 422)
        self.assertFalse(bad.json()["success"])
        self.assertEqual(bad.json()["error"]["code"], "validation_error")

        live = self.client.get("/health/live")
        self.assertEqual(live.status_code, 200)
        ready = self.client.get("/health/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["data"]["status"], "ready")
        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("cognimem_http_requests_total", metrics.text)
        demo = self.client.get("/demo/")
        self.assertEqual(demo.status_code, 200)
        self.assertIn("CogniMem", demo.text)

    def test_export_and_deletion_remove_persisted_data(self):
        token = self.signup()
        self.client.post("/api/v1/memories", headers=self.auth(token), json={"content": "My private note", "session_id": "s1"})
        exported = self.client.get("/api/v1/me/export?format=json", headers=self.auth(token))
        self.assertEqual(exported.status_code, 200)
        payload = json.loads(exported.content)
        self.assertEqual(payload["account"]["email"], "user@example.com")
        self.assertEqual(payload["memories"][0]["content"], "My private note")

        deleted = self.client.delete("/api/v1/me", headers=self.auth(token))
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(deleted.json()["data"]["memories_deleted"], 1)
        self.assertEqual(self.client.get("/api/v1/me", headers=self.auth(token)).status_code, 401)
        self.assertNotIn("My private note", self.settings.store_path.read_text())

    def test_keyword_search_is_user_scoped(self):
        token = self.signup()
        other = self.signup("other@example.com")
        self.client.post("/api/v1/memories", headers=self.auth(token), json={"content": "Project Atlas", "session_id": "s1"})
        self.client.post("/api/v1/memories", headers=self.auth(other), json={"content": "Project Atlas secret", "session_id": "s1"})
        result = self.client.post("/api/v1/memories/search", headers=self.auth(token), json={"query": "Atlas", "mode": "keyword"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()["data"]), 1)
        self.assertEqual(result.json()["data"][0]["content"], "Project Atlas")


class SettingsFromEnvTests(unittest.TestCase):
    SECRET = {"COGNIMEM_JWT_SECRET": "x" * 40}

    def settings(self, **env):
        with mock.patch.dict(os.environ, {**self.SECRET, **env}, clear=True):
            return Settings.from_env()

    def test_retrieval_defaults_are_exact_search_and_runtime_threads(self):
        settings = self.settings()
        self.assertFalse(settings.ann_enabled)
        self.assertEqual(settings.ann_exact_threshold, 5000)
        self.assertIsNone(settings.candidate_pool)
        self.assertIsNone(settings.model_threads)

    def test_retrieval_overrides_are_parsed(self):
        settings = self.settings(COGNIMEM_ANN_ENABLED="on", COGNIMEM_ANN_EXACT_THRESHOLD="1000",
                                 COGNIMEM_CANDIDATE_POOL="50", COGNIMEM_MODEL_THREADS="4")
        self.assertEqual((settings.ann_enabled, settings.ann_exact_threshold, settings.candidate_pool,
                          settings.model_threads), (True, 1000, 50, 4))

    def test_invalid_retrieval_settings_fail_at_config_time(self):
        # Threshold 0 used to pass config and then make the module-level `app` silently None.
        for env in ({"COGNIMEM_ANN_EXACT_THRESHOLD": "0"}, {"COGNIMEM_ANN_EXACT_THRESHOLD": "-5"},
                    {"COGNIMEM_ANN_ENABLED": "maybe"}, {"COGNIMEM_CANDIDATE_POOL": "201"},
                    {"COGNIMEM_MODEL_THREADS": "0"}, {"COGNIMEM_MODEL_THREADS": "two"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.settings(**env)


if __name__ == "__main__":
    unittest.main()


class ModuleImportTests(unittest.TestCase):
    def test_configured_service_with_bad_setting_fails_loudly(self):
        import os
        import subprocess
        import sys
        env = {**os.environ, 'COGNIMEM_JWT_SECRET': 'x' * 48, 'COGNIMEM_ANN_EXACT_THRESHOLD': '0'}
        result = subprocess.run([sys.executable, '-c', 'import api.app'], env=env, capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent.parent)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('COGNIMEM_ANN_EXACT_THRESHOLD', result.stderr)
        unconfigured = {key: value for key, value in env.items() if key != 'COGNIMEM_JWT_SECRET'}
        result = subprocess.run([sys.executable, '-c', 'import api.app; assert api.app.app is None'],
                                env=unconfigured, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
