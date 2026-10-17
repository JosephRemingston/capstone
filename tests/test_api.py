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


class AskEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = TestClient(create_app(Settings.for_testing(Path(self.tmp.name))))
        token = self.client.post("/api/v1/auth/signup", json={"email": "u@example.com", "password": "a-very-strong-password"}).json()["data"]["access_token"]
        self.headers = {"Authorization": f"Bearer {token}"}
        for text in ("I live in Chennai.", "I moved to Bengaluru."):
            self.client.post("/api/v1/memories", headers=self.headers, json={"content": text, "session_id": "s"})

    def test_extractive_answer_cites_the_current_fact_only(self):
        response = self.client.post("/api/v1/memories/ask", headers=self.headers,
                                    json={"query": "Where do I live?", "mode": "keyword", "rerank": False})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertFalse(data["abstain"])
        self.assertEqual(data["generator"], "extractive")
        text = " ".join(s["text"] for s in data["statements"])
        self.assertIn("Bengaluru", text)
        self.assertNotIn("Chennai", text)
        self.assertTrue(all(e["source_id"] in data["sources"] for s in data["statements"] for e in s["evidence"]))

    def test_validation_auth_isolation_and_provider_outage(self):
        ask = lambda body, headers=None: self.client.post("/api/v1/memories/ask", headers=self.headers if headers is None else headers, json=body)
        self.assertEqual(ask({"query": "x"}, headers={}).status_code, 401)
        self.assertEqual(ask({"query": "  "}).status_code, 422)
        blank_search = self.client.post("/api/v1/memories/search", headers=self.headers, json={"query": "  "})
        self.assertEqual(blank_search.status_code, 422)   # was an unserialisable 500
        self.assertEqual(blank_search.json()["error"]["code"], "validation_error")
        self.assertEqual(ask({"query": "x", "mode": "recency"}).status_code, 422)
        self.assertEqual(ask({"query": "x", "as_of": "not-a-date"}).status_code, 422)
        self.assertEqual(ask({"query": "x", "generator": "gemini", "abstain": True}).status_code, 422)
        other = self.client.post("/api/v1/auth/signup", json={"email": "o@example.com", "password": "another-strong-password"}).json()["data"]["access_token"]
        empty = ask({"query": "Where do I live?", "mode": "keyword"}, headers={"Authorization": f"Bearer {other}"}).json()["data"]
        self.assertTrue(empty["abstain"])
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}), \
                mock.patch("main.retrieval.generation.load_dotenv", create=True), \
                mock.patch("dotenv.load_dotenv"):
            response = ask({"query": "Where do I live?", "mode": "keyword", "rerank": False, "generator": "gemini"})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn("GOOGLE_API_KEY", response.text)


class HistoryAndGraphEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = TestClient(create_app(Settings.for_testing(Path(self.tmp.name))))
        self.headers = self.login("u@example.com")
        self.created = [self.client.post("/api/v1/memories", headers=self.headers,
                                         json={"content": text, "session_id": "s"}).json()["data"]
                        for text in ("I live in Chennai.", "I moved to Bengaluru.")]

    def login(self, email):
        token = self.client.post("/api/v1/auth/signup", json={"email": email, "password": "a-very-strong-password"}).json()["data"]["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_history_lists_revisions_and_is_private(self):
        old = self.created[0]["id"]
        response = self.client.get(f"/api/v1/memories/{old}/history", headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        statuses = [r["memory_status"] for r in response.json()["data"]]
        self.assertEqual(statuses[0], "active")
        self.assertEqual(statuses[-1], "superseded")
        other = self.login("o@example.com")
        self.assertEqual(self.client.get(f"/api/v1/memories/{old}/history", headers=other).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/memories/missing/history", headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get(f"/api/v1/memories/{old}/history").status_code, 401)

    def test_graph_relations_query_validation_and_isolation(self):
        response = self.client.get("/api/v1/graph", headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        relations = response.json()["data"]["relations"]
        self.assertTrue(any(r["object_label"] == "Bengaluru" for r in relations), relations)
        self.assertFalse(any(r["object_label"] == "Chennai" for r in relations))
        queried = self.client.get("/api/v1/graph", headers=self.headers, params={"query": "Bengaluru", "hops": 1}).json()["data"]
        self.assertIn("entities", queried)
        self.assertIn("paths", queried)
        self.assertEqual(self.client.get("/api/v1/graph", headers=self.headers, params={"as_of": "nope"}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/graph", headers=self.headers, params={"hops": 9}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/graph").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/graph", headers=self.login("o@example.com")).json()["data"]["relations"], [])


class EvaluationEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        reports = Path(self.tmp.name) / "reports"
        reports.mkdir()
        for tag, created, value in (("baseline", "2026-10-12T01:00:00", 0.5), ("final", "2026-10-12T18:00:00", 0.9)):
            (reports / f"{tag}_summary.json").write_text(json.dumps({
                "sections": {"longmemeval": {}, "_scorers": {}},
                "flat": {"longmemeval.all.memory_retention": value, "lifecycle.passed": 16, "note": "text", "flag": True},
                "manifest": {"tag": tag, "created_at": created, "git_head": "abc", "argv": ["secret-argv"]}}))
        (reports / "retention_summary.json").write_text(json.dumps({"ensemble_macro_f1": 0.89}))
        (reports / "retention_ensemble.json").write_text(json.dumps({"model": "m", "class_values": [0.3, 0.6, 0.8], "golden": {"x": 1}}))
        (Path(self.tmp.name) / "outside_summary.json").write_text(json.dumps({"flat": {"leak": 1}}))
        patcher = mock.patch.dict(os.environ, {"COGNIMEM_REPORTS_DIR": str(reports)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(Settings.for_testing(Path(self.tmp.name))))
        token = self.client.post("/api/v1/auth/signup", json={"email": "e@example.com", "password": "a-very-strong-password"}).json()["data"]["access_token"]
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_lists_snapshots_oldest_first_and_skips_non_snapshots(self):
        response = self.client.get("/api/v1/evaluation/snapshots", headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        rows = response.json()["data"]
        self.assertEqual([r["tag"] for r in rows], ["baseline", "final"])
        self.assertEqual(rows[0]["sections"], ["longmemeval"])
        self.assertEqual(rows[0]["metrics"], 2)

    def test_snapshot_returns_numeric_metrics_and_safe_manifest(self):
        data = self.client.get("/api/v1/evaluation/snapshots/final", headers=self.headers).json()["data"]
        self.assertEqual(data["flat"], {"longmemeval.all.memory_retention": 0.9, "lifecycle.passed": 16})
        self.assertNotIn("argv", data["manifest"])
        self.assertEqual(data["manifest"]["git_head"], "abc")

    def test_unknown_or_malicious_tags_are_404_and_auth_is_required(self):
        for tag in ("missing", "retention", "..%2Foutside", "../outside", "a" * 65):
            with self.subTest(tag=tag):
                self.assertEqual(self.client.get(f"/api/v1/evaluation/snapshots/{tag}", headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/evaluation/snapshots").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/evaluation/model").status_code, 401)

    def test_model_report_omits_internal_details(self):
        data = self.client.get("/api/v1/evaluation/model", headers=self.headers).json()["data"]
        self.assertEqual(data["class_values"], [0.3, 0.6, 0.8])
        self.assertNotIn("golden", data)


class WebAppMountTests(unittest.TestCase):
    def test_built_app_is_served_with_spa_fallback_and_no_traversal(self):
        with TemporaryDirectory() as tmp:
            web = Path(tmp) / "dist"
            (web / "assets").mkdir(parents=True)
            (web / "index.html").write_text("<div id=root>INDEX</div>")
            (web / "assets" / "app.js").write_text("console.log(1)")
            (Path(tmp) / "secret.txt").write_text("SECRET")
            with mock.patch.dict(os.environ, {"COGNIMEM_WEB_DIR": str(web)}):
                client = TestClient(create_app(Settings.for_testing(Path(tmp))))
            for path in ("/app", "/app/", "/app/memories/123", "/app/..%2Fsecret.txt", "/app/%2E%2E/secret.txt"):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("INDEX", response.text)
                    self.assertNotIn("SECRET", response.text)
            self.assertEqual(client.get("/app/assets/app.js").text, "console.log(1)")
            self.assertEqual(client.get("/health/live").status_code, 200)

    def test_missing_build_leaves_app_unmounted(self):
        with TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"COGNIMEM_WEB_DIR": str(Path(tmp) / "absent")}):
            client = TestClient(create_app(Settings.for_testing(Path(tmp))))
        self.assertEqual(client.get("/app").status_code, 404)


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
