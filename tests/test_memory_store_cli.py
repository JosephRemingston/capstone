from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import LocalMemoryStore, MemoryCore, MemoryInput
from main.__main__ import main as cli_main


class LocalMemoryStoreTests(unittest.TestCase):
    def test_save_list_get_and_search_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalMemoryStore(Path(temp_dir) / "memories.jsonl")
            core = MemoryCore()
            preference = store.save(
                core.process(
                    MemoryInput(
                        content="I prefer concise Python explanations.",
                        user_id="u1",
                        session_id="s1",
                    )
                )
            )
            task = store.save(
                core.process(
                    MemoryInput(
                        content="Remind me to review the capstone report tomorrow.",
                        user_id="u1",
                        session_id="s2",
                    )
                )
            )
            store.save(
                core.process(
                    MemoryInput(
                        content="hello",
                        user_id="u2",
                        session_id="s3",
                    )
                )
            )

            self.assertEqual(store.get(preference.id), preference)
            self.assertEqual(len(store.list(user_id="u1")), 2)
            self.assertEqual(store.list(category="task"), [task])
            self.assertEqual(store.search("python", user_id="u1"), [preference])


class CliTests(unittest.TestCase):
    def run_cli(self, argv: list[str]) -> tuple[int, object]:
        output = io.StringIO()
        with redirect_stdout(output):
            status = cli_main(argv)
        return status, json.loads(output.getvalue())

    def test_process_saves_and_list_reads_from_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = str(Path(temp_dir) / "memories.jsonl")

            status, created = self.run_cli(
                [
                    "--store",
                    store_path,
                    "process",
                    "I prefer concise technical summaries.",
                    "--user-id",
                    "u1",
                    "--session-id",
                    "s1",
                    "--interaction-score",
                    "0.5",
                ]
            )

            self.assertEqual(status, 0)
            self.assertEqual(created["category"], "preference")
            self.assertEqual(created["tier"], "long_term")

            status, records = self.run_cli(["--store", store_path, "list", "--user-id", "u1"])

            self.assertEqual(status, 0)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["id"], created["id"])

    def test_get_missing_record_returns_error_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status, payload = self.run_cli(
                [
                    "--store",
                    str(Path(temp_dir) / "memories.jsonl"),
                    "get",
                    "missing-id",
                ]
            )

            self.assertEqual(status, 1)
            self.assertEqual(payload["error"], "record_not_found")


if __name__ == "__main__":
    unittest.main()
