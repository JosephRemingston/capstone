from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import LocalMemoryStore, MemoryCore, MemoryInput
from main.__main__ import build_parser, main as cli_main


class LocalMemoryStoreTests(unittest.TestCase):
    def test_save_list_get_and_search_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalMemoryStore(Path(temp_dir) / "memories.jsonl")
            core = MemoryCore.with_heuristic()
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
                    "--scorer",
                    "heuristic",
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


class CliRetrievalOptionTests(unittest.TestCase):
    def test_global_ann_flag_defaults_to_environment(self) -> None:
        parser = build_parser()
        self.assertIsNone(parser.parse_args(["search", "q", "--mode", "hybrid"]).ann)
        self.assertTrue(parser.parse_args(["--ann", "search", "q", "--mode", "hybrid"]).ann)

    def test_rag_commands_honour_global_ann_flag(self) -> None:
        from unittest import mock
        from main.retrieval import rag as rag_module
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Path(temp_dir) / "memories.jsonl"
            captured = {}

            class StubRAG:
                def __init__(self, store, **options):
                    captured.update(options)

                def search(self, query, **kwargs):
                    captured["limit"] = kwargs["limit"]
                    return []

            clean_env = {k: v for k, v in os.environ.items() if not k.startswith("COGNIMEM_")}
            with mock.patch.object(rag_module, "MemoryRAG", StubRAG), \
                    mock.patch.dict(os.environ, clean_env, clear=True), redirect_stdout(io.StringIO()):
                code = cli_main(["--store", str(store), "--ann", "search", "q", "--mode", "hybrid",
                                 "--user-id", "u", "--limit", "80"])
            self.assertEqual(code, 0)
            self.assertTrue(captured["ann_enabled"])
            self.assertEqual(captured["ann_exact_threshold"], 5000)
            self.assertEqual(captured["limit"], 80)


if __name__ == "__main__":
    unittest.main()
