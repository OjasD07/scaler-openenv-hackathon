from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from email_triage_env.baseline import run_baseline
from email_triage_env.server.app import app
from email_triage_env.tasks import DATASET


class OpenEnvSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_manifest_and_tasks(self) -> None:
        manifest_response = self.client.get("/manifest")
        self.assertEqual(manifest_response.status_code, 200)
        manifest = manifest_response.json()
        self.assertEqual(manifest["name"], "email_triage_env")
        self.assertEqual(manifest["task_count"], 3)
        self.assertEqual(manifest["supported_tools"], ["lookup_order", "check_payment", "get_user_history"])
        self.assertEqual(manifest["dataset_summary"]["dataset_size"], len(DATASET))
        label_fields = {"category", "priority", "department", "action"}
        self.assertTrue(label_fields.isdisjoint(manifest["dataset_summary"]["sample_email"]))

        tasks_response = self.client.get("/tasks")
        self.assertEqual(tasks_response.status_code, 200)
        tasks = tasks_response.json()
        self.assertEqual(tasks["task_count"], 3)
        self.assertEqual(len(tasks["tasks"]), 3)

    def test_reset_step_cycle(self) -> None:
        reset_response = self.client.post("/reset", json={"task_id": 1, "seed": 7})
        self.assertEqual(reset_response.status_code, 200)
        payload = reset_response.json()
        observation = payload["observation"]
        self.assertIn("current_email", observation)
        self.assertGreaterEqual(observation["remaining_emails"], 1)
        label_fields = {"category", "priority", "department", "action"}
        self.assertTrue(label_fields.isdisjoint(observation["current_email"]))
        self.assertTrue(label_fields.isdisjoint(payload["state"]["email_data"]))

        action = {
            "category": "spam",
            "priority": "low",
            "department": "ignore",
            "action": "archive",
        }
        step_response = self.client.post("/step", json={"action": action})
        self.assertEqual(step_response.status_code, 200)
        step_payload = step_response.json()
        self.assertIn("reward", step_payload)
        self.assertIn("observation", step_payload)
        self.assertTrue(label_fields.isdisjoint(step_payload["observation"]["current_email"]))
        self.assertTrue(label_fields.isdisjoint(step_payload["state"]["email_data"]))

    def test_baseline_quality(self) -> None:
        scores = run_baseline()
        self.assertGreaterEqual(scores.task_1, 0.9)
        self.assertGreaterEqual(scores.task_2, 0.9)
        self.assertGreaterEqual(scores.task_3, 0.9)
        self.assertGreaterEqual(scores.average, 0.9)

    def test_grader_hides_component_feedback_by_default(self) -> None:
        self.client.post("/reset", json={"task_id": 3, "email_id": "em-001"})
        response = self.client.post(
            "/grader",
            json={
                "task_id": 3,
                "email_id": "em-001",
                "action": {
                    "category": "billing",
                    "priority": "high",
                    "department": "finance",
                    "action": "reply",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("score", payload)
        self.assertNotIn("breakdown", payload["details"])
        self.assertEqual(payload["details"]["component_feedback"], "hidden")


if __name__ == "__main__":
    unittest.main()
