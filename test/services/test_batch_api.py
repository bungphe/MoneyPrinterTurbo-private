import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient

from app.asgi import app
from app.models import const
from app.services import batch
from app.services import video as vd
from app.utils import utils


def _make_video(path: str, seconds: int, frequency: int):
    subprocess.run(
        [
            vd.get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=540x960:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )


class TestBatchAndMergeApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.source_dir = utils.task_dir("api-test-source")
        cls.video_a = os.path.join(cls.source_dir, "a.mp4")
        cls.video_b = os.path.join(cls.source_dir, "b.mp4")
        _make_video(cls.video_a, 2, 440)
        _make_video(cls.video_b, 2, 550)
        cls.created_tasks = []

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.source_dir, ignore_errors=True)
        for task_id in cls.created_tasks:
            shutil.rmtree(utils.task_dir(task_id), ignore_errors=True)

    def _wait(self, task_id: str, timeout: int = 120) -> dict:
        self.created_tasks.append(task_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.client.get(f"/api/v1/tasks/{task_id}").json()["data"]
            if data.get("state") in (const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED):
                return data
            time.sleep(0.5)
        self.fail(f"task {task_id} did not finish in {timeout}s")

    def test_batch_with_merge(self):
        videos = {"Chủ đề A": self.video_a, "Chủ đề B": self.video_b}

        def fake_start(task_id, params):
            if params.video_subject == "Lỗi":
                raise RuntimeError("tts failed")
            return {"videos": [videos[params.video_subject]], "script": "kịch bản"}

        body = {
            "items": [
                {"video_subject": "Chủ đề A", "video_script_style": "ban_hang"},
                {"video_subject": "Lỗi"},
                {"video_subject": "Chủ đề B"},
            ],
            "settings": {"video_subject": "", "video_language": "vi-VN"},
            "merge_videos": True,
            "merge_transition": "crossfade",
        }
        with mock.patch.object(batch.tm, "start", side_effect=fake_start), mock.patch.object(
            batch.llm, "generate_post_metadata", return_value={"title": "Tiêu đề", "hashtags": ["#a"]}
        ):
            response = self.client.post("/api/v1/videos/batch", json=body)
            self.assertEqual(response.status_code, 200)
            data = self._wait(response.json()["data"]["task_id"])

        self.assertEqual(data["state"], const.TASK_STATE_COMPLETE)
        self.assertEqual(data["progress"], 100)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["success"], 2)
        self.assertEqual([r["status"] for r in data["results"]], ["thanh_cong", "that_bai", "thanh_cong"])
        self.assertEqual(data["results"][1]["error"], "tts failed")
        self.assertEqual(data["results"][0]["title"], "Tiêu đề")
        self.assertTrue(data["results"][0]["videos"][0].startswith("http://testserver/tasks/"))
        self.assertTrue(data["merged_video"].endswith("/tong_hop.mp4"))
        self.assertTrue(data["csv_file"].endswith("/ket_qua.csv"))
        merged_path = os.path.join(utils.task_dir(data["task_id"]), "tong_hop.mp4")
        duration, has_audio = vd._probe_media(merged_path)
        self.assertAlmostEqual(duration, 3, delta=0.15)  # 2 + 2 - 1 giây hòa cảnh
        self.assertTrue(has_audio)

    def test_merge_by_files_and_task_ids(self):
        response = self.client.post(
            "/api/v1/videos/merge",
            json={"video_files": ["tasks/api-test-source/a.mp4", self.video_b], "transition": "none"},
        )
        self.assertEqual(response.status_code, 200)
        first = self._wait(response.json()["data"]["task_id"])
        self.assertEqual(first["state"], const.TASK_STATE_COMPLETE)
        self.assertTrue(first["videos"][0].startswith("http://testserver/tasks/"))
        first_output = os.path.join(utils.task_dir(first["task_id"]), "tong_hop.mp4")
        self.assertAlmostEqual(vd._probe_media(first_output)[0], 4, delta=0.15)

        # nối tiếp kết quả của task trước bằng task_id, và dùng link do API trả về
        response = self.client.post(
            "/api/v1/videos/merge",
            json={"task_ids": [first["task_id"]], "video_files": [first["videos"][0]]},
        )
        self.assertEqual(response.status_code, 200)
        second = self._wait(response.json()["data"]["task_id"])
        self.assertEqual(second["state"], const.TASK_STATE_COMPLETE)
        second_output = os.path.join(utils.task_dir(second["task_id"]), "tong_hop.mp4")
        self.assertAlmostEqual(vd._probe_media(second_output)[0], 8, delta=0.2)

    def test_merge_rejects_files_outside_storage(self):
        for bad_path in ("../../etc/passwd", "/etc/passwd", "tasks/../../app/config/config.py"):
            response = self.client.post(
                "/api/v1/videos/merge", json={"video_files": [bad_path, self.video_a]}
            )
            self.assertEqual(response.status_code, 403, bad_path)

    def test_merge_validation(self):
        response = self.client.post("/api/v1/videos/merge", json={"video_files": [self.video_a]})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/v1/videos/merge", json={"task_ids": ["khong-ton-tai"]})
        self.assertEqual(response.status_code, 400)
        # app trả 400 cho dữ liệu không hợp lệ (bộ xử lý lỗi validation của dự án)
        response = self.client.post("/api/v1/videos/batch", json={"items": []})
        self.assertEqual(response.status_code, 400)

    def test_get_task_does_not_rewrite_stored_paths(self):
        from app.services import state as sm

        sm.state.update_task("api-test-paths", state=const.TASK_STATE_COMPLETE, videos=[self.video_a])
        self.created_tasks.append("api-test-paths")
        self.client.get("/api/v1/tasks/api-test-paths")
        self.assertEqual(sm.state.get_task("api-test-paths")["videos"], [self.video_a])


if __name__ == "__main__":
    unittest.main()
