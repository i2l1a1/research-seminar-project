"""
locust -f tests/load_test.py --headless -u 3 -r 1 -t 30s --host http://127.0.0.1:8000
"""

from locust import HttpUser, between, task

FIXED_QUERY = "Кто был первым президентом США?"


class GenerateUser(HttpUser):
    wait_time = between(0.2, 0.8)

    @task
    def post_generate(self) -> None:
        self.client.post(
            "/v1/generate",
            json={"query": FIXED_QUERY},
            name="POST /v1/generate",
            headers={"Content-Type": "application/json"},
        )
