from locust import HttpUser, task, between
import json


class APIUser(HttpUser):
    host = "http://127.0.0.1:8082"
    wait_time = between(0.5, 1.5)

    @task
    def messages(self):
        payload = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": "Hello, world!"
            }]
        }
        self.client.post(
            "/v1/messages",
            json=payload,
            headers={"Content-Type": "application/json"}
        )