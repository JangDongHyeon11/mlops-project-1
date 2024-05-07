from locust import HttpUser, task, between, TaskSet

class ApiBehavior(TaskSet):
    @task(1)  # Weight of 1, means this is equally likely to be picked as the other task
    def extract_keywords(self):
        payload = {
            "query": "스포츠 동영상을 실시간 스트리밍으로 입력하면, 해당 스포츠의 주요한 장면을 적절한 시점에 인지 및 분류하는 모델을 만드는 연구에 도전하고 싶습니다. 입력 동영상이 미리 적절하게 편집된 동영상 클립이 아니고, 실시간 스트리밍 영상이기 때문에 훨씬 난이도가 높은 연구주제에 해당합니다."
        }
        self.client.post("/extract_keywords", json=payload)

    @task(1)
    def search(self):
        payload = {
            "options": ["video", "real-time streaming", "recognizing and classifying key moments in sports", "sports video analysis"]
        }
        self.client.post("/search", json=payload)

class ApiUser(HttpUser):
    tasks = [ApiBehavior]
    wait_time = between(1, 2)  # Simulate a wait of 1-2 seconds between requests