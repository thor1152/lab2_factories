from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal

from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
from app.features.factory import FeatureGeneratorFactory

import os
import json

router = APIRouter()


class EmailRequest(BaseModel):
    subject: str
    body: str
    mode: Literal["topic", "similar_email"] = "topic"
    store_email: bool = True
    ground_truth: Optional[str] = None


class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str


class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]
    stored_email_id: Optional[int] = None


class EmailAddResponse(BaseModel):
    message: str
    email_id: int


class TopicAddRequest(BaseModel):
    topic: str
    description: str


def _data_file_path(filename: str) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(project_root, "data", filename)


def _load_json(filename: str, default):
    path = _data_file_path(filename)
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def _save_json(filename: str, data) -> None:
    path = _data_file_path(filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
            
        result = inference_service.classify_email(email, mode=request.mode)
        # store email updates    
        stored_email_id: Optional[int] = None

        if request.store_email:
            factory = FeatureGeneratorFactory()
            feat = factory.generate_all_features(email, generator_names=["email_embeddings"])

            embedding = feat.get("email_embeddings_average_embedding")
            if embedding is None:
                raise ValueError(
                    "Missing email_embeddings_average_embedding. "
                    "Check EmailEmbeddingsFeatureGenerator feature names."
                )

            emails: List[Dict[str, Any]] = _load_json("emails.json", default=[])
            if not isinstance(emails, list):
                emails = []

            next_id = max((e.get("id", 0) for e in emails), default=0) + 1
            emails.append(
                {
                    "id": next_id,
                    "subject": request.subject,
                    "body": request.body,
                    "ground_truth": request.ground_truth,
                    "embedding": embedding,
                }
            )

            _save_json("emails.json", emails)
            stored_email_id = next_id

        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"],
            stored_email_id=stored_email_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics")
async def topics():
    """Get available email topics (from the model pipeline info)."""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}


@router.post("/topics")
async def create_topics(request: TopicAddRequest):
   #add new topic
    try:
        topics = _load_json("topic_keywords.json", default={})
        if not isinstance(topics, dict):
            topics = {}

        topics[request.topic] = {"description": request.description}
        _save_json("topic_keywords.json", topics)

        return {"message": "Topic added successfully", "topic": request.topic}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/info")
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()