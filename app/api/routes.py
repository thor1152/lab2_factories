from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
import os
import json

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str
    store_email: bool = False
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

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

def _data_file_path(filename: str) -> str:
    app_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(app_root, "data", filename)

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
        result = inference_service.classify_email(email)

         # OPTIONAL: store the email (new)
        if request.store_email:
            factory = FeatureGeneratorFactory()

            # Only need embeddings for storage, not every generator
            feat = factory.generate_all_features(email, generator_names=["email_embeddings"])
            embedding = feat.get("email_embeddings_average_embedding")
            if embedding is None:
                raise ValueError("Missing email_embeddings_average_embedding (check embedding generator output name).")

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
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

#post endpoint to add new topic
'''Endpoint to add new topics'''
@router.post("/topics")
async def create_topics(request: TopicAddRequest):
    try:
        add_topic(request.topic, request.description)

        return {
            "message": "Topic added successfully",
            "topic": request.topic
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

