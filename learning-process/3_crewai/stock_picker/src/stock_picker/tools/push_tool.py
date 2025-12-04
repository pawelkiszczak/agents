from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests


class PushNotificationInput(BaseModel):
    "A message to be sent to the user"
    message: str = Field(description='A message to be sent to the user')

class PushNotificationTool(BaseTool):
    name: str = "Send a push notification"
    description: str = (
        "This tool is used to send a push notification to the user"
    )
    args_schema: Type[BaseModel] = PushNotificationInput

    def _run(self, message: str) -> dict:
        pushover_user = os.getenv('PUSHOVER_USER')
        pushover_token = os.getenv('PUSHOVER_TOKEN')
        pushover_url = 'https://api.pushover.net/1/messages.json'
        
        print(f"Push: {message}")
        payload = {
            "user": pushover_user,
            "token": pushover_token,
            "message": str(message) if not isinstance(message, str) else message
        }
        
        response = requests.post(pushover_url, data=payload)
        
        print(f"Response: {response.status_code} | {response.text}")
        
        if response.status_code == 200:
            return {'notification': 'ok'}
        else:
            return {
                'notification': 'error', 
                'error_code': response.status_code,
                'error_message': f"{response.reason}\n{response.text}"}
