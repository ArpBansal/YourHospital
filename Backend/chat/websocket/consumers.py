import json
from channels.generic.websocket import AsyncWebsocketConsumer
import google.generativeai as genai
from django.conf import settings
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HealthcareBackend.settings")
django.setup()

from urllib.parse import parse_qs
from rest_framework_simplejwt.tokens import AccessToken
from django.contrib.auth.models import AnonymousUser
from channels.db import database_sync_to_async

# Initialize Gemini API AFTER Django settings are ready
API_KEY = getattr(settings, "GEMINI_API_KEY", None)

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    raise ValueError("GEMINI_API_KEY is not set in settings.py")

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = await self.authenticate_user()

        if not self.user or isinstance(self.user, AnonymousUser):
            print("WebSocket REJECTED: Authentication failed")
            await self.close() # Reject Connection
        print(f"WebSocket CONNECTED: {self.user.username}")

        await self.accept()

        previous_messages = await self.get_previous_messages(self.user)
    
        # Send previous messages to the client
        await self.send(text_data=json.dumps({
            "previous_messages": previous_messages
        }))

    @database_sync_to_async
    def get_previous_messages(self, user):
        """Retrieve last 20 messages for the user"""
        from chat.models import ChatMessage  # Import to avoid circular import issues
        messages = ChatMessage.objects.filter(user=user).order_by("-timestamp")[:20]

        return [
            {"message": msg.message, "response": msg.response, "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
            for msg in messages
        ]

    async def disconnect(self, code):
        pass

    async def receive(self, text_data):
        """Receives message from WebSocket and processes it."""
        data = json.loads(text_data)
        user_message = data.get("message", "")
        message_id = data.get("message_id", None)  # Get message_id from client

        if not user_message:
            await self.send(text_data=json.dumps({"error": "Message cannot be empty"}))
            return

        # Create empty chat message to store in database
        chat = await self.create_chat_message(self.user, user_message, "")

        # Stream the AI-generated response
        response_text = ""
        async for chunk in self.get_gemini_response(user_message):
            response_text += chunk
            # Include message_id in response to help client track which message this belongs to
            await self.send(json.dumps({
                "message": chunk, 
                "message_id": message_id,
                "streaming": True
            }))

        # Save the final AI response
        await self.update_chat_response(chat, response_text)
        
        # DO NOT send the full response again - this is what's causing the duplication
        # The streaming chunks above are sufficient

    async def get_gemini_response(self, user_message):
        """Stream response from Gemini API word-by-word."""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_message, stream=True)  # Enable streaming
        for chunk in response:
            yield chunk.text  # Send partial response

    @database_sync_to_async
    def create_chat_message(self, user, message, response):
        """Store user message."""
        from chat.models import ChatMessage
        return ChatMessage.objects.create(user=user, message=message, response=response)

    @database_sync_to_async
    def update_chat_response(self, chat, response):
        """Update AI response after streaming completes."""
        chat.response = response
        chat.save()

    async def authenticate_user(self):
        """Extract JWT token and authenticate user asynchronously."""
        query_params = parse_qs(self.scope["query_string"].decode())
        token = query_params.get("token", [None])[0]  # Extract JWT token

        if not token:
            print("No JWT token found in WebSocket URL")
            return AnonymousUser()

        try:
            access_token = AccessToken(token)  # Validate JWT
            user = await self.get_user(access_token["user_id"])  # Use async user retrieval
            if user:
                print(f"Authenticated user: {user.username}")
                return user
            else:
                print("User not found")
                return AnonymousUser()
        except Exception as e:
            print(f"JWT Authentication failed: {e}")  # Log error
            return AnonymousUser()

    @database_sync_to_async
    def get_user(self, user_id):
        from users.models import CustomUser  # Import here to avoid circular imports
        """Run database query asynchronously to fetch the user."""
        try:
            return CustomUser.objects.get(id=user_id)
        except CustomUser.DoesNotExist:
            return None