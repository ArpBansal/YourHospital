# API Documentation for Frontend Integration

## Overview
This document outlines all the backend functionalities and API endpoints implemented in the project. It provides the necessary details for the frontend team to integrate with the backend seamlessly.

---

## **Authentication Module** (JWT-based + OTP Verification)

### **1. User Registration (OTP-based)**
**Endpoint:** `POST /auth/register/`

**Request Body:**
```json
{
    "username": "user6",
    "email": "nvl0t@vvatxiy.com",
    "password":"ayush@1234",
    "confirm_password":"ayush@1234"
}
```

**Response:**
- Returns success message and triggers an OTP to the provided email.

### **2. Verify OTP**
**Endpoint:** `POST /auth/verify-otp/`

**Request Body:**
```json
{
    "email": "nvl0t@vvatxiy.com",
    "otp": "260365"
}
```

**Response:**
- Returns success message upon correct OTP verification.

### **3. Resend OTP** (90-second timer enforced)
**Endpoint:** `POST /auth/resend-otp/`

**Request Body:**
```json
{
    "email": "nvl0t@vvatxiy.com"
}
```

**Response:**
- Returns success message confirming OTP resend.

### **4. User Login**
**Endpoint:** `POST /auth/login/`

**Request Body:**
```json
{
    "username":"user3",
    "password":"ayush@1234"
}
```

**Response:**
- Returns JWT access and refresh tokens.

---

## **Healthcare Module**

### **1. Submit Health Form**
**Endpoint:** `POST /healthcare/form/submit/`

**Headers:**
```
Authorization: Bearer <JWT_ACCESS_TOKEN>
Content-Type: application/json
```

**Request Body:**
```json
{
    "name": "User4",
    "age": 60,
    "gender": "Male",
    "contact_details": "9876543240",
    "chronic_conditions": "Arthritis, High Cholesterol",
    "past_surgeries": "Hip Replacement",
    "allergies": "Penicillin",
    "medications": "Atorvastatin, Ibuprofen",
    "symptoms": "Joint pain and stiffness",
    "symptom_severity": "Moderate",
    "symptom_duration": "Chronic",
    "mental_health_stress": false,
    "mental_health_anxiety": false,
    "mental_health_depression": false,
    "vaccination_history": "Covid-19, Pneumonia",
    "accessibility_needs": "Hearing aid support",
    "pregnancy_status": "Not Applicable",
    "emergency_contact": {
      "name": "Emma Green",
      "relationship": "Daughter",
      "number": "4376543241"
    },
    "health_insurance_provider": "WellCare",
    "health_insurance_policy": "MN123456",
    "preferred_language": "English",
    "research_participation": false
}
```

**Response:**
- Returns a confirmation that the health form was successfully submitted.

### **2. Retrieve User’s Health Form**
**Endpoint:** `GET /healthcare/form/me/`

**Headers:**
```
Authorization: Bearer <JWT_ACCESS_TOKEN>
Content-Type: application/json
```

**Response:**
- Returns the user’s submitted health form details.

### **3. Update User’s Health Form**
**Endpoint:** `PATCH /healthcare/form/me/update/`

**Headers:**
```
Authorization: Bearer <JWT_ACCESS_TOKEN>
Content-Type: application/json
```

**Request Body:**
```json
{
    "name":"User1"
}
```

**Response:**
- Returns updated health form details.

---

## **Chat Module**

### **1. AI Chat (WebSockets + Gemini API Integration)**
**Endpoint:** `ws://127.0.0.1:8000/chat/ai/`

**Description:**
- Real-time AI chat is implemented using WebSockets.
- Integrated with Gemini API for AI-powered responses.

**Expected Message Format:**
```json
{
    "message": "Hello, AI!"
}
```

**Response:**
- AI-generated response from Gemini API in real time.

---

## **Notes for Frontend Integration:**
- JWT authentication is required for all healthcare endpoints.
- OTP registration flow must be handled with a 90-second timer before allowing OTP resend.
- AI chat must use WebSockets for real-time messaging.
- Ensure all request bodies match the formats provided above.

