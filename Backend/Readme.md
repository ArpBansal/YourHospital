Here’s your updated README with the installation steps in place of the "Dependencies" section:

---

# HealthcareBackend

HealthcareBackend is a Django-based backend project for managing healthcare-related data and operations. This project provides APIs for handling patient records, appointments, and other healthcare services.

## Features

- User authentication and authorization with Email OTP verification
- Patient record form and management
- AI-powered chat assistant using WebSockets and Gemini API
- API endpoints for CRUD operations
- Admin interface for managing data

## Prerequisites

Before you begin, ensure you have atleast met the following requirements:

- Python 3.6 or higher
- Django 5.1.4 or higher
- pip (Python package installer)
- Git (for version control)

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ayush-github123/Health-Backend.git
    cd HealthcareBackend
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Set up the environment variables:**
    Create a `.env` file in the `HealthcareBackend` directory and add the following:
    ```plaintext
    SECRET_KEY='your-secret-key'
    DEBUG=True
    GOOGLE_CLIENT_ID='your-google-client-id'
    GOOGLE_SECRET='your-google-secret'
    GEMINI_API_KEY='your-gemini-api-key'
    EMAIL_HOST_USER='your-email'
    EMAIL_HOST_PASSWORD='your-email-password'
    ```

6. **Set up the database:**

    ```bash
    python manage.py migrate
    ```

7. **Create a superuser:**

    ```bash
    python manage.py createsuperuser
    ```

    Follow the prompts to create a superuser account.

8. **Run the development server:**

    ```bash
    python manage.py runserver
    ```

    The server will start at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Usage

### User Authentication
- **Register**: `POST /auth/register/`
- **Verify OTP**: `POST /auth/verify-otp/`
- **Resend OTP**: `POST /auth/resend-otp/`
- **Login**: `POST /auth/login/`
- **Logout**: `POST /auth/logout/`

### Health Form Management
- **Submit Health Form**: `POST /healthcare/form/submit/`
- **Retrieve User’s Health Form**: `GET /healthcare/form/me/`
- **Update User’s Health Form**: `PATCH /healthcare/form/me/update/`

### AI Chat Assistant
- **WebSocket Endpoint**: `ws://127.0.0.1:8000/ws/chat/`

## Testing

To test the API endpoints, you can use the `test.rest` file with the REST Client extension in VSCode.

1. **Install the REST Client extension** in VSCode.
2. **Open the `test.rest` file** in VSCode.
3. **Send requests** by clicking on the "Send Request" links in the `test.rest` file.

The `test.rest` file contains predefined requests for various API endpoints, making it easy to test the functionality of the backend.

## Deployment

To deploy this project to a production environment, follow these steps:

1. **Set up a production server** (e.g., DigitalOcean, AWS, Heroku).
2. **Install the necessary software** (e.g., Python, pip, virtualenv).
3. **Clone the repository** to the server.
4. **Create and activate a virtual environment**.
5. **Install the required packages** using `pip install -r requirements.txt`.
6. **Set up the database** and apply migrations.
7. **Configure the server** (e.g., using Gunicorn and Nginx).
8. **Set up environment variables** for sensitive data (e.g., SECRET_KEY, DATABASE_URL).
9. **Run the server** in production mode.

## Contributing

To contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

If you have any questions or need further assistance, feel free to contact:

- Ayush Rai - [ayushrai31593@gmail.com](mailto:ayushrai31593@gmail.com)
- Project Link: [https://github.com/ayush-github123/Health-Backend](https://github.com/ayush-github123/Health-Backend)

---
