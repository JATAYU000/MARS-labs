#  Backend Api (Apppy)

## Quick Navigation

- [overview](1-overview.md)
- [backend-api-(app.py)](2-backend-api-(app.py).md)
  - [api-endpoints](2.1-api-endpoints.md)
  - [mentor-class](2.2-mentor-class.md)
- [user-interfaces](3-user-interfaces.md)
  - [chat-interface](3.1-chat-interface.md)
  - [progress-dashboard](3.2-progress-dashboard.md)
  - [admin-interface](3.3-admin-interface.md)
- [database-architecture](4-database-architecture.md)
  - [user-and-progress-models](4.1-user-and-progress-models.md)
  - [database-migrations](4.2-database-migrations.md)
- [rag-system](5-rag-system.md)
  - [vector-database](5.1-vector-database.md)
  - [language-models](5.2-language-models.md)
- [deployment-and-configuration](6-deployment-and-configuration.md)
  - [dependencies](6.1-dependencies.md)
  - [audio-processing-features](6.2-audio-processing-features.md)

## Table of Contents

- [Backend API (app.py)](#backend-api-apppy)
  - [Purpose and Scope](#purpose-and-scope)
  - [System Architecture](#system-architecture)
    - [Backend API Components](#backend-api-components)
    - [Request Flow](#request-flow)
  - [Core Components](#core-components)
    - [Flask Application](#flask-application)
    - [RemoteOllamaWrapper](#remoteollamawrapper)
    - [Mentor Class](#mentor-class)
    - [Database Models](#database-models)
  - [RAG Implementation](#rag-implementation)
  - [Language Model Integration](#language-model-integration)
  - [Session Management](#session-management)
  - [API Endpoints](#api-endpoints)
  - [Database Operations](#database-operations)
  - [Helper Functions](#helper-functions)
  - [Initialization and Startup](#initialization-and-startup)

# Backend API (app.py)

Relevant source files

* [api/app.py](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py)

## Purpose and Scope

This document details the core Flask application that serves as the backend of the MARS-labs system. The app.py file implements the server-side logic, including the RESTful API endpoints, database models, and Retrieval-Augmented Generation (RAG) functionality with multiple language model options. For specific details about the API endpoints, see [API Endpoints](/JATAYU000/MARS-labs/2.1-api-endpoints), and for more information about the Mentor class, see [Mentor Class](/JATAYU000/MARS-labs/2.2-mentor-class).

Sources: [api/app.py1-661](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L1-L661)

## System Architecture

The backend API is built on Flask and serves as the central coordinator for the entire MARS-labs system. It integrates multiple components: a RAG pipeline, user data management, and progress tracking functionality.

### Backend API Components

Sources: [api/app.py1-33](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L1-L33) [api/app.py276-291](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L276-L291)

### Request Flow

Sources: [api/app.py394-639](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L394-L639)

## Core Components

### Flask Application

The Flask application is initialized with SQLite database configuration using SQLAlchemy. It supports multiple database binds and includes configuration for file uploads.

```
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
app.config['SQLALCHEMY_BINDS'] = {
    'db1': 'sqlite:///database1.db',
    'db2': 'sqlite:///database2.db'
}

```

Sources: [api/app.py276-284](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L276-L284)

### RemoteOllamaWrapper

The `RemoteOllamaWrapper` class encapsulates interactions with a remote Ollama instance through ngrok tunneling.

The `invoke()` method sends prompts to the remote Ollama API and processes the streaming responses into a consolidated text output.

Sources: [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92)

### Mentor Class

The `Mentor` class is the core component that manages the RAG pipeline. It handles document ingestion, vector store management, and query processing.

The Mentor class can use either Ollama or Gemini models for generating responses, and it manages a ChromaDB vector store for document retrieval.

Sources: [api/app.py95-273](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L95-L273)

### Database Models

The app defines two primary database models:

The Progress model uses a JSON field to store flexible progress data structure for different subjects and activities.

Sources: [api/app.py303-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L303-L331)

## RAG Implementation

The RAG (Retrieval-Augmented Generation) system is implemented through the following components:

1. **Document Ingestion**: PDF documents are uploaded via the `/upload` endpoint, processed by the Mentor's `ingest()` method, split into chunks, and stored in the ChromaDB vector database.
2. **Query Processing**: User questions sent to the `/ask` endpoint are processed by retrieving relevant document chunks from the vector store, formatting them with the question into a prompt, and sending the combined context to the selected language model.

Sources: [api/app.py185-205](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L185-L205) [api/app.py207-229](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L207-L229) [api/app.py470-507](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L470-L507) [api/app.py510-561](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L510-L561)

## Language Model Integration

The system supports two LLM options:

1. **Ollama**: Accessed via a remote instance through ngrok tunneling using the `RemoteOllamaWrapper` class.
2. **Gemini**: Accessed through Google's Generative AI API.

Model selection is managed during Mentor initialization and can be changed at runtime through the `/ask` endpoint's model parameter.

Sources: [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92) [api/app.py134-145](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L134-L145) [api/app.py219-225](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L219-L225) [api/app.py231-255](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L231-L255)

## Session Management

User sessions are managed through Flask's session mechanism and a custom `ensure_session()` function:

Each user session is assigned a unique ID, which is associated with a dedicated Mentor instance stored in the `chat_instances` dictionary. Old instances are periodically cleaned up by the `cleanup_old_instances()` function registered with `@app.before_request`.

Sources: [api/app.py340-352](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L340-L352) [api/app.py642-653](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L642-L653)

## API Endpoints

The Flask application exposes several key endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Render the main application page |
| `/chat` | GET | Render the chat interface |
| `/admin` | GET | Render the admin interface |
| `/user/progress` | GET | Render the progress tracking page |
| `/ask` | POST | Process user questions via the RAG system |
| `/upload` | POST | Ingest documents into the vector store |
| `/upload_npy` | POST | Upload pre-generated embeddings |
| `/clear` | POST | Clear conversation history |
| `/users` | GET | Retrieve user information |
| `/update` | POST | Create a new user |
| `/progress/<user_id>` | GET | Get progress data for a user |
| `/progress/<user_id>` | PUT | Update progress data for a user |
| `/progress` | POST | Add new progress data for a user |

Sources: [api/app.py394-639](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L394-L639)

## Database Operations

The application uses SQLAlchemy to interact with SQLite databases. Key operations include:

1. **User Management**:

   * Creating users (`/update` endpoint)
   * Retrieving user information (`/users` endpoint)
2. **Progress Tracking**:

   * Adding progress data (`/progress` POST endpoint)
   * Retrieving progress data (`/progress/<user_id>` GET endpoint)
   * Updating progress data (`/progress/<user_id>` PUT endpoint)

The Progress model stores data in a flexible JSON structure that can accommodate different subject areas and activity types.

Sources: [api/app.py303-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L303-L331) [api/app.py400-453](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L400-L453) [api/app.py577-610](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L577-L610)

## Helper Functions

Several helper functions support the core functionality:

1. `ensure_session()`: Manages user sessions and associated Mentor instances.
2. `process_npy_files()`: Processes pre-generated embeddings and text chunks for insertion into the vector database.
3. `cleanup_old_instances()`: Removes chat instances that aren't associated with active sessions.

Sources: [api/app.py340-352](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L340-L352) [api/app.py355-390](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L355-L390) [api/app.py642-653](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L642-L653)

## Initialization and Startup

During application startup, the following actions occur:

1. Database tables are created using `db.create_all()` within the app context.
2. When running the application directly (`if __name__ == '__main__'`), the Flask development server starts with debugging enabled.

Sources: [api/app.py656-660](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L656-L660)