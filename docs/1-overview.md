#  Overview

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

- [Overview](#overview)
  - [System Purpose](#system-purpose)
  - [Core Architecture](#core-architecture)
  - [Key Components and Code Mapping](#key-components-and-code-mapping)
    - [Component Relationships](#component-relationships)
  - [Data Flow](#data-flow)
    - [RAG Question-Answering Flow](#rag-question-answering-flow)
    - [Document Ingestion Flow](#document-ingestion-flow)
  - [Database Schema](#database-schema)
  - [Technology Stack](#technology-stack)
  - [API Endpoints Summary](#api-endpoints-summary)

# Overview

Relevant source files

* [README.md](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/README.md)
* [api/app.py](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py)

MARS-labs is an educational platform that combines AI-powered knowledge retrieval with personalized learning progress tracking. This document provides a high-level overview of the system architecture, key components, and how they interact.

For detailed information about specific components, refer to:

* Backend API details in [Backend API (app.py)](/JATAYU000/MARS-labs/2-backend-api-(app.py))
* User interface documentation in [User Interfaces](/JATAYU000/MARS-labs/3-user-interfaces)
* Database structure in [Database Architecture](/JATAYU000/MARS-labs/4-database-architecture)
* Retrieval system in [RAG System](/JATAYU000/MARS-labs/5-rag-system)

## System Purpose

MARS-labs serves as an AI-powered educational assistant that:

1. Provides context-aware responses to user queries using Retrieval-Augmented Generation (RAG)
2. Tracks user progress across various subjects and activities
3. Personalizes learning experiences based on user history and progress
4. Offers document ingestion capabilities for domain-specific knowledge

Sources: [api/app.py1-661](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L1-L661) [README.md1-79](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/README.md#L1-L79)

## Core Architecture

The MARS-labs architecture consists of four main layers:

1. **Client Layer**: Web interface for users to interact with the system
2. **Application Layer**: Flask-based API that manages requests, sessions, and orchestrates components
3. **Model Layer**: Language model integration with support for both Ollama and Gemini
4. **Storage Layer**: Persistent storage including ChromaDB for vector embeddings and SQLite for user data

Sources: [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92) [api/app.py95-273](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L95-L273) [api/app.py277-295](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L277-L295) [api/app.py303-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L303-L331)

## Key Components and Code Mapping

### Component Relationships

Each component in the system maps to specific code entities:

| Component | Code Entity | Location |
| --- | --- | --- |
| Chat Interface | Route `/chat` | [api/app.py456-460](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L456-L460) |
| Question Answering | Route `/ask` | [api/app.py510-561](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L510-L561) |
| Document Upload | Route `/upload` | [api/app.py471-507](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L471-L507) |
| RAG Engine | `Mentor` class | [api/app.py95-273](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L95-L273) |
| LLM Integration | `RemoteOllamaWrapper` & Gemini | [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92) [api/app.py231-255](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L231-L255) |
| User Management | `User` model & routes | [api/app.py303-317](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L303-L317) [api/app.py577-610](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L577-L610) |
| Progress Tracking | `Progress` model & routes | [api/app.py319-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L319-L331) [api/app.py400-453](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L400-L453) |

Sources: [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92) [api/app.py95-273](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L95-L273) [api/app.py303-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L303-L331) [api/app.py456-460](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L456-L460) [api/app.py471-507](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L471-L507) [api/app.py510-561](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L510-L561)

## Data Flow

### RAG Question-Answering Flow

This diagram shows how a user question flows through the system:

1. User submits a question through the chat interface
2. Flask API receives the request via the `/ask` endpoint
3. The `Mentor` class retrieves relevant context from ChromaDB
4. Based on the selected model (Ollama or Gemini), the query and context are sent to the appropriate LLM
5. The generated response is returned to the user

Sources: [api/app.py510-561](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L510-L561) [api/app.py207-229](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L207-L229)

### Document Ingestion Flow

This diagram illustrates the document ingestion process:

1. Admin uploads a document (PDF) through the admin interface
2. Flask API receives the file via the `/upload` endpoint
3. The `Mentor` class processes the document, splitting it into chunks
4. The chunks are embedded and stored in ChromaDB
5. Confirmation of successful ingestion is returned to the admin

Sources: [api/app.py471-507](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L471-L507) [api/app.py185-205](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L185-L205)

## Database Schema

The database schema consists of two main tables:

1. **User** - Stores basic user information
2. **Progress** - Contains user progress data in JSON format, allowing for flexible tracking of various subjects and activities

Sources: [api/app.py303-317](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L303-L317) [api/app.py319-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L319-L331)

## Technology Stack

MARS-labs leverages several key technologies:

| Component | Technology |
| --- | --- |
| Web Framework | Flask |
| Database | SQLAlchemy with SQLite |
| Vector Database | ChromaDB |
| Text Embeddings | HuggingFace e5-base |
| Language Models | Ollama (via Ngrok) and Google Gemini |
| Document Processing | LangChain (PyPDFLoader, RecursiveCharacterTextSplitter) |
| RAG Implementation | LangChain |

The system is designed with flexibility in mind, allowing for easy swapping between language models and future extensions.

Sources: [api/app.py1-32](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L1-L32) [api/app.py277-295](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L277-L295) [README.md47-70](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/README.md#L47-L70)

## API Endpoints Summary

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Main landing page |
| `/chat` | GET | Chat interface |
| `/ask` | POST | Process user questions |
| `/upload` | POST | Handle document uploads |
| `/progress` | POST | Add progress data |
| `/progress/<user_id>` | GET | Get user progress |
| `/progress/<user_id>` | PUT | Update user progress |
| `/users` | GET | Get user information |
| `/update` | POST | Create new user |
| `/admin` | GET | Admin interface |
| `/user/progress` | GET | User progress view |
| `/clear` | POST | Clear conversation history |

For detailed information about these endpoints, see [API Endpoints](/JATAYU000/MARS-labs/2.1-api-endpoints).

Sources: [api/app.py394-639](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L394-L639)