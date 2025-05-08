#  Deployment And Configuration

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

- [Deployment and Configuration](#deployment-and-configuration)
  - [System Requirements](#system-requirements)
    - [Hardware Requirements](#hardware-requirements)
    - [Software Dependencies](#software-dependencies)
  - [Environment Setup](#environment-setup)
    - [Python Environment](#python-environment)
    - [Environment Variables](#environment-variables)
- [Database Configuration](#database-configuration)
- [LLM Configuration](#llm-configuration)
- [Ollama Configuration (if using Ollama)](#ollama-configuration-if-using-ollama)
- [Or for remote Ollama server through Ngrok](#or-for-remote-ollama-server-through-ngrok)
- [ChromaDB Configuration](#chromadb-configuration)
- [Server Configuration](#server-configuration)
  - [Deployment Architecture](#deployment-architecture)
    - [System Deployment Components](#system-deployment-components)
    - [Configuration Components](#configuration-components)
  - [Configuration Options](#configuration-options)
    - [Database Configuration](#database-configuration)
    - [LLM Configuration](#llm-configuration)
      - [Ollama Configuration](#ollama-configuration)
      - [Gemini Configuration](#gemini-configuration)
    - [Vector Database Configuration](#vector-database-configuration)
  - [Deployment Steps](#deployment-steps)
    - [Local Development Deployment](#local-development-deployment)
    - [Production Deployment](#production-deployment)
    - [LLM Backend Setup](#llm-backend-setup)
      - [Local Ollama Setup](#local-ollama-setup)
      - [Remote Ollama Setup](#remote-ollama-setup)
      - [Gemini Setup](#gemini-setup)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)
    - [Logs and Debugging](#logs-and-debugging)
  - [Security Considerations](#security-considerations)
  - [Data Loading and Initialization](#data-loading-and-initialization)
    - [Document Loading Process](#document-loading-process)

# Deployment and Configuration

Relevant source files

* [.gitignore](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/.gitignore)
* [requirements2.txt](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/requirements2.txt)

This document provides detailed instructions for deploying and configuring the MARS-labs educational platform. It covers system requirements, environment setup, configuration options, and deployment procedures for both development and production environments.

For information about specific dependencies, see [Dependencies](/JATAYU000/MARS-labs/6.1-dependencies). For details on audio processing setup, see [Audio Processing Features](/JATAYU000/MARS-labs/6.2-audio-processing-features).

## System Requirements

### Hardware Requirements

Based on the components used in MARS-labs, the following hardware specifications are recommended:

| Component | Minimum Requirement | Recommended |
| --- | --- | --- |
| CPU | Dual-core 2.0 GHz | Quad-core 2.5+ GHz |
| RAM | 4 GB | 8+ GB |
| Storage | 1 GB free space | 5+ GB free space |
| Network | Broadband connection | High-speed connection |

### Software Dependencies

MARS-labs requires Python 3.8+ and numerous dependencies listed in the requirements file. Key dependencies include:

| Category | Key Dependencies |
| --- | --- |
| Web Framework | Flask (3.1.0), Werkzeug (3.1.3) |
| Database | SQLAlchemy (2.0.38), Alembic (1.14.1), ChromaDB (0.6.3) |
| AI/ML | LangChain (0.3.19), HuggingFace Hub (0.29.1), Google Generative AI (0.8.4) |
| Vector Embeddings | sentence-transformers (3.4.1), fastembed (0.6.0) |
| Utilities | python-dotenv (1.0.1), PyYAML (6.0.2) |

Sources: [requirements2.txt1-179](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/requirements2.txt#L1-L179)

## Environment Setup

### Python Environment

1. Create a virtual environment:

2. Activate the virtual environment:

3. Install dependencies:

### Environment Variables

Create a `.env` file in the root directory with the following configuration:

```
# Database Configuration
DATABASE_URL=sqlite:///instance/app.db

# LLM Configuration
LLM_TYPE=gemini  # or "ollama"
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL=http://localhost:11434
# Or for remote Ollama server through Ngrok
OLLAMA_BASE_URL=https://your-ngrok-url.ngrok.io

# ChromaDB Configuration
CHROMA_DB_DIR=./chroma_db

# Server Configuration
FLASK_APP=app.py
FLASK_ENV=development  # or "production"

```

Sources: [.gitignore6](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/.gitignore#L6-L6)

## Deployment Architecture

### System Deployment Components

Sources: Deployment Architecture diagram from provided information

### Configuration Components

Sources: Component Relationship diagram from provided information

## Configuration Options

### Database Configuration

The SQLite database is configured using SQLAlchemy and initialized using Alembic migrations. The default location is `instance/app.db`.

To initialize the database:

### LLM Configuration

#### Ollama Configuration

To use Ollama:

1. Set `LLM_TYPE=ollama` in the `.env` file
2. Configure the Ollama URL:
   * Local: `OLLAMA_BASE_URL=http://localhost:11434`
   * Remote: `OLLAMA_BASE_URL=https://your-ngrok-url.ngrok.io`

#### Gemini Configuration

To use Google's Gemini API:

1. Set `LLM_TYPE=gemini` in the `.env` file
2. Add your API key: `GEMINI_API_KEY=your_gemini_api_key_here`

### Vector Database Configuration

ChromaDB is configured via the `CHROMA_DB_DIR` environment variable:

```
CHROMA_DB_DIR=./chroma_db

```

The directory will be created automatically if it doesn't exist.

Sources: [.gitignore1-6](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/.gitignore#L1-L6)

## Deployment Steps

### Local Development Deployment

1. Clone the repository:

2. Set up the Python environment:

3. Create and configure the `.env` file with appropriate settings.
4. Initialize the database:

5. Run the application:

6. Access the application at `http://localhost:5000`

### Production Deployment

For production deployment:

1. Use a production-ready web server (Nginx or Apache) as a reverse proxy
2. Configure HTTPS with SSL certificates
3. Set the Flask environment to production:

```
FLASK_ENV=production

```

4. Use a production WSGI server like Gunicorn:

5. Configure your web server to proxy requests to Gunicorn

Example Nginx configuration:

Sources: Deployment Architecture diagram from provided information

### LLM Backend Setup

#### Local Ollama Setup

1. Install Ollama from <https://ollama.ai/download>
2. Start the Ollama service
3. Configure `.env` with `OLLAMA_BASE_URL=http://localhost:11434`

#### Remote Ollama Setup

1. Install Ollama on a remote server
2. Install Ngrok on the remote server
3. Start Ollama on the default port (11434)
4. Create an Ngrok tunnel:

5. Update `.env` with the Ngrok URL:

```
OLLAMA_BASE_URL=https://your-ngrok-url.ngrok.io

```

#### Gemini Setup

1. Create a Google AI Studio account
2. Generate an API key for Gemini
3. Add the key to your `.env` file

Sources: Deployment Architecture diagram from provided information

## Troubleshooting

### Common Issues and Solutions

| Issue | Possible Solution |
| --- | --- |
| ChromaDB connection error | Ensure the `CHROMA_DB_DIR` path exists and is writable |
| Ollama connection failed | Check if Ollama server is running and URL is correct |
| Gemini API error | Verify your API key is valid and has sufficient quota |
| Database migration errors | Delete the migration folder and reinitialize with `flask db init` |
| Missing dependencies | Ensure all dependencies from `requirements2.txt` are installed |

### Logs and Debugging

For development, logs appear in the console. To enable debug mode:

```
FLASK_DEBUG=1

```

For production, consider setting up structured logging to a file and implement log rotation.

## Security Considerations

When deploying to production, follow these security best practices:

1. Never expose the `.env` file or its contents publicly
2. Use strong API keys and rotate them periodically
3. Ensure the SQLite database file is not in a web-accessible directory
4. Configure CORS appropriately to restrict cross-origin requests
5. Use HTTPS for all connections
6. Implement proper authentication for admin endpoints

## Data Loading and Initialization

After deployment, you'll need to:

1. Load documents into the vector store via the admin interface
2. Create initial user accounts
3. Configure any subject-specific data for the progress tracking system

### Document Loading Process

Sources: Core System Components diagram from provided information