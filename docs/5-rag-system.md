#  Rag System

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

- [RAG System](#rag-system)
  - [Overview](#overview)
  - [Core Components](#core-components)
    - [Vector Database](#vector-database)
    - [Document Processing](#document-processing)
    - [Language Models](#language-models)
  - [Document Ingestion Process](#document-ingestion-process)
  - [Query Processing Flow](#query-processing-flow)
  - [Prompt Template](#prompt-template)
  - [Remote Ollama Integration](#remote-ollama-integration)
  - [Gemini Integration](#gemini-integration)
  - [NPY Files Processing](#npy-files-processing)
  - [Integration with the Main Application](#integration-with-the-main-application)
  - [Session Management](#session-management)
  - [Error Handling and Logging](#error-handling-and-logging)
  - [Performance Considerations](#performance-considerations)

# RAG System

Relevant source files

* [README.md](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/README.md)
* [api/app.py](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py)

This document describes the Retrieval-Augmented Generation (RAG) system implementation in MARS-labs. The RAG system is responsible for retrieving relevant context from documents and generating context-aware responses using language models.

For information about the specific language models used, see [Language Models](/JATAYU000/MARS-labs/5.2-language-models).

## Overview

The RAG system in MARS-labs combines document retrieval with language model generation to provide accurate and contextually relevant responses. It first retrieves relevant passages from a vector database based on similarity search, then uses these passages as context for a language model to generate informed responses.

Sources: [api/app.py17-28](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L17-L28) [api/app.py95-274](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L95-L274)

## Core Components

The RAG system is primarily implemented in the `Mentor` class which orchestrates document retrieval, context augmentation, and response generation.

Sources: [api/app.py95-274](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L95-L274) [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92)

### Vector Database

The system uses ChromaDB, a vector database implementation from LangChain, to store document embeddings and retrieve relevant texts based on similarity search.

Key configurations include:

* Embedding model: HuggingFace's "intfloat/e5-base"
* Storage location: "chroma\_db/" directory
* Retrieval method: similarity search with search\_kwargs={"k": 10, "score\_threshold": 0.0}

Sources: [api/app.py151-167](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L151-L167) [api/app.py335-336](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L335-L336)

### Document Processing

Documents are processed using the following components:

* `PyPDFLoader`: Loads PDF files
* `RecursiveCharacterTextSplitter`: Splits documents into manageable chunks (512 characters with 100 character overlap)
* `HuggingFaceEmbeddings`: Generates vector embeddings using "intfloat/e5-base" model

Sources: [api/app.py99-101](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L99-L101) [api/app.py155](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L155-L155) [api/app.py185-205](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L185-L205)

### Language Models

The system supports two language models:

1. **Ollama**: Accessed through a remote API using ngrok tunneling
2. **Gemini**: Accessed through Google Generative AI API

The model selection is configurable and can be changed dynamically.

Sources: [api/app.py38](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L38-L38) [api/app.py134-145](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L134-L145) [api/app.py526-527](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L526-L527)

## Document Ingestion Process

The document ingestion process converts PDFs into vector embeddings stored in ChromaDB.

Sources: [api/app.py185-205](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L185-L205) [api/app.py471-507](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L471-L507)

## Query Processing Flow

When a user asks a question, the system processes it through the following steps:

Sources: [api/app.py207-229](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L207-L229) [api/app.py231-255](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L231-L255) [api/app.py510-561](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L510-L561)

## Prompt Template

The system uses a carefully engineered prompt template to format the context and query before sending it to the language model:

```
You are an expert mentor providing personalized guidance based on the following context. 

CONTEXT: {context}

USER QUERY: {question}

INSTRUCTIONS:
1. Prioritize information from the provided context, but supplement with general knowledge when necessary.
2. Deliver concise but sufficiently explanatory, factually accurate responses with examples(if available) that directly address the query.
3. Do not mention the text, if it is from the text or outside the text, just give the response and don't specify whether it is form the text or outside.
4. Consider the chat history (included in the query) to personalize your guidance.
5. Maintain a supportive, encouraging tone throughout.
6. If the user's input is not in English, respond in the same language.
7. If chat history shows a model switch occurred, adjust your response style accordingly.
8. Remember this is a RAG (Retrieval-Augmented Generation) implementation using two models.

Respond in a clear, helpful manner that builds the user's confidence while providing accurate information.

```

Sources: [api/app.py107-127](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L107-L127) [api/app.py233-251](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L233-L251)

## Remote Ollama Integration

For the Ollama model, the system uses a custom `RemoteOllamaWrapper` class to communicate with a remote Ollama instance via ngrok tunneling:

Sources: [api/app.py62-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L62-L92) [api/app.py222](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L222-L222)

## Gemini Integration

For the Gemini model, the system uses the Google Generative AI API:

Sources: [api/app.py231-255](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L231-L255) [api/app.py224](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L224-L224)

## NPY Files Processing

The system also supports loading pre-generated embeddings and chunks from NPY files:

Sources: [api/app.py355-390](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L355-L390) [api/app.py613-639](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L613-L639)

## Integration with the Main Application

The RAG system is integrated with the Flask application through several endpoints:

| Endpoint | HTTP Method | Function | Description |
| --- | --- | --- | --- |
| `/ask` | POST | `ask_question()` | Process user questions and return responses |
| `/upload` | POST | `upload_file()` | Upload and ingest documents |
| `/upload_npy` | POST | `upload_npy()` | Upload pre-generated embeddings |
| `/clear` | POST | `clear_conversation()` | Clear the conversation history and vector store |

Source: [api/app.py456-460](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L456-L460) [api/app.py471-507](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L471-L507) [api/app.py510-561](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L510-L561) [api/app.py565-574](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L565-L574) [api/app.py613-639](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L613-L639)

## Session Management

The system maintains separate RAG instances for each user session:

Sources: [api/app.py299](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L299-L299) [api/app.py339-352](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L339-L352)

## Error Handling and Logging

The RAG system implements error handling throughout its components to ensure robustness:

* Exception handling in `ingest()` method
* Exception handling in `ask()` method
* Exception handling in `RemoteOllamaWrapper.invoke()`
* Exception handling in document processing

The system also uses LangChain debugging features with `set_verbose(True)` and `set_debug(True)` to facilitate troubleshooting.

Sources: [api/app.py31-32](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L31-L32) [api/app.py185-205](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L185-L205) [api/app.py207-229](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L207-L229) [api/app.py90-92](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L90-L92)

## Performance Considerations

* **Chunk Size**: Documents are split into 512-character chunks with 100-character overlap
* **Top-k Retrieval**: Retrieves top 10 most similar chunks for context
* **Embedding Model**: Uses efficient e5-base embeddings from HuggingFace
* **Model Selection**: Supports both local (via Ollama) and cloud-based (via Gemini) models

Sources: [api/app.py99-101](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L99-L101) [api/app.py155](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L155-L155) [api/app.py165-167](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/app.py#L165-L167)