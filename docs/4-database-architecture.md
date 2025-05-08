#  Database Architecture

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

- [Database Architecture](#database-architecture)
  - [Purpose and Scope](#purpose-and-scope)
  - [Database Technology Overview](#database-technology-overview)
  - [Core Schema Design](#core-schema-design)
    - [Primary Database Schema](#primary-database-schema)
    - [Database Table Details](#database-table-details)
      - [User Table](#user-table)
      - [Progress Table](#progress-table)
  - [JSON Data Structure](#json-data-structure)
    - [Progress JSON Schema](#progress-json-schema)
  - [Alternative Database Schemas](#alternative-database-schemas)
    - [Alternative Progress Schema](#alternative-progress-schema)
    - [Chatbot Database](#chatbot-database)
  - [Database File Organization](#database-file-organization)
  - [Database Access Patterns](#database-access-patterns)
  - [Integration with MARS-labs Components](#integration-with-mars-labs-components)
  - [Data Model Evolution](#data-model-evolution)
  - [Summary](#summary)

# Database Architecture

Relevant source files

* [api/instance/database1.db](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/instance/database1.db)
* [instance/database1.db](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/instance/database1.db)
* [instance/database2.db](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/instance/database2.db)
* [instance/default.db](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/instance/default.db)

## Purpose and Scope

This document describes the database architecture of the MARS-labs educational platform. It covers the database technologies used, schema structures, relationships between data models, and how the system stores and retrieves information. The document focuses on the persistent data storage components that support user management, progress tracking, and educational content management.

For specific details about the implementation of user and progress data models, see [User and Progress Models](/JATAYU000/MARS-labs/4.1-user-and-progress-models).

## Database Technology Overview

MARS-labs employs SQLite as its primary database management system. SQLite provides a lightweight, serverless database solution that stores the entire database as a single file on disk. This choice aligns with the application's needs for simplicity, reliability, and ease of deployment.

The system uses multiple database files stored in the `instance/` directory, each with specific purposes:

Sources: api/instance/database1.db, instance/database1.db, instance/database2.db, instance/default.db

## Core Schema Design

### Primary Database Schema

The main database schema contains two core tables with a relationship that forms the foundation of user progress tracking:

Sources: api/instance/database1.db:8-14, api/instance/database1.db:2-7

### Database Table Details

#### User Table

The User table stores basic information about students using the platform:

| Column Name | Data Type | Constraints | Description |
| --- | --- | --- | --- |
| id | INTEGER | PRIMARY KEY | Unique identifier for each user |
| first\_name | VARCHAR(50) | NOT NULL | User's first name |
| last\_name | VARCHAR(50) | NOT NULL | User's last name |
| age | VARCHAR(4) | NOT NULL | User's age |

Sources: api/instance/database1.db:9-13

#### Progress Table

The Progress table implements a flexible schema using JSON data storage:

| Column Name | Data Type | Constraints | Description |
| --- | --- | --- | --- |
| id | INTEGER | PRIMARY KEY | Unique identifier for each progress record |
| user\_id | INTEGER | NOT NULL, FOREIGN KEY | Reference to User.id |
| progress\_data | JSON | NOT NULL | Structured JSON data containing progress information |

The `progress_data` field uses JSON to store structured progress information, allowing for flexibility in tracking different types of educational activities.

Sources: api/instance/database1.db:2-7

## JSON Data Structure

The Progress table uses a JSON-based approach to store structured data about student progress across various subjects and activities. This design provides flexibility to accommodate different types of learning content without schema changes.

### Progress JSON Schema

```
{
    "class": <integer>,
    "subject_progress": {
        "<subject_name>": {
            "completed_experiments": [<array_of_experiment_names>],
            "completed_quizzes": [<array_of_quiz_names>]
        },
        ...
    }
}

```

Example progress data:

This JSON structure enables the system to track:

* Student's class/grade level
* Progress across multiple subjects simultaneously
* Different types of activities (experiments, quizzes) within each subject
* Specific completed activity names

Sources: api/instance/database1.db:17-18

## Alternative Database Schemas

### Alternative Progress Schema

The system includes an alternative schema for progress tracking in `database2.db`:

This schema uses a more traditional relational approach with fixed columns rather than flexible JSON data.

Sources: instance/database2.db:2-9

### Chatbot Database

The system also includes a separate database table for storing question-answer pairs used by the chatbot functionality:

This table stores pre-defined question-answer pairs with a unique constraint on questions to prevent duplicates.

Sources: instance/default.db:2-8

## Database File Organization

The database files are stored in the instance directory, which is a standard location for Flask applications to store instance-specific data:

```
MARS-labs/
├── api/
│   └── instance/
│       └── database1.db     # Primary database with user and progress data
└── instance/
    ├── database1.db         # Copy or alternative version of primary database
    ├── database2.db         # Alternative progress schema
    └── default.db           # Chatbot Q&A database

```

The presence of database files in both `api/instance/` and `instance/` directories suggests the system may have multiple entry points or deployment configurations.

Sources: api/instance/database1.db, instance/database1.db, instance/database2.db, instance/default.db

## Database Access Patterns

The MARS-labs system accesses the database primarily from the Flask API backend. The typical data flow for user and progress data follows this pattern:

Sources: api/instance/database1.db

## Integration with MARS-labs Components

The database system integrates with the broader MARS-labs architecture as follows:

The diagram shows how the SQLite databases fit into the larger system architecture, working alongside the vector database (ChromaDB) and language models to provide a complete educational experience.

Sources: api/instance/database1.db, instance/default.db

## Data Model Evolution

The presence of multiple database schemas (in database1.db and database2.db) suggests that the data model has evolved over time. The system appears to have transitioned from:

1. A fixed-column approach for progress data (in database2.db)
2. To a more flexible JSON-based approach (in database1.db)

This evolution reflects a common pattern in application development, where initial fixed schemas are replaced with more flexible approaches as requirements evolve and data structures become more complex.

Sources: api/instance/database1.db, instance/database2.db

## Summary

The MARS-labs database architecture employs SQLite as its foundation with a combination of traditional relational tables and flexible JSON-based storage. This hybrid approach allows the system to maintain relational integrity while accommodating the complex, hierarchical, and evolving nature of educational progress data.

Key strengths of this architecture include:

* Simplicity of deployment (file-based SQLite)
* Relational integrity for user-to-progress relationships
* Flexible JSON storage for complex progress structures
* Separate databases for distinct functional areas (user data vs. chatbot data)

This architecture effectively supports the platform's educational goals by providing persistent, structured storage for user information and learning progress across diverse subjects and activities.