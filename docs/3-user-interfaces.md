#  User Interfaces

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

- [User Interfaces](#user-interfaces)
  - [Overview of User Interfaces](#overview-of-user-interfaces)
  - [Interface Integration](#interface-integration)
    - [UI Components and Data Flow](#ui-components-and-data-flow)
  - [Chat Interface](#chat-interface)
    - [Key Features](#key-features)
    - [Interface Structure](#interface-structure)
    - [Chat UI Components](#chat-ui-components)
    - [Implementation Details](#implementation-details)
      - [Message Processing Flow](#message-processing-flow)
      - [Audio Input Processing](#audio-input-processing)
      - [Text-to-Speech Capability](#text-to-speech-capability)
  - [Progress Dashboard](#progress-dashboard)
    - [Key Features](#key-features)
    - [Interface Structure](#interface-structure)
    - [Dashboard Components](#dashboard-components)
      - [Subject Progress Visualization](#subject-progress-visualization)
      - [Recommendation System](#recommendation-system)
      - [Subject Detail Modal](#subject-detail-modal)
      - [User Profile Drawer](#user-profile-drawer)
  - [Admin Interface](#admin-interface)
    - [Key Features](#key-features)
    - [Interface Structure](#interface-structure)
    - [Admin Dashboard Components](#admin-dashboard-components)
      - [Login System](#login-system)
      - [Document Upload](#document-upload)
      - [Embeddings Management](#embeddings-management)
      - [System Control and Monitoring](#system-control-and-monitoring)
  - [Integration with Landing Page](#integration-with-landing-page)
  - [Responsive Design and Accessibility](#responsive-design-and-accessibility)
  - [Summary](#summary)

# User Interfaces

Relevant source files

* [api/templates/admin\_page.html](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html)
* [api/templates/chat.html](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html)
* [api/templates/index.html](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/index.html)
* [api/templates/progress\_page.html](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html)

This document provides a technical overview of the user interfaces in the MARS-labs system. It covers the design, implementation, and interaction patterns of the primary interfaces that users and administrators interact with. For information about the backend API that powers these interfaces, see [Backend API (app.py)](/JATAYU000/MARS-labs/2-backend-api-(app.py)).

## Overview of User Interfaces

MARS-labs features three main user interfaces that serve different purposes within the educational platform:

1. **Chat Interface**: A conversational interface allowing users to interact with different AI models for knowledge retrieval
2. **Progress Dashboard**: A visualization system for tracking user learning progress across different subjects
3. **Admin Interface**: A control panel for system administrators to manage documents and monitor system usage

These interfaces work together to provide a comprehensive educational experience by combining AI-powered knowledge retrieval with progress tracking.

Sources: [api/templates/chat.html1-1112](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L1-L1112) [api/templates/progress\_page.html1-1438](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1-L1438) [api/templates/admin\_page.html1-700](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L1-L700) [api/templates/index.html1-502](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/index.html#L1-L502)

## Interface Integration

The interfaces in MARS-labs function as a cohesive system, with the landing page (index.html) serving as the entry point. From there, users can navigate to subject-specific content, engage with the chat assistant, view their progress, or access administrative functions if they have the appropriate permissions.

### UI Components and Data Flow

Sources: [api/templates/index.html245-256](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/index.html#L245-L256) [api/templates/chat.html994-1017](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L994-L1017) [api/templates/progress\_page.html1216-1259](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1216-L1259)

## Chat Interface

The Chat Interface is the primary means through which users interact with the MARS-labs AI system. It provides a conversational experience similar to popular chat applications, with additional educational features.

### Key Features

* Real-time messaging with AI assistant
* Support for multiple AI models (ollama and gemini)
* Voice input via speech recognition
* Text-to-speech output capabilities
* Markdown rendering for formatted responses
* Message history and conversation context

### Interface Structure

Sources: [api/templates/chat.html535-608](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L535-L608) [api/templates/chat.html624-1109](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L624-L1109)

### Chat UI Components

1. **Chat Header**: Contains model selection options, user information, and menu
2. **Message Container**: Displays the conversation history with formatted messages
3. **Input Area**: Text input, audio recording button, and send button
4. **Audio Processing**: Speech recognition and text-to-speech capabilities

The chat interface is implemented as an HTML5 application with extensive CSS styling for visual appeal and JavaScript for dynamic interaction.

### Implementation Details

#### Message Processing Flow

When a user sends a message through the chat interface, the following process occurs:

1. The message is captured from the input field
2. The message is displayed in the UI
3. A typing indicator is shown
4. The message is sent to the backend via a POST request to the `/ask` endpoint
5. The backend processes the message through the RAG pipeline
6. The response is returned and displayed with markdown formatting
7. Optional text-to-speech conversion occurs if enabled

Key code sections handling this functionality:

Sources: [api/templates/chat.html986-1023](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L986-L1023)

#### Audio Input Processing

The chat interface includes audio recording functionality that enables users to speak rather than type. This feature uses the browser's Web Audio API and sends recordings to an external transcription service (AssemblyAI):

1. Audio is captured through the device microphone
2. The audio is recorded and uploaded to AssemblyAI
3. The transcription is returned and inserted into the input field
4. The user can review and send the transcribed text

Sources: [api/templates/chat.html747-878](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L747-L878)

#### Text-to-Speech Capability

The interface also includes text-to-speech functionality that can read AI responses aloud:

1. The user clicks the speak button
2. The latest AI response text is extracted
3. The text is sent to a speech generation API
4. The returned audio URL is played through the browser audio element

Sources: [api/templates/chat.html659-745](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/chat.html#L659-L745)

## Progress Dashboard

The Progress Dashboard provides users with a visual representation of their learning journey across different subjects. It displays completion metrics for various learning activities and offers personalized recommendations.

### Key Features

* Subject-based progress visualization
* Detailed activity tracking
* Personalized recommendations based on progress
* Interactive subject details modal
* User profile and settings access

### Interface Structure

Sources: [api/templates/progress\_page.html1-1438](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1-L1438)

### Dashboard Components

#### Subject Progress Visualization

Each subject is displayed as a card with:

* Subject name
* Circular progress indicator showing completion percentage
* Number of simulations available
* Click interaction to view details

The progress visualization is implemented using SVG graphics for the circular progress indicators and JavaScript for calculating and updating the progress values.

Sources: [api/templates/progress\_page.html1261-1270](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1261-L1270)

#### Recommendation System

The dashboard includes a recommendation system that analyzes the user's progress and suggests areas to focus on:

Sources: [api/templates/progress\_page.html1272-1311](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1272-L1311)

#### Subject Detail Modal

Clicking on a subject card opens a modal that displays:

* Detailed list of simulations for the subject
* Progress status for each simulation task (procedure, quiz, animation, etc.)
* Visual indicators of completion status

Sources: [api/templates/progress\_page.html1379-1416](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1379-L1416)

#### User Profile Drawer

The dashboard includes a user profile drawer accessible through an icon in the top-right corner:

* Personal information (name, email)
* Class level
* Overall progress statistics
* Settings and logout options

Sources: [api/templates/progress\_page.html232-380](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L232-L380) [api/templates/progress\_page.html1216-1259](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/progress_page.html#L1216-L1259)

## Admin Interface

The Admin Interface provides system administrators with tools to manage the MARS-labs platform, including document ingestion, user management, and system monitoring.

### Key Features

* Secure login system
* Document upload for the RAG system
* Vector embeddings management
* System statistics and monitoring
* Database management tools

### Interface Structure

Sources: [api/templates/admin\_page.html1-700](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L1-L700)

### Admin Dashboard Components

#### Login System

The admin interface begins with a login screen that provides secure access to administrative functions:

* Username and password authentication
* Client-side validation
* Error messaging for failed authentication attempts

While the provided code uses a simple hardcoded authentication for demonstration purposes, a production system would implement proper server-side authentication.

Sources: [api/templates/admin\_page.html248-262](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L248-L262) [api/templates/admin\_page.html336-351](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L336-L351)

#### Document Upload

The document upload section allows administrators to:

* Select and upload PDF files
* View upload progress with a progress bar
* Receive status updates on the upload process
* See notifications of successful or failed uploads

Sources: [api/templates/admin\_page.html354-377](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L354-L377) [api/templates/admin\_page.html425-502](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L425-L502)

#### Embeddings Management

The admin interface provides tools for managing the embeddings used by the RAG system:

* Upload pre-generated embeddings files (.npy format)
* Upload text chunks associated with the embeddings
* Monitor upload progress
* Receive status notifications

Sources: [api/templates/admin\_page.html380-422](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L380-L422) [api/templates/admin\_page.html505-587](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L505-L587)

#### System Control and Monitoring

The control panel section provides:

* System statistics (file count, user count)
* Recent activity log
* Control buttons for system maintenance
* Logout functionality

Sources: [api/templates/admin\_page.html300-331](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L300-L331) [api/templates/admin\_page.html590-697](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/admin_page.html#L590-L697)

## Integration with Landing Page

The landing page (index.html) serves as the central hub for accessing all the interfaces. It includes:

1. A navigation menu for accessing different subject areas
2. A floating chat button that opens the chat interface in a side drawer
3. Links to the progress dashboard
4. Links to the admin interface (for authorized users)

The chat interface is embedded as an iframe within a side drawer that can be toggled open and closed:

Sources: [api/templates/index.html245-257](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/index.html#L245-L257) [api/templates/index.html259-293](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/index.html#L259-L293)

## Responsive Design and Accessibility

All interfaces in MARS-labs are designed with responsiveness in mind, adapting to different screen sizes and device types. The CSS uses:

* Flexible layouts with responsive units
* Media queries for different screen sizes
* Mobile-friendly touch targets
* Accessible color contrasts and font sizes

Example of responsive design implementation:

Sources: [api/templates/index.html396-427](https://github.com/JATAYU000/MARS-labs/blob/d77026a0/api/templates/index.html#L396-L427)

## Summary

The MARS-labs system provides three primary user interfaces:

1. A chat interface for AI-powered knowledge retrieval
2. A progress dashboard for tracking learning activities
3. An admin interface for system management

These interfaces work together to create a comprehensive educational platform that combines the power of retrieval-augmented generation with structured learning progress tracking. The interfaces are designed to be responsive, accessible, and user-friendly while providing powerful educational tools.

The user interfaces connect to the backend API (app.py) which handles data processing, user management, and integration with the RAG system for knowledge retrieval. This separation of concerns ensures that the frontend interfaces can focus on user experience while the backend handles the complex data processing and AI integration.