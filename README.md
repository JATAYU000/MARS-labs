MARS-labs: Olabs Hackathon Solution
===================================

MARS-labs is a solution developed for the Olabs Hackathon. It is built using Python for the backend logic, along with HTML, and CSS templates for the frontend. The project integrates API endpoints, database interactions, and file uploads into a modular structure.

Table of Contents
-----------------

-   [Installation](#installation)
-   [Usage](#usage)


Installation
------------

1.  **Clone the Repository**

    ```
    git clone https://github.com/JATAYU000/MARS-labs.git
    cd MARS-labs
    ```

2.  **Install Dependencies**

    The project provides requirements files. Start by installing the  dependencies:

    ```
    pip install -r requirements2.txt
    ```

Usage
-----

-   **Starting the Application**

    Run the main script to start the application:

    ```
    python3 api/app.py
    ```

Additional
----------
- **Collab Notebook** : https://colab.research.google.com/drive/1yYQGq3-ow0Fy1OuigYvsH9bi6KDl4kM3?usp=sharing
  

Key Features & Benefits:
------------------------

-    **Advanced Conversational AI**
    Utilizes leading language models (with support for both ChatOllama and Gemini) to generate human-like, insightful answers that adapt to users' progress and context.

-   **Dynamic Document Ingestion & Retrieval**
    Converts uploaded documents (e.g., PDFs) into vector embeddings using cutting-edge models (like HuggingFace's e5-base), allowing the system to "understand" and recall specific content. This empowers users to ask questions based on detailed, domain-specific documents.

-   **Personalized Mentorship**
    By integrating user progress data and chat history, MARS-labs tailors responses to each user, enhancing learning outcomes and ensuring that guidance is both relevant and supportive.

-   **Scalable & Modular Architecture**
    Built on Flask with SQLAlchemy for robust session management and multiple database support, the platform is designed to scale. Its modular codebase supports future integrations and easy model switching based on evolving user needs.

-   **Rapid Deployment & Innovation**
    Born out of a hackathon environment, MARS-labs embodies agile development and innovative problem solving. It proves that high-quality, enterprise-grade solutions can emerge in fast-paced, competitive settings.

**Technical Overview:**
MARS-labs integrates Flask as its web framework with a SQLAlchemy-backed database to handle user progress and file uploads. It uses LangChain and ChromaDB to manage document ingestion and similarity search. The system's flexible design allows for the dynamic selection of LLMs, making it future-proof in an ever-changing AI landscape.

**Market Opportunity:**
As the demand for personalized, AI-driven learning and knowledge management tools grows, MARS-labs positions itself as a key enabler for educational institutions and enterprises. By offering an intuitive chat interface coupled with precise, context-rich responses, it fills a crucial gap in the market for scalable, interactive mentorship platform


Screenshots:
-----------
[img](./screenshots/250301_15h45m22s_screenshot.png)
[img](./screenshots/250301_15h43m41s_screenshot.png)
[img](./screenshots/250301_15h40m31s_screenshot.png)
[img](./screenshots/250301_15h40m55s_screenshot.png)
[img](./screenshots/250301_15h37m19s_screenshot.png)
