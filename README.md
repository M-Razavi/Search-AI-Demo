# Search-AI-Demo

## Overview

This project is a Spring Boot application that demonstrates the use of AI models for semantic search and chat functionalities. The project is divided into two main parts:

1. **SemanticSearchApplication**: The main application that sets up the Spring Boot environment and mocking the necessary services for semantic search.
2. **SpringAiTests**: A set of tests to demo some capabilities of the AI models and services integrated into the application.

## SemanticSearchApplication
A Spring Boot application that implements AI-powered semantic search for users within an organization.

### Key Components

1. **SearchController**: Handles API requests for user searches
2. **SearchService**: Core logic for processing natural language queries
3. **OllamaConfig**: Sets up AI model and function callbacks
4. **Repositories**: Simulate databases for users, projects, teams, and mentions
5. **Converters**: Transform AI responses into structured data

### Key Features

- Natural language query processing
- Context-aware search (org, team, user)
- Integration with Ollama for AI functionality
- Flexible data retrieval via function callbacks

## SpringAiTests

The `SpringAiTests` class contains various methods to demo the functionality of the AI models and VectorStore with help of PGVector and Llama3.1.

- **tellMeJoke**: Sends a simple user prompt asking for telling a joke.
- **tellMeJokeSystemPrompt**: Asks for a joke in different "voices" (e.g., Picasso, Musk, Shakespeare) using a system prompt.

      
- **structuredOutput**: Retrieves a structured output for a given prompt.
- **structuredOutputList**: Retrieves a structured output for a given prompt, maps it to a list.


- **noContext**: Sends a prompt without additional context.
- **stuffPrompt**: Sends a prompt with extra contextual information.
- **preLoadData**: Preloads data and write into PGVector, which preparing system for subsequent question-answering method.
- **purposeQuestionWIthPreLoadedData**: Asks a question that needs preloaded data and evaluates the relevancy of the response. (RAG)


- **conversationMemory**: Tests chat memory by asking the user’s name and recalling it in subsequent interactions.


- **multiModality**: Sends a prompt involving an image and asks for an explanation of its contents.


## Running the Application

### Prerequisites

- Java 11 or higher
- Maven
- Docker

### Running the Application Locally

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-repo/Search-AI-Demo.git
    cd Search-AI-Demo
    ```

2. **Running Required LLMs and Containers**: 
    - **Navigate to the `Docker-Script` directory**:
    ```sh
    cd Docker-Script
    ```
   - **Run the Docker Compose file**:
    ```sh
    docker-compose -f docker-compose-pgvector.yml up -d
    ```

3. **Running Required LLMs**: 
   - Follow the instructions on the [Llama3.1 GitHub page](https://github.com/ollama/ollama) to download and set up the models locally and after that:
        ```sh
       ollama pull llama3.1
       ollama pull llava
       ```
