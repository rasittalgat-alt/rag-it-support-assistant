# Development of RAG-based AI system – IT Support RAG Assistant

## 1. Project Overview

**Project name:** IT Support RAG Assistant  
**Author:** Talgat Rashit  

The goal of this project is to develop a Retrieval-Augmented Generation (RAG) system that assists IT Support by answering typical user questions based on an internal knowledge base (FAQ, instructions, policies, and example tickets).

The user asks a question (e.g. *“How can I connect to corporate VPN from home?”*), the system finds relevant documents in the knowledge base using vector search and then uses an LLM to generate a clear and accurate answer based strictly on these documents.

---

## 2. Main Idea and Use Cases

### 2.1. Main Idea

Build a conversational assistant for IT Support that:

1. **Understands** natural language questions from employees.
2. **Searches** in a knowledge base using semantic vector search.
3. **Augments** an LLM with retrieved documents to generate factual answers.
4. **Explains** solutions in simple, step-by-step form.

### 2.2. Example Use Cases

- Wi-Fi issues:  
  *“How do I connect to the corporate Wi-Fi on Windows?”*
- VPN access:  
  *“I can’t access internal systems from home, what should I do?”*
- Corporate email:  
  *“How to set up corporate email on my phone?”*
- Printers:  
  *“Network printer does not print, how can I fix it?”*
- Accounts & passwords:  
  *“How can I reset my corporate password?”*

---

## 3. RAG Concepts in This Project

This project implements a classic RAG pipeline:

1. **Knowledge Base (KB)**  
   A small but representative set of IT support documents:
   - FAQs (Wi-Fi, VPN, email, printers, passwords);
   - How-to guides / runbooks;
   - Policy excerpts (password rules, SLAs);
   - Example tickets with resolutions.

2. **Chunking**  
   - Each document is split into small chunks (e.g. 100–200 tokens).
   - Each chunk becomes a separate retrievable unit.

3. **Embeddings**  
   - For every chunk an embedding vector is created.
   - For user queries we also compute embeddings using the same model.

4. **Vector Search**  
   - User query embedding is compared to chunk embeddings.
   - Top-N most similar chunks are selected as context.

5. **Augmented Generation**  
   - LLM receives:
     - original user question;
     - retrieved context chunks.
   - LLM generates an answer, restricted to the provided context.

---

## 4. System Design and Architecture

### 4.1. High-Level Components

1. **User Interface (UI)**
   - Simple web UI (Streamlit) or CLI.
   - Input: user question.
   - Output: AI answer and optionally sources (titles / IDs of documents).

2. **Embeddings Client**
   - Wrapper for an embedding model (e.g. `text-embedding-3-small` or `text-embedding-ada-002`).
   - Responsibilities:
     - create embeddings for chunks when building the dataset;
     - create embeddings for user query at runtime.

3. **Vector Database**
   - Stores:
     - embedding vector;
     - original chunk text;
     - metadata (type, category, source).
   - Provides k-nearest neighbors (k-NN) search by similarity.

4. **LLM Client**
   - Wrapper for DIAL or another LLM provider.
   - Handles prompt construction and response generation.

5. **RAG Pipeline**
   - Orchestrates the steps:
     1. embed user query;
     2. search relevant chunks in vector DB;
     3. build prompt with context;
     4. call LLM and return answer.

6. **Data Ingestion Pipeline**
   - Offline script(s) that:
     1. reads raw dataset files;
     2. cleans and splits them into chunks;
     3. creates embeddings;
     4. uploads vectors + metadata into the vector database.

### 4.2. Request Flow

1. User submits a question via UI.
2. Backend calls Embeddings Client to get query vector.
3. Vector DB performs similarity search and returns top-N chunks.
4. RAG Pipeline builds a prompt with:
   - system instruction;
   - concatenated context chunks;
   - user question.
5. LLM Client sends this prompt to the LLM provider and gets an answer.
6. UI displays the answer and (optionally) the sources.

---

## 5. Dataset Concept

### 5.1. Types of Data

The dataset will contain several logical groups of documents:

1. **IT Support FAQs**
   - Wi-Fi connection instructions;
   - VPN access;
   - Email setup;
   - Password reset;
   - Network printer configuration.

2. **How-to Guides / Runbooks**
   - Step-by-step VPN setup on Windows/macOS;
   - Corporate Wi-Fi connection for different OSes;
   - Configuring corporate email on desktop and mobile.

3. **Policies and SLAs**
   - Password requirements (length, complexity, rotation);
   - SLA excerpts: incident priority levels, response and resolution times.

4. **Example Tickets**
   - Simplified and anonymized tickets with resolutions, such as:
     - “Cannot access corporate email from home”;
     - “VPN disconnected every 5 minutes”;
     - “Network printer doesn’t print for specific user”.

### 5.2. Raw Data Format

Planned raw data structure:

- `data/raw/faqs.yaml` – structured FAQ entries;  
- `data/raw/runbooks/*.md` – how-to guides as Markdown files;  
- `data/raw/policies/*.md` – policy excerpts;  
- `data/raw/tickets.json` – example tickets.

Example FAQ in YAML:

```yaml
- id: faq_wifi_001
  category: wifi
  question: "How to connect to corporate Wi-Fi on Windows?"
  answer: |
    To connect to corporate Wi-Fi on Windows:
    1. Open Wi-Fi settings.
    2. Select the network "CORP-WIFI".
    3. Use your corporate username and password.
    4. Accept the security certificate if prompted.
