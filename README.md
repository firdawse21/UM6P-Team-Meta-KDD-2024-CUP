# UM6P-Team-Meta-KDD-2024-CUP
![banner image](https://aicrowd-production.s3.eu-central-1.amazonaws.com/challenge_images/meta-kdd-cup-24/meta_kdd_cup_24_banner.jpg)


# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
5. [Getting Started](#-getting-started)
   - [Install dependencies](#️-Install-dependencies)
   - [Download the models](#-Download-the-models)
      - [Setup](#setup)
      - [How to make a submission?](#-how-to-make-a-submission)
      - [What hardware does my code run on?](#-what-hardware-does-my-code-run-on-)
      - [How are my model responses parsed by the evaluators?](#-how-are-my-model-responses-parsed-by-the-evaluators-)
      - [Baselines](#baselines)
6. [Frequently Asked Questions](#-frequently-asked-questions)
6. [Important Links](#-important-links)


# 📖 Competition Overview
The Comprehensive RAG (CRAG) Benchmark evaluates RAG systems across five domains and eight question types, and provides a practical set-up to evaluate RAG systems. In particular, CRAG includes questions with answers that change from over seconds to over years; it considers entity popularity and covers not only head, but also torso and tail facts; it contains simple-fact questions as well as 7 types of complex questions such as comparison, aggregation and set questions to test the reasoning and synthesis capabilities of RAG solutions.

# 📊 Dataset

The CRAG dataset is designed to support the development and evaluation of Retrieval-Augmented Generation (RAG) models. It consists of two main types of data:

1. **Question Answering Pairs:** Pairs of questions and their corresponding answers.
2. **Retrieval Contents:** Contents for information retrieval to support answer generation.

Retrieval contents are divided into two types to simulate practical scenarios for RAG:

1. **Web Search Results:** For each question, up to `50` **full HTML pages** are stored, retrieved using the question text as a search query. For Task 1, `5 pages` are **randomly selected** from the `top-10 pages`. These pages are likely relevant to the question, but relevance is not guaranteed.
2. **Mock KGs and APIs:** The Mock API is designed to mimic real-world **Knowledge Graphs (KGs)** or **API searches**. Given some input parameters, they output relevant results, which may or may not be helpful in answering the user's question.

- **Task #1:** [Retreival Summarization Task Page](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files)
- **Task #2:** [Mock API Repository](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api)
- **Task #3:** [End to End Retreival Augmentation Task Page](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/end-to-end-retrieval-augmented-generation)


# 🏁 Getting Started
   1. **✍️ Install dependencies**
   ```bash
       pip install -r requirements.txt
       ```

   2. **🚴 Download the models**

   3.Test your model locally using python local_evaluation.py.



# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024

