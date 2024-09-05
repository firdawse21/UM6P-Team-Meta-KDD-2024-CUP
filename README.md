# UM6P-Team-Meta-KDD-2024-CUP
the Comprehensive RAG Benchmark - Meta KDD 2024 CUP
![banner image](https://aicrowd-production.s3.eu-central-1.amazonaws.com/challenge_images/meta-kdd-cup-24/meta_kdd_cup_24_banner.jpg)
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/yWurtB2huX)

# Meta KDD Cup '24 [CRAG: Comprehensive RAG Benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) Starter Kit

# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
5. [Getting Started](#-getting-started)
   - [How to write your own model?](#️-how-to-write-your-own-model)
   - [How to start participating?](#-how-to-start-participating)
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
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your model.
4. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
5. [**Submit**](#-how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#-how-to-make-a-submission). The automated evaluation will evaluate the submissions on the public test set and report the metrics on the leaderboard of the competition.

# ✍️ How to write your own model?

Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models for this competition.

# 🚴 How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/-/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/user/ssh.html).


2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/forks/new) to create a fork.

3.  **Clone the repository**

    ```bash
    git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USERNAME>/meta-comphrehensive-rag-benchmark-starter-kit.git
    cd meta-comphrehensive-rag-benchmark-starter-kit
    ```

4. **Install** competition specific dependencies!
    ```bash
    cd meta-comphrehensive-rag-benchmark-starter-kit
    pip install -r requirements.txt
    ```

5. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

6. Test your model locally using `python local_evaluation.py`.


# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024

