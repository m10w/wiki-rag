import logging
import json
import wikipediaapi
from helpers import upload_to_blob

def extract_and_upload_wiki_data(topics, container_name):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='WikiDataExtractor/1.0 (hamed2005@gmail.com)', language='en')

    for topic in topics:
        logging.info(f"Extracting data for: {topic}")
        page = wiki_wiki.page(topic)
        if page.exists():
            topic_data = {
                "title": page.title,
                "summary": page.summary,
                "sections": [s.title for s in page.sections],
                "categories": list(page.categories.keys()),
                "content": page.text,
            }
            json_data = json.dumps(topic_data, indent=4)

            # Blob name will be the topic title with spaces replaced by underscores
            blob_name = f'{topic.replace(' ', '_')}.json'

            # upload to Blob storage
            upload_to_blob(json_data, container_name, blob_name)

        else:
            logging.info(f"Page '{topic}' does not exist.")


if __name__ == "__main__":
    topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Large Language Models",
    "Transformers (machine learning model)",
    "Neural Networks",
    "Reinforcement Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Semi-supervised Learning",
    "Self-supervised Learning",
    "Statistical Learning",
    "Bayesian Networks",
    "Probabilistic Graphical Models",
    "Support Vector Machines",
    "Decision Trees",
    "Random Forests",
    "Gradient Boosting",
    "K-Nearest Neighbors",
    "Clustering Algorithms",
    "Principal Component Analysis",
    "Linear Regression",
    "Logistic Regression",
    "Optimization in Machine Learning",
    "Backpropagation",
    "Convolutional Neural Networks",
    "Recurrent Neural Networks",
    "Long Short-Term Memory (LSTM)",
    "Attention Mechanism",
    "Sequence-to-Sequence Models",
    "Word Embeddings",
    "Tokenization in NLP",
    "Text Summarization",
    "Text Classification",
    "Sentiment Analysis",
    "Question Answering Systems",
    "Pretrained Models in NLP",
    "Fine-Tuning in NLP",
    "Zero-Shot Learning",
    "Few-Shot Learning",
    "Meta-Learning",
    "Generative Adversarial Networks (GANs)",
    "Autoencoders",
    "Variational Autoencoders",
    "Contrastive Learning",
    "Self-Attention",
    "Explainable AI",
    "Interpretable Machine Learning",
    "Fairness in Machine Learning",
    "Bias in AI",
    "Ethical AI",
    "AI Safety",
    "Adversarial Attacks in AI",
    "Robust Machine Learning",
    "Differential Privacy",
    "Federated Learning",
    "Distributed Machine Learning",
    "Information Theory",
    "Entropy in Information Theory",
    "Mutual Information",
    "KL Divergence",
    "Cross-Entropy Loss",
    "Shannon's Information Theory",
    "Model Compression",
    "Knowledge Distillation",
    "Transfer Learning",
    "Multimodal Machine Learning",
    "Human-in-the-Loop AI",
    "Active Learning",
    "Bayesian Optimization",
    "Hyperparameter Tuning",
    "Markov Chains",
    "Monte Carlo Methods",
    "Expectation-Maximization Algorithm",
    "Graph Neural Networks",
    "Sparse Neural Networks",
    "Transformers in Vision",
    "Vision-Language Models",
    "Diffusion Models in AI",
    "OpenAI GPT Models",
    "BERT (Bidirectional Encoder Representations from Transformers)",
    "ChatGPT",
    "Google DeepMind",
    "AlphaGo",
    "AI Ethics and Policy",
    "Responsible AI",
    "Regulation of AI",
    "History of AI",
    "Applications of AI",
    "AI in Healthcare",
    "AI in Finance",
    "AI in Education",
    "AI in Autonomous Vehicles",
    "Causal Inference in AI",
    "Reproducibility in AI",
    "Computational Complexity in AI",
    "Sparse Representations in AI"
]
    container_name = "wiki-data"
    extract_and_upload_wiki_data(topics, container_name)
