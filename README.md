# RentRentOut - AI Microservice for Ad Categorization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZbhCI4aOtBGhqLK4RmqyRamllwBLAVSk?usp=sharing)

This is a Python microservice running in the background of my **Rent Rent Out** platform (Spring Boot + Angular). Its core purpose is to automatically predict and suggest the correct category (out of 300+ available options) based on the user's ad title input.

## Under the Hood

Since the platform is newly built and lacks a historical database of 100,000+ real user ads, I had to make mock the training data from scratch.

1. **Data Engineering:** I wrote a script that extracts real category names and IDs from the relational database and merges them with a curated list of synonyms and brand names (e.g., "Hilti", "iPhone", "pressure washer"). This generated a balanced dataset of over 6,000 realistic ad titles.
2. **Preprocessing:** All text goes through normalization (removing Balkan diacritics like š, đ, č, ć, ž and mapping them to basic Latin characters). The text is then converted into tensors using scikit-learn's `TfidfVectorizer` (utilizing 1-2 n-grams to capture multi-word phrases).
3. **The Model:** The architecture is a classic *Feedforward Neural Network (MLP)* built in **PyTorch**. It consists of Linear layers, ReLU activations, and Dropout regularization to prevent overfitting on the mock data.
4. **Deployment:** The learned weights (`.pth`) and vectorizers/encoders (`.pkl`) are loaded into RAM via a **FastAPI** server. The Spring Boot backend communicates with this service through an internal REST API and forwards the predictions to the Angular frontend.

The model achieves over 98% accuracy on the synthetic test set.

## Future Work

The current TF-IDF + MLP model is a highly performant and lightweight baseline. However, synthetic data doesn't perfectly mimic real users typing with typos and local slang.

The next architectural step is implementing **Shadow Mode** on the backend. The AI will process real user inputs in the background (without impacting the UX) and log the categories users ultimately end up selecting manually. 

Once enough real-world corrections are collected (creating a *Data Flywheel* effect), this data will be used to fine-tune the model. The long-term plan includes transitioning to an LLM/Transformer architecture (such as a localized BERT model) for a deeper semantic understanding of the text.
