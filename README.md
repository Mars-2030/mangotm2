

## Local Installation and Setup

Follow these steps to set up and run the project on your local machine.

### Prerequisites

*   **Python 3.9**: This project is tested and configured for Python 3.9. You can check your version with `python --version`.
*   **Git**: Required to clone the repository.

### Step 1: Clone the Repository

Open your terminal or command prompt and clone the repository to your local machine.

```bash
git clone https://github.com/Mars-2030/mangotm2.git
cd mangotm2
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

*   **On macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
*   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

You should see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

### Step 3: Install Required Packages

Install all the necessary Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```


### Step 4: NLTK Data (Automatic Download)

The application is designed to automatically download the necessary NLTK data packages (`stopwords`, `punkt`, `wordnet`) on its first run. It will create a folder named `nltk_data_streamlit` in your project directory to store them.

When you run the app for the first time, you may see download progress in your terminal. This is normal.

---

## 🚀 How to Run the Application

Once you have completed the installation and setup, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

Your default web browser will automatically open a new tab with the running application.



## 📂 Project File Structure

```
mangotm2/
│
├── app.py                   # Main Streamlit application script
├── requirements.txt         # List of Python dependencies
│
├── NotoSansSC-Regular.ttf   # (You must add this) Font for Chinese word clouds
├── cn_stopwords.txt         # (You must add this) Stopwords list for Chinese
│
├── nltk_data_streamlit/     # (Auto-generated) Stores NLTK data packages
│
└── README.md                # This file
```
