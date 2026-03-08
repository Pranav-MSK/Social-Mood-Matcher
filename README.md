# 🎭 Social Mood Matcher

**AI-Powered Image Sentiment & Social Caption Generator**

Social Mood Matcher is an intelligent web application that analyzes the **visual mood of an uploaded image** and automatically generates **engaging social media captions and hashtags**.

The system uses Artificial Intelligence to understand the **context, sentiment, and category** of an image and produces captions suitable for platforms like **Twitter, Instagram, and Facebook**.

---

# 👨‍💻 Authors

* **Pranav M S Krishnan**
* **Ganesh R**
* **Makesh P**
* **Santhosh R**

---

# 🚀 Features

### 📷 Image Mood Detection

Analyzes uploaded images to detect the **sentiment and vibe** of the scene.

### ✍️ AI Caption Generation

Automatically generates **creative captions** tailored to the detected mood.

### #️⃣ Smart Hashtag Suggestions

Suggests **relevant and trending hashtags** based on image content.

### 📊 Sentiment Visualization

Displays detected sentiment along with **confidence score**.

### 🎯 Platform Optimization

Ensures captions fit within **platform-specific character limits**.

### 🧠 Gemini AI Integration

Supports **Google Gemini API** for advanced image understanding.

### 🔧 Offline Mode

Uses **BLIP and NLP models locally** when Gemini API is not enabled.

### 📜 Caption History

Stores previously generated captions for easy reference.

---

# 🧠 Technologies Used

| Technology                    | Purpose                         |
| ----------------------------- | ------------------------------- |
| **Python**                    | Core programming language       |
| **Streamlit**                 | Interactive web interface       |
| **Google Gemini API**         | Advanced AI image understanding |
| **BLIP Model**                | Image captioning (local AI)     |
| **DistilBERT / NLP Models**   | Sentiment detection             |
| **Hugging Face Transformers** | AI model framework              |
| **Pillow (PIL)**              | Image processing                |

---

# ⚙️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/social-mood-matcher.git
cd social-mood-matcher
```

---

### 2️⃣ Create Virtual Environment (Optional)

```
python -m venv venv
```

Activate:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# 🔑 Gemini API Setup (Optional)

To enable advanced AI analysis:

1. Get a **Gemini API Key** from Google AI Studio
2. Create a `.env` file in the project root

```
GEMINI_API_KEY=your_api_key_here
```

If not provided, the app automatically uses **local AI models**.

---

# ▶️ Running the Application

Start the Streamlit app:

```
streamlit run app.py
```

Open the browser at:

```
http://localhost:8501
```

---

# 📸 How It Works

1. Upload an image
2. AI detects the **sentiment and category**
3. The system generates a **caption** based on the detected mood
4. Relevant **hashtags** are suggested
5. Character limits are enforced for the selected platform

---

# 🎬 Demo

Upload an image and click **Generate Caption & Hashtags** to see AI-powered results instantly.

---

# 📜 License

This project is developed for **educational and academic purposes**.

---

# ❤️ Acknowledgements

* Streamlit
* Hugging Face Transformers
* Google Gemini API
* BLIP Image Captioning Model
