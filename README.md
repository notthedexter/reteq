# SocialWiz - AI-Powered Conversation Assistant 🤖💬

A FastAPI-based AI chatbot service that helps you craft perfect messages for different social situations using advanced language models (Groq API with LLaMA 3.3 70B).

## 🌟 Features

### Mode 1: Message Rewriter
Rewrite your messages with different moods/tones:
- **Casual**: Light, friendly, and relaxed
- **Flirty**: Playful and charming
- **Nonchalant**: Cool and unbothered
- **Slap Back**: Clever, witty comebacks
- **Playful**: Fun and teasing
- **Assertive**: Confident and direct
- **Curious**: Genuinely interested
- **Serious**: Thoughtful and mature
- **Apologetic**: Understanding and remorseful
- **Encouraging**: Supportive and uplifting
- **Sarcastic**: Cleverly ironic
- **Empathetic**: Compassionate and connected

### Mode 2: Icebreaker Generator
Generate engaging conversation starters:
- If you...
- Between...
- If I was...
- Would you rather...
- What is something that...
- In a situation where...
- How would you handle...

### Mode 3: Curveball Handler ⚡ (NEW)
Handle awkward or confusing situations with various moods:
- **Casual**: Keep it light and easy-going
- **Flirty**: Turn awkward into playful
- **Nonchalant**: Stay cool and unbothered
- **Slap Back**: Deliver clever comebacks
- **Playful**: Use humor to diffuse tension
- **Assertive**: Address situations head-on
- **Curious**: Show genuine interest
- **Serious**: Handle with maturity
- **Apologetic**: Show understanding and care
- **Encouraging**: Be positive and supportive
- **Sarcastic**: Use witty humor playfully
- **Empathetic**: Deep emotional connection

**Bonus**: Supports screenshot analysis for visual context!

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API Key ([Get one here](https://console.groq.com))
- Google API Key (optional, for image analysis)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/notthedexter/socialwiz.git
cd socialwiz
```

2. **Create virtual environment**
```bash
python -m venv demoenv
# Windows
demoenv\Scripts\activate
# Linux/Mac
source demoenv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional: For Mode 3 image analysis
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Run the server**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 Usage Examples

### Mode 1: Rewrite Message
```bash
curl -X POST "http://localhost:8000/api/mode1/rewrite" \
  -H "Content-Type: application/json" \
  -d '{
    "original_message": "Hey, are you coming to the party tonight?",
    "response": "Probably, depends on how late it goes.",
    "mood": "casual",
    "personal_context": "Prefer low-key gatherings and leave early."
  }'
```

**Response:**
```json
{
  "rewritten_reply": "Yeah I might swing by! Just depends on the vibe - I usually don't stay super late but we'll see how it goes 😊"
}
```

### Mode 2: Generate Icebreaker
```bash
curl -X POST "http://localhost:8000/api/mode2/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "opener_type": "would_you_rather",
    "context": "We both love traveling and outdoor activities."
  }'
```

**Response:**
```json
{
  "icebreaker": "Would you rather explore a hidden temple in the jungle or camp under the northern lights?"
}
```

### Mode 3: Handle Curveball
```bash
curl -X POST "http://localhost:8000/api/mode3/handle" \
  -H "Content-Type: application/json" \
  -d '{
    "situation_description": "She just sent three laughing emojis and nothing else",
    "mood": "curious"
  }'
```

**Response:**
```json
{
  "curveball_reply": "Okay, what's so funny? Did I miss something? 😄",
  "image_analysis_used": false
}
```

## 🏗️ Project Structure

```
socialwiz/
├── app/
│   ├── core/
│   │   └── config.py
│   └── services/
│       ├── chatbot/           # Message Rewriter
│           ├── chatbot_router.py
│           ├── chatbot_schema.py
│           └── chatbot.py
├── main.py
├── requirements.txt
├── test_mode3.py
└── .env
```

## 🧪 Testing

### Test Mode 3
```bash
python test_mode3.py
```

### Manual Testing
Use the interactive API docs at `http://localhost:8000/docs`

## 🔧 Configuration

### Available Models (Groq)
- `llama-3.3-70b-versatile` (default, recommended)
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

### Temperature Settings
- Mode 1: 0.7 (balanced creativity)
- Mode 2: 0.8 (more creative icebreakers)
- Mode 3: 0.8 (creative curveball handling)

## 📝 Requirements

```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.10.0
python-dotenv==1.0.0
google-generativeai==0.3.2
httpx==0.25.2
groq==0.11.0
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/api/mode1/rewrite` | POST | Rewrite message with mood |
| `/api/mode2/generate` | POST | Generate icebreaker |
| `/api/mode3/handle` | POST | Handle curveball situation |

## 🎨 Mode 1 & Mode 3 Moods Explained

| Mood | Use When | Example Situation |
|------|----------|-------------------|
| **Casual** | Keep it light | Random emojis, unclear messages |
| **Flirty** | Turn awkward into playful | Unexpected compliment or tease |
| **Nonchalant** | Stay unbothered | Being tested, need to play it cool |
| **Slap Back** | Deliver witty comebacks | Being teased, want to counter smartly |
| **Playful** | Keep things fun | Lighten the mood, friendly banter |
| **Assertive** | Be direct and confident | Need to state your position clearly |
| **Curious** | Need clarification | Confusing or out-of-context message |
| **Serious** | Handle maturely | Important or sensitive topics |
| **Apologetic** | Fix misunderstanding | Accidentally offended someone |
| **Encouraging** | Boost confidence | They seem uncertain or awkward |
| **Sarcastic** | Playful banter | Being teased, want to tease back |
| **Empathetic** | Show understanding | They seem upset or vulnerable |

## 💡 Tips for Best Results

1. **Be Specific**: Provide clear context in your requests
2. **Choose Appropriate Mood**: Match the relationship and situation
3. **Combine Modes**: Use Mode 3 → Mode 1 for perfect replies
4. **Use Screenshots**: Enable image analysis for better context
5. **Iterate**: Try different moods to find the best fit

## 🔒 Security Notes

- Never commit `.env` file to version control
- Use environment-specific API keys
- Implement rate limiting for production
- Add authentication for sensitive deployments
- Validate and sanitize all user inputs

## 🚧 Roadmap

- [ ] Add more curveball moods
- [ ] Support for multiple languages
- [ ] Conversation history context
- [ ] User preferences and learning
- [ ] Voice message analysis
- [ ] Real-time suggestions
- [ ] Mobile app integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙋 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [@notthedexter](https://github.com/notthedexter)

## 🎉 Acknowledgments

- Powered by [Groq](https://groq.com) - Lightning-fast AI inference
- Uses [LLaMA 3.3 70B](https://huggingface.co/meta-llama) - Meta's powerful language model
- Built with [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- Optional image analysis with [Google Gemini Vision](https://ai.google.dev/)

---

Made with ❤️ for better conversations
