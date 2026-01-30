# 🎓 EduAssist - Hệ thống Trợ lý Giáo dục AI

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

EduAssist là một hệ thống trợ lý giáo dục thông minh, sử dụng AI để hỗ trợ việc ghi âm, phiên âm, tóm tắt và trả lời câu hỏi từ nội dung cuộc họp/bài giảng. Hệ thống tích hợp công nghệ ASR (Automatic Speech Recognition), RAG (Retrieval-Augmented Generation), và LLM để cung cấp trải nghiệm học tập tương tác.

## ✨ Tính năng chính

- 🎤 **Nhận dạng giọng nói thời gian thực**: Sử dụng mô hình ChunkFormer để chuyển đổi giọng nói thành văn bản
- 📝 **Xử lý văn bản thông minh**: Tự động thêm dấu câu, chuẩn hóa văn bản tiếng Việt (Inverse Text Normalization)
- 🤖 **RAG (Retrieval-Augmented Generation)**: Tìm kiếm và trả lời câu hỏi dựa trên ngữ cảnh cuộc họp
- 📊 **Tóm tắt tự động**: Sử dụng LLM MapReduce để tóm tắt nội dung dài
- 💾 **Quản lý cơ sở dữ liệu**: Lưu trữ cuộc họp, transcript, tài liệu và lịch sử hội thoại
- 🔍 **Vector Search**: Sử dụng FAISS để tìm kiếm ngữ nghĩa hiệu quả
- 🖥️ **Giao diện người dùng**: Gradio UI thân thiện và dễ sử dụng

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐
│   Frontend UI   │ (Gradio)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI App   │ (REST API)
└────────┬────────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌─────────┐           ┌──────────────┐
│ Services│           │   Database   │
│  Layer  │           │ (PostgreSQL) │
└────┬────┘           └──────────────┘
     │
     ├─ ASR (ChunkFormer)
     ├─ Punctuation & Normalization
     ├─ RAG Processor
     ├─ LLM (Ollama/Local)
     └─ Vector Store (FAISS)
```

## 📁 Cấu trúc dự án

```
Fall25_Capstone_DataFlow/
├── api/                          # Backend API
│   ├── routes/                   # API endpoints
│   │   └── routes.py            # Định nghĩa các route
│   ├── services/                 # Business logic
│   │   ├── chunkformer_stt.py   # Speech-to-Text service
│   │   ├── local_llm.py         # Local LLM integration
│   │   ├── rag_processor.py     # RAG processing
│   │   ├── vcdb_faiss.py        # Vector database
│   │   ├── punctuation_processing.py
│   │   └── Vietnamese-Inverse-Text-Normalization/
│   ├── database/                 # Database layer
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── database.py          # Database config
│   │   └── crud.py              # CRUD operations
│   ├── utils/                    # Utilities
│   └── config.py                 # Configuration
├── frontend/                     # Giao diện người dùng
│   ├── new_ui.py                # UI chính (Gradio)
│   ├── asr_realtime_ui.py       # Real-time ASR UI
│   └── notebooklm_ui.py         # NotebookLM-style UI
├── Benchmark_Rag/               # Benchmark & evaluation
├── scripts/                      # Scripts tiện ích
│   └── init_database.py         # Database initialization
├── main.py                       # FastAPI application
├── requirements.txt              # Dependencies
└── README.md                     # Documentation (bạn đang đọc)
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python**: 3.12+
- **PostgreSQL**: 14+
- **Ollama**: (tùy chọn, cho LLM local)
- **CUDA**: (tùy chọn, cho GPU acceleration)

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd Fall25_Capstone_DataFlow
```

### Bước 2: Cài đặt dependencies

#### Trên Ubuntu/Linux:

```bash
# Cài đặt system dependencies
sudo apt install -y build-essential gcc g++ make cmake ninja-build pkg-config \
                    python3-dev python3.12-dev libopenblas-dev

# Tạo virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Cài đặt Python packages
pip install -r requirements.txt
```

#### Trên Windows:

```powershell
# Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate

# Cài đặt Python packages
pip install -r requirements.txt
```

### Bước 3: Cấu hình môi trường

Tạo file `.env` hoặc `api/private_config.py`:

```python
# Database
DATABASE_URL = "postgresql://user:password@localhost:5432/eduassist"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# API Keys (nếu dùng)
GOOGLE_API_KEY = "your-api-key"
OPENAI_API_KEY = "your-api-key"

# Paths
VECTORSTORE_DIR = "./vectorstores"
MODEL_DIR = "./model"
```

### Bước 4: Khởi tạo database

```bash
# Tạo database PostgreSQL
createdb eduassist

# Chạy migration
python scripts/init_database.py
```

### Bước 5: Download models

```bash
# Download ChunkFormer model (nếu chưa có)
# Model sẽ được đặt trong thư mục model/

# Pull Ollama model (nếu dùng Ollama)
ollama pull llama3.2:3b
```

## 🎯 Sử dụng

### Chạy Backend API

```bash
# Development mode
python main.py

# Hoặc với uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại: `http://localhost:8000`

### Chạy Frontend UI

```bash
# UI chính (tích hợp đầy đủ)
python frontend/new_ui.py

# Hoặc ASR real-time UI
python frontend/asr_realtime_ui.py
```

UI sẽ mở tại: `http://localhost:7860`

### API Endpoints chính

- `POST /api/meetings/create` - Tạo cuộc họp mới
- `POST /api/transcripts/add` - Thêm transcript
- `POST /api/chat` - Chat/Q&A với RAG
- `POST /api/summarize` - Tóm tắt nội dung
- `GET /api/meetings/{meeting_id}` - Lấy thông tin cuộc họp

Xem full API documentation tại: `http://localhost:8000/docs`

## 📖 Hướng dẫn sử dụng

### 1. Ghi âm và phiên âm

```python
# Khởi động ASR worker
start_asr()

# Hệ thống sẽ tự động:
# - Nhận diện giọng nói real-time
# - Thêm dấu câu
# - Chuẩn hóa văn bản tiếng Việt
# - Lưu vào database
```

### 2. Tìm kiếm và trả lời câu hỏi

```python
# Qua UI: Nhập câu hỏi vào chatbox
# Hệ thống sẽ:
# 1. Tìm kiếm context liên quan trong vectorstore
# 2. Kết hợp context với câu hỏi
# 3. Gửi đến LLM để tạo câu trả lời
```

### 3. Tóm tắt nội dung

```python
# Tóm tắt tự động được trigger khi:
# - Có đủ transcript mới
# - Người dùng request tóm tắt

# MapReduce LLM xử lý văn bản dài:
# Map → Collapse → Reduce
```

## 🔧 Cấu hình nâng cao

### Tinh chỉnh Vector Search

Trong `api/services/vcdb_faiss.py`:

```python
# Semantic chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Hybrid search
SIMILARITY_THRESHOLD = 0.7
TOP_K = 5
```

### Tinh chỉnh LLM

Trong `api/services/local_llm.py`:

```python
# Temperature cho creativity
temperature = 0.7

# Max tokens
max_tokens = 2048

# Context window
context_window = 4096
```

### Cấu hình Database

```python
# Connection pool
POOL_SIZE = 10
MAX_OVERFLOW = 20
POOL_PRE_PING = True
```

## 🧪 Testing & Benchmarking

```bash
# Chạy benchmark RAG
cd Benchmark_Rag
python run_benchmark.py

# Build benchmark database
python build_benchmark_db.py
```

## 📊 Database Schema

### Tables chính:

- **Meeting**: `id`, `name`, `start_time`, `end_time`, `metadata`
- **Transcript**: `id`, `meeting_id`, `content`, `timestamp`, `speaker`
- **Document**: `id`, `meeting_id`, `file_path`, `embedding_path`
- **Conversation**: `id`, `meeting_id`, `created_at`
- **Message**: `id`, `conversation_id`, `role`, `content`, `timestamp`
- **Summarize**: `id`, `meeting_id`, `content`, `timestamp`

Chi tiết xem: [api/database/DATABASE_SUMMARY.md](api/database/DATABASE_SUMMARY.md)

## 🛠️ Công nghệ sử dụng

### Core Technologies
- **FastAPI**: Web framework
- **PostgreSQL**: Database
- **SQLAlchemy**: ORM
- **Gradio**: UI framework

### AI/ML
- **ChunkFormer**: ASR model
- **FAISS**: Vector similarity search
- **Ollama**: Local LLM runtime
- **LangChain**: LLM orchestration
- **HuggingFace**: Embeddings

### Processing
- **Pynini**: Text normalization
- **PunctCapSegModelONNX**: Punctuation restoration
- **LLM MapReduce**: Long document processing

## 📝 Tài liệu tham khảo

- [API Documentation](api/README_DOC.md)
- [Database Setup](api/database/SETUP_DATABASE.md)
- [MapReduce Guide](api/services/README_MAPREDUCE.md)
- [ChunkFormer Model](api/services/chunkformer/README.md)

## 🐛 Troubleshooting

### Lỗi database connection

```bash
# Kiểm tra PostgreSQL đang chạy
sudo systemctl status postgresql

# Kiểm tra connection string
psql $DATABASE_URL
```

### Lỗi model loading

```bash
# Kiểm tra GPU memory
nvidia-smi

# Giảm batch size hoặc chuyển sang CPU
export CUDA_VISIBLE_DEVICES=""
```

### Lỗi Ollama

```bash
# Kiểm tra Ollama service
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

## 🤝 Đóng góp

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

**Fall 2025 Capstone Project**

- Project: DataFlow - EduAssist
- Institution: [Your Institution]
- Supervisor: [Supervisor Name]

## 📧 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ:
- Email: [your-email@example.com]
- GitHub Issues: [repository-url/issues]

---

⭐ Nếu project này hữu ích, hãy cho chúng tôi một star!
