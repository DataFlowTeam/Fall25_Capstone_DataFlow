"""
Meeting Assistant UI - NotebookLM Style
Clean, modern interface with smooth workflow
"""

import os
import sys
import threading
from typing import Optional
import asyncio
from queue import Queue, Empty
import time
import warnings
import base64
from datetime import datetime

import gradio as gr
import pynini
from punctuators.models import PunctCapSegModelONNX
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.chunkformer_stt import ChunkFormer
from api.private_config import *
from api.config import *
from api.services.vcdb_faiss import VectorStore
from api.services.local_llm import LanguageModelOllama
from api.services.rag_processor import RagProcessor
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from api.database.database import SessionLocal
from api.database import crud
from api.services.ollama_mapreduce import OllamaMapReducePipeline, load_config


warnings.filterwarnings("ignore")

# =========================
# GLOBAL STATE
# =========================
current_meeting_id: Optional[int] = None
current_meeting_title: str = ""
current_meeting_context: str = ""

meeting_faiss: Optional[VectorStore] = None
transcript_faiss: Optional[VectorStore] = None
cache_faiss: Optional[VectorStore] = None

# =========================
# ITN MODEL
# =========================
def init_itn_model(itn_model_dir: str):
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify/tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize/verbalize.far")
    
    reader_classifier = pynini.Far(classifier_far)
    reader_verbalizer = pynini.Far(verbalizer_far)
    return reader_classifier.get_fst(), reader_verbalizer.get_fst()

# =========================
# INIT MODELS
# =========================
chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
punc_model = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    ort_providers=["CPUExecutionProvider"],
)
itn_classifier, itn_verbalizer = init_itn_model(ITN_REPO)
llm = LanguageModelOllama("shmily_006/Qw3:4b_4bit", temperature=0.5)
model_embedding = HuggingFaceEmbeddings(
    model_name=MODEL_EMBEDDING,
    model_kwargs={"trust_remote_code": True}
)

config = load_config("/home/bojjoo/Code/EduAssist/api/services/config_ollama_mapreduce.yaml")
mapreduce_pipeline = OllamaMapReducePipeline(config)

# =========================
# QUEUES & LOCKS
# =========================
job_queue = Queue(maxsize=0)
summarizer_queue = Queue(maxsize=0)
embedding_queue = Queue(maxsize=0)

faiss_lock = threading.Lock()
transcript_lock = threading.Lock()
summary_lock = threading.Lock()
db_lock = threading.Lock()

# =========================
# ASYNC LOOP
# =========================
_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None

def start_async_loop():
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None:
        _ASYNC_LOOP = asyncio.new_event_loop()
        t = threading.Thread(target=lambda: asyncio.set_event_loop(_ASYNC_LOOP) or _ASYNC_LOOP.run_forever(), daemon=True)
        t.start()

def run_async(coro, timeout=None):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result(timeout=timeout)

start_async_loop()

# =========================
# UI STATE
# =========================
asr_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
transcript_text = ""
summary_text = ""

rag_processor = RagProcessor(
    job_queue=job_queue,
    embedding_queue=embedding_queue,
    n_commits_to_combine=3,
    overlap_m=1,
    timeout_sec=15.0,
)

# =========================
# WORKERS
# =========================
def worker_loop(worker_id: int):
    while True:
        try:
            text = job_queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            normalize_prompt = llm.normalize_text(current_meeting_context, text)
            normalized = run_async(llm.async_generate(normalize_prompt), timeout=60.0)
            
            if not normalized or normalized.strip().casefold() == "none":
                continue
            
            if cache_faiss and cache_faiss.is_already_retrieved(normalized, similarity_threshold=0.7):
                continue
            
            related_docs = ""
            if meeting_faiss:
                related_docs = run_async(meeting_faiss.hybrid_search(normalized), timeout=60.0)
            
            summarizer_queue.put({"utterance": normalized, "related_docs": related_docs})
            
            if cache_faiss:
                with faiss_lock:
                    cache_faiss.add_cache(normalized)
        except Exception as e:
            print(f"[Worker-{worker_id}] ERROR: {e}")
        finally:
            job_queue.task_done()

def embedding_worker():
    while True:
        try:
            item = embedding_queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            if isinstance(item, dict):
                clean_text = (item.get("text") or "").strip()
                start_ms = item.get("start_time_ms", 0)
                end_ms = item.get("end_time_ms", 0)
            else:
                clean_text = (item or "").strip()
                start_ms = end_ms = 0
            
            if clean_text and transcript_faiss:
                with faiss_lock:
                    transcript_faiss.add_transcript(clean_text, start_ms, end_ms)
        except Exception as e:
            print(f"[EmbeddingWorker] ERROR: {e}")
        finally:
            try:
                embedding_queue.task_done()
            except:
                pass

def summarizer_loop():
    global summary_text
    while True:
        try:
            item = summarizer_queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            utter = item.get("utterance", "")
            docs = item.get("related_docs", "")
            
            sum_prompt = SUMMARIZE_DOCUMENT_PROMPT.format(utterance=utter, related_docs=docs)
            summary = run_async(llm.async_generate(sum_prompt), timeout=60.0)
            
            with summary_lock:
                if summary_text:
                    summary_text = f"{summary_text}\n\n──────────\n{summary.strip()}"
                else:
                    summary_text = summary.strip()
            
            # Lưu summary vào database
            if current_meeting_id and summary.strip():
                try:
                    db = SessionLocal()
                    try:
                        crud.create_summarize(
                            db=db,
                            meeting_id=current_meeting_id,
                            content=summary.strip(),
                            summary_type="realtime",
                            title=f"Summary at {datetime.now().strftime('%H:%M:%S')}"
                        )
                    finally:
                        db.close()
                except Exception as db_err:
                    print(f"[Summarizer] DB ERROR: {db_err}")
        except Exception as e:
            print(f"[Summarizer] ERROR: {e}")
        finally:
            try:
                summarizer_queue.task_done()
            except:
                pass

# Start workers
for i in range(2):
    threading.Thread(target=worker_loop, args=(i+1,), daemon=True).start()
threading.Thread(target=summarizer_loop, daemon=True).start()
threading.Thread(target=embedding_worker, daemon=True).start()

# =========================
# CHUNKFORMER CALLBACK
# =========================
def on_update(event: str, payload: dict):
    global transcript_text
    
    with transcript_lock:
        if event == "partial":
            display = (payload.get("display") or "").strip()
            if display:
                transcript_text = display
        
        elif event == "commit":
            display = (payload.get("display") or payload.get("committed") or "").strip()
            if display:
                transcript_text = display
            
            new_commit = (payload.get("new_commit") or "").strip()
            if new_commit:
                rag_processor.process_new_commit(payload)
        
        elif event == "final_flush":
            text = (payload.get("text") or "").strip()
            if text:
                transcript_text = text
            rag_processor.flush_all(reason="final_flush")

# =========================
# ASR WORKER
# =========================
def asr_worker():
    try:
        chunkformer.chunkformer_asr_realtime_punc_norm(
            mic_sr=16000, stream_chunk_sec=0.5, lookahead_sec=0.5,
            left_context_size=128, right_context_size=32, max_overlap_match=32,
            vad_threshold=0.01, vad_min_silence_blocks=2,
            punc_model=punc_model, punc_window_words=100, punc_commit_margin_words=50,
            itn_classifier=itn_classifier, itn_verbalizer=itn_verbalizer,
            on_update=on_update, stop_event=stop_event, return_final=False,
        )
    except Exception as e:
        print(f"[ASR] Error: {e}")

# =========================
# DATABASE FUNCTIONS
# =========================
def create_meeting(title: str, description: str):
    global current_meeting_id, current_meeting_title, current_meeting_context
    global meeting_faiss, transcript_faiss, cache_faiss
    
    if not title.strip():
        return "⚠️ Vui lòng nhập tiêu đề cuộc họp", gr.update(visible=False)
    
    try:
        db = SessionLocal()
        try:
            meeting = crud.create_meeting(db, title=title, description=description)
            current_meeting_id = meeting.id
            current_meeting_title = meeting.title
            current_meeting_context = ""
            
            folder = f"meeting_{meeting.id}"
            meeting_faiss = VectorStore(folder+"/documents", model_embedding)
            transcript_faiss = VectorStore(folder+"/transcripts", model_embedding)
            cache_faiss = VectorStore(folder+"/cache", model_embedding)
            
            msg = f"""
### ✅ Cuộc họp đã được tạo!

**{title}**  
ID: `{meeting.id}` | Status: `{meeting.status}`

{description if description else '_Không có mô tả_'}

---
📎 **Bước tiếp theo:** Upload tài liệu liên quan đến cuộc họp
"""
            return msg, gr.update(visible=True)
        finally:
            db.close()
    except Exception as e:
        return f"❌ Lỗi: {e}", gr.update(visible=False)

def upload_documents(files):
    global current_meeting_context
    
    if current_meeting_id is None:
        return "⚠️ Vui lòng tạo cuộc họp trước!", ""
    
    if not files:
        return "⚠️ Vui lòng chọn ít nhất một tài liệu!", ""
    
    try:
        db = SessionLocal()
        try:
            all_chunks = []
            doc_names = []
            
            for file in files:
                filename = os.path.basename(file.name)
                file_type = os.path.splitext(filename)[1].lower().replace('.', '')
                file_size = os.path.getsize(file.name) if os.path.exists(file.name) else 0
                
                doc = crud.create_document(
                    db=db, meeting_id=current_meeting_id,
                    filename=filename, file_path=file.name,
                    file_type=file_type, file_size=file_size
                )
                doc_names.append(filename)
                
                chunks = meeting_faiss.recursive_chunking(file.name)
                all_chunks.extend(chunks)
                
                crud.update_document_embedding(
                    db=db, document_id=doc.id,
                    vector_store_path=f"./vectorstores/meeting_{current_meeting_id}/documents",
                    embedding_model=MODEL_EMBEDDING, chunk_count=len(chunks)
                )
            
            if all_chunks:
                faiss_db = meeting_faiss.create_vectorstore(all_chunks)
                meeting_faiss.faiss_save_local(faiss_db, "")
                # meeting_faiss.db = faiss_db
                meeting_faiss.load_vectorstore()

                documents = ""
                for i in chunks[:25]:
                    documents += i.page_content + "\n-----\n"


                question = "Tài liệu này nói về vấn đề gì, có những khái niệm nào cần lưu ý, hãy trả lời theo format 'Meeting Context:'"
                current_meeting_context = mapreduce_pipeline.run(documents, question, chunk_size=4096)

                crud.update_meeting(
                    db=db, meeting_id=current_meeting_id,
                    meeting_context=current_meeting_context
                )
            
            msg = f"""
### ✅ Tài liệu đã được xử lý!

**Đã upload:** {len(doc_names)} tài liệu  
**Chunks:** {len(all_chunks)} đoạn văn bản  

📄 Files:
{chr(10).join([f'- {name}' for name in doc_names])}

---
🎤 **Bước tiếp theo:** Chuyển sang tab "Ghi âm" để bắt đầu cuộc họp
"""
            preview = current_meeting_context
            return msg, preview
        finally:
            db.close()
    except Exception as e:
        return f"❌ Lỗi: {e}", ""

def start_recording():
    global asr_thread, transcript_text, summary_text
    
    if current_meeting_id is None:
        return gr.update(), gr.update(), "⚠️ Chưa tạo cuộc họp!"
    
    try:
        db = SessionLocal()
        try:
            crud.update_meeting(db, current_meeting_id, status="in_progress")
        finally:
            db.close()
    except:
        pass
    
    with transcript_lock:
        transcript_text = ""
    with summary_lock:
        summary_text = ""
    
    stop_event.clear()
    
    if asr_thread is None or not asr_thread.is_alive():
        asr_thread = threading.Thread(target=asr_worker, daemon=True)
        asr_thread.start()
        return gr.update(value=""), gr.update(value=""), "🎙️ Đang ghi âm..."
    else:
        return gr.update(), gr.update(), "✅ Đang ghi âm"

def stop_recording():
    global transcript_text
    
    if current_meeting_id is None:
        return "⚠️ Chưa có cuộc họp!", gr.update(visible=False)
    
    stop_event.set()
    rag_processor.flush_all(reason="stop")
    
    try:
        db = SessionLocal()
        try:
            with transcript_lock:
                final = transcript_text
            
            if final.strip():
                crud.create_transcript(
                    db=db, meeting_id=current_meeting_id,
                    content=final, duration_ms=0, language="vi"
                )
            
            crud.update_meeting(db, current_meeting_id, status="completed")
        finally:
            db.close()
        
        # Hiển thị modal xác nhận tạo biên bản
        return "⏹️ Đã dừng và lưu transcript", gr.update(visible=True)
    except Exception as e:
        return f"⚠️ Lỗi: {e}", gr.update(visible=False)


def generate_meeting_minutes():
    """
    Tạo biên bản cuộc họp bằng MapReduce pipeline
    """
    global transcript_text

    if current_meeting_id is None:
        return (
            "⚠️ Chưa có cuộc họp!",
            gr.update(visible=False),  # minutes_modal
            "",  # minutes_display
            gr.update(open=False)  # minutes_accordion
        )

    try:
        with transcript_lock:
            document = transcript_text

        if not document.strip():
            return (
                "⚠️ Không có transcript để tạo biên bản!",
                gr.update(visible=False),
                "",
                gr.update(open=False)
            )

        # Chạy MapReduce pipeline
        question = "Tóm tắt các ý chính của cuộc họp, trình bày rõ ràng thành từng mục nếu cần thiết"
        result = mapreduce_pipeline.run(document, question, chunk_size=4096)

        # Lưu biên bản vào database (vd: description)
        db = SessionLocal()
        try:
            crud.update_meeting(
                db=db,
                meeting_id=current_meeting_id,
                description=f"{result}\n\n---\n_Biên bản được tạo tự động từ transcript_"
            )
        finally:
            db.close()

        # ✅ Trả về: status, ẩn modal, nội dung biên bản, MỞ accordion
        return (
            "✅ Đã tạo biên bản cuộc họp thành công!",
            gr.update(visible=False),  # ẩn modal
            result,  # HIỂN THỊ FULL BIÊN BẢN
            gr.update(open=True)  # mở accordion
        )

    except Exception as e:
        return (
            f"❌ Lỗi khi tạo biên bản: {e}",
            gr.update(visible=False),
            "",
            gr.update(open=False)
        )


def cancel_meeting_minutes():
    """Hủy tạo biên bản"""
    return (
        "ℹ️ Đã hủy tạo biên bản",
        gr.update(visible=False),   # ẩn modal
        "",                         # clear minutes_display
        gr.update(open=False)       # đóng accordion
    )


def poll_ui():
    with transcript_lock:
        txt = transcript_text
    with summary_lock:
        sumtxt = summary_text
    return gr.update(value=txt), gr.update(value=sumtxt)

def chat_qa(history, message):
    if current_meeting_id is None:
        return (history or []) + [(message, "⚠️ Vui lòng tạo cuộc họp trước!")], ""
    
    if not message:
        return history, ""
    
    try:
        db = SessionLocal()
        try:
            db_history = crud.get_conversation_history(db, current_meeting_id, last_n=5)
            history_str = "\n\n".join([
                f"{'User' if h['role']=='human' else 'AI'}: {h['content']}" 
                for h in db_history
            ])
            
            reformulated = run_async(
                llm.reformulate_question(message, history_str, current_meeting_context),
                timeout=60.0
            )
            
            if reformulated.get("type") == 0:
                reply = run_async(
                    llm.normal_qa_handler(
                        reformulated["new_question"],
                        history_str, current_meeting_context
                    ), timeout=60.0
                )
            else:
                related_docs = ""
                related_transcript = ""
                
                if meeting_faiss:
                    related_docs = run_async(
                        meeting_faiss.hybrid_search(reformulated["new_question"]),
                        timeout=60.0
                    )
                
                if transcript_faiss and transcript_faiss.db:
                    related_transcript = run_async(
                        transcript_faiss.hybrid_search(reformulated["new_question"]),
                        timeout=60.0
                    )
                
                reply = run_async(
                    llm.rag_qa_handler(
                        reformulated["new_question"], history_str,
                        current_meeting_context, related_docs, related_transcript
                    ), timeout=60.0
                )
            
            crud.add_message(db, current_meeting_id, role="human", content=message)
            crud.add_message(db, current_meeting_id, role="ai", content=reply,
                           extra_data={"sources": ["documents", "transcripts"]})
        finally:
            db.close()
        
        return (history or []) + [(message, reply)], ""
    except Exception as e:
        return (history or []) + [(message, f"❌ Lỗi: {e}")], ""

def load_meetings():
    try:
        db = SessionLocal()
        try:
            meetings = crud.get_all_meetings(db, skip=0, limit=50)
            choices = [(f"{m.title} (ID: {m.id})", m.id) for m in meetings]
            return gr.update(choices=choices)
        finally:
            db.close()
    except:
        return gr.update(choices=[])


def select_meeting(meeting_id):
    global current_meeting_id, current_meeting_title, current_meeting_context
    global meeting_faiss, transcript_faiss, cache_faiss
    global transcript_text, summary_text

    if not meeting_id:
        return "⚠️ Vui lòng chọn cuộc họp!", "", "", "", "", []

    try:
        db = SessionLocal()
        try:
            meeting = crud.get_meeting(db, meeting_id)
            if not meeting:
                return "❌ Không tìm thấy cuộc họp!", "", "", "", "", []

            current_meeting_id = meeting.id
            current_meeting_title = meeting.title
            current_meeting_context = meeting.meeting_context or ""

            folder = f"meeting_{meeting.id}"
            meeting_faiss = VectorStore(folder + "/documents", model_embedding)
            transcript_faiss = VectorStore(folder + "/transcripts", model_embedding)
            cache_faiss = VectorStore(folder + "/cache", model_embedding)

            docs = crud.get_documents(db, meeting_id)
            transcript = crud.get_transcript(db, meeting_id)
            messages = crud.get_messages(db, meeting_id)
            summaries = crud.get_summarizes(db, meeting_id)

            # Load transcript vào transcript_text
            with transcript_lock:
                transcript_text = transcript.content if transcript else ""
            
            # Load summaries vào summary_text
            with summary_lock:
                if summaries:
                    summary_parts = []
                    for s in summaries:
                        summary_parts.append(s.content)
                    summary_text = "\n\n──────────\n".join(summary_parts)
                else:
                    summary_text = ""
            
            # Tạo chatbot history từ messages
            chatbot_history = []
            for msg in messages:
                if msg.role == "human":
                    # Tìm message AI tiếp theo
                    ai_msg = None
                    msg_index = messages.index(msg)
                    if msg_index + 1 < len(messages) and messages[msg_index + 1].role == "ai":
                        ai_msg = messages[msg_index + 1]
                        chatbot_history.append((msg.content, ai_msg.content))

            # HEADER ngắn
            header_md = f"""### 📋 {meeting.title}

ID: `{meeting.id}` · Status: `{meeting.status}`
"""

            # Danh sách tài liệu
            if docs:
                doc_lines = "\n".join([f"- {d.filename}" for d in docs[:20]])
                docs_block = f"📄 **Danh sách tài liệu ({len(docs)}):**\n{doc_lines}"
            else:
                docs_block = "📄 **Danh sách tài liệu:** _Chưa có tài liệu nào_"

            # Meeting context
            if meeting.meeting_context:
                ctx = meeting.meeting_context.strip()
                if len(ctx) > 1200:
                    ctx = ctx[:1200] + "..."
                context_block = f"🧠 **Meeting Context:**\n\n{ctx}"
            else:
                context_block = "🧠 **Meeting Context:** _Chưa có meeting context_"

            detail_md = f"""
**Transcript:** {'✅' if transcript else '❌'} ({transcript.word_count if transcript else 0} từ)  
**Tin nhắn:** {len(messages)} messages  
**Summaries:** {len(summaries)} tóm tắt

**Mô tả:** {meeting.description or '_Không có mô tả_'}

---
{docs_block}

---
{context_block}
"""

            status_msg = f"✅ Đã load cuộc họp ID={meeting.id}"
            return status_msg, header_md, detail_md, transcript_text, summary_text, chatbot_history
        finally:
            db.close()
    except Exception as e:
        return f"❌ Lỗi: {e}", "", "", "", "", []


import re

def load_meeting_cards():
    """
    Trả về danh sách meetings để hiển thị dạng card trên trang chủ.
    Mỗi phần tử là một list [markdown_text] để dùng với gr.Dataset.
    """
    try:
        db = SessionLocal()
        try:
            meetings = crud.get_all_meetings(db, skip=0, limit=50)
            samples = []
            for m in meetings:
                desc = (m.description or "").strip()
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                status = (m.status or "").strip()
                status_icon = "🟢" if status == "in_progress" else ("✅" if status == "completed" else "📁")
                md = f"""**{m.title}**  

ID: `{m.id}` · {status_icon} `{status}`  

_{desc or "Không có mô tả"}_
"""
                samples.append([md])
            return gr.update(samples=samples)
        finally:
            db.close()
    except Exception as e:
        print("[load_meeting_cards] ERROR:", e)
        return gr.update(samples=[])


def open_meeting_from_card(sample):
    """
    sample là [markdown_text] từ Dataset.
    Trả về:
    - Ẩn home_view, hiện meeting_view
    - Cập nhật meeting_header_box
    - Cập nhật status_box
    - Cập nhật nội dung chi tiết (meeting_detail_box) nhưng vẫn ẩn
    - Load transcript, summary và chatbot history
    """
    if not sample or not sample[0]:
        return (
            gr.update(visible=True),    # home_view
            gr.update(visible=False),   # meeting_view
            "### 📋 Chưa chọn cuộc họp",# meeting_header_box
            "_Chưa bắt đầu_",           # status_box
            "",                         # meeting_detail_box
            gr.update(visible=False),   # meeting_detail_group
            "",                         # transcript_display
            "",                         # summary_display
            [],                         # chatbot
        )
    text = sample[0]
    m = re.search(r"ID:\s*`(\d+)`", text)
    if not m:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "### 📋 Chưa chọn cuộc họp",
            "_Chưa bắt đầu_",
            "",
            gr.update(visible=False),
            "",
            "",
            [],
        )
    meeting_id = int(m.group(1))
    status_msg, header_md, detail_md, transcript_txt, summary_txt, chatbot_hist = select_meeting(meeting_id)

    return (
        gr.update(visible=False),   # home_view ẩn
        gr.update(visible=True),    # meeting_view hiện
        header_md,                  # meeting_header_box
        status_msg,                 # status_box
        detail_md,                  # meeting_detail_box (chưa hiện, chỉ set content)
        gr.update(visible=False),   # meeting_detail_group: vẫn ẩn, chờ bấm "xem chi tiết"
        transcript_txt,             # transcript_display
        summary_txt,                # summary_display
        chatbot_hist,               # chatbot history
    )



def create_meeting_and_go(title: str, description: str):
    msg, _ = create_meeting(title, description)   # create_meeting đã set current_meeting_id

    header_md = ""
    detail_md = ""
    try:
        if current_meeting_id is not None:
            db = SessionLocal()
            try:
                meeting = crud.get_meeting(db, current_meeting_id)
                if meeting:
                    docs = crud.get_documents(db, meeting.id)

                    # HEADER ngắn
                    header_md = f"""### 📋 {meeting.title}

ID: `{meeting.id}` · Status: `{meeting.status}`
"""

                    # docs
                    if docs:
                        doc_lines = "\n".join([f"- {d.filename}" for d in docs[:20]])
                        docs_block = f"📄 **Danh sách tài liệu ({len(docs)}):**\n{doc_lines}"
                    else:
                        docs_block = "📄 **Danh sách tài liệu:** _Chưa có tài liệu nào_"

                    # context
                    ctx = (meeting.meeting_context or "").strip()
                    if ctx:
                        if len(ctx) > 1200:
                            ctx = ctx[:1200] + "..."
                        context_block = f"🧠 **Meeting Context:**\n\n{ctx}"
                    else:
                        context_block = "🧠 **Meeting Context:** _Chưa có meeting context_"

                    detail_md = f"""
**Mô tả:** {meeting.description or '_Không có mô tả_'}
---
{docs_block}
---
{context_block}
"""
            finally:
                db.close()
    except Exception as e:
        print("[create_meeting_and_go] ERROR:", e)

    return (
        msg,                        # create_status
        header_md,                  # meeting_header_box
        gr.update(visible=False),   # home_view
        gr.update(visible=True),    # meeting_view
        detail_md,                  # meeting_detail_box
        gr.update(visible=False),   # meeting_detail_group (ẩn, chờ bấm "xem chi tiết")
    )


def go_home():
    """
    Quay lại trang chủ, reset transcript/summary/chatbot & dừng ghi âm nếu còn.
    Đồng thời ẩn box chi tiết meeting.
    """
    global transcript_text, summary_text
    stop_event.set()
    with transcript_lock:
        transcript_text = ""
    with summary_lock:
        summary_text = ""

    return (
        gr.update(visible=True),    # home_view
        gr.update(visible=False),   # meeting_view
        "_Chưa bắt đầu_",           # status_box
        "",                         # transcript_display
        "",                         # summary_display
        [],                         # chatbot (clear history)
        gr.update(visible=False),   # meeting_detail_group
        "",                         # meeting_detail_box
    )

# =========================
# GRADIO UI - NotebookLM Style
# =========================

# =========================
# HELPER: ENCODE LOGO
# =========================
def get_logo_base64():
    logo_path = "../images/vimeeting_logo.png"
    try:
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"[LOGO] Error loading logo: {e}")
        return ""

custom_css = """
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    padding: 0 24px 40px 24px;
}
.tab-nav button {
    font-size: 16px;
    font-weight: 500;
}
.meeting-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 16px;
    margin: 16px 0 24px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.meeting-header h1 {
    margin: 0;
}
.meeting-header img {
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.home-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}
.home-subtitle {
    color: #9ca3af;
    margin-bottom: 24px;
}
#meeting-grid .wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}
#meeting-grid .wrap > div {
    flex: 0 0 260px;
}
#meeting-grid .wrap > div > div {
    background: #111827;
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    cursor: pointer;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
#meeting-grid .wrap > div > div:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    background: #1f2937;
}
.upload-zone {
    border: 2px dashed #4f46e5;
    border-radius: 12px;
    padding: 20px;
    background: #020617;
}
.chat-container {
    border-radius: 16px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.35);
}
"""

with gr.Blocks(title="ViMeeting - NotebookLM Style", css=custom_css, theme=gr.themes.Soft()) as demo:
    logo_base64 = get_logo_base64()
    gr.HTML(f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <img src="{logo_base64}" alt="logo" style="height:48px; width:auto;">
            <div>
                <h1 style="margin:0; font-size:28px;">ViMeeting</h1>
                <p style="margin:0; font-size:14px; color:#e0e0e0;">Powered by DataFlow</p>
            </div>
        </div>
    """)
    with gr.Column(scale=0.5):
        gr.Markdown("Tạo, quản lý và ghi âm các cuộc họp của bạn.")


    # ==================== HOME PAGE ====================
    with gr.Column(visible=True) as home_view:
        gr.Markdown(
            "## Sổ ghi chú của tôi\n"
            "Các cuộc họp sẽ xuất hiện tại đây giống như các notebook trong NotebookLM.",
            elem_classes=["home-title"]
        )

        # Hàng nút tạo và làm mới
        with gr.Row():
            create_new_btn = gr.Button("➕ Tạo cuộc họp mới", variant="primary", size="sm", scale=0)
            refresh_home_btn = gr.Button("🔄", size="sm", variant="secondary", scale=0)

        # Form tạo cuộc họp — ẩn mặc định
        # === FORM TẠO CUỘC HỌP ===
        with gr.Group(visible=False) as create_meeting_box:
            with gr.Column():  # 👈 Bao tất cả trong 1 Column duy nhất, không dùng Row đầu tiên
                meeting_title = gr.Textbox(
                    label="Tiêu đề cuộc họp",
                    placeholder="VD: Họp kế hoạch Q1 2025",
                    lines=1
                )

                meeting_desc = gr.Textbox(
                    label="Mô tả (tùy chọn)",
                    placeholder="Thảo luận kế hoạch kinh doanh và mục tiêu...",
                    lines=3
                )

                with gr.Row():
                    create_btn = gr.Button("✅ Tạo", variant="primary")
                    cancel_create_btn = gr.Button("❌ Hủy", variant="secondary")

                create_status = gr.Markdown("")

        # Danh sách meetings
        gr.Markdown("### 📚 Các cuộc họp của tôi")

        meetings_grid = gr.Dataset(
            label="",
            components=[gr.Markdown()],
            samples=[],
            elem_id="meeting-grid"
        )

    # ==================== MEETING PAGE ====================
    with gr.Column(visible=False) as meeting_view:
        # Top bar: Back + header + nút xem chi tiết
        with gr.Row():
            back_btn = gr.Button("⬅️ Về trang chủ", variant="secondary", size="sm", scale=0)
            meeting_header_box = gr.Markdown("### 📋 Chưa chọn cuộc họp")

        with gr.Row():
            detail_btn = gr.Button("ℹ️ Xem chi tiết", size="sm", variant="secondary", scale=0)

        # Box chi tiết (ẩn mặc định)
        with gr.Group(visible=False) as meeting_detail_group:
            meeting_detail_box = gr.Markdown("")

        with gr.Accordion("📎 Tài liệu cuộc họp", open=False):
            with gr.Row(elem_classes=["upload-zone"]):
                with gr.Column():
                    file_input = gr.File(
                        label="Chọn tài liệu",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt"]
                    )
                    upload_btn = gr.Button("📤 Upload & Phân tích", variant="primary", size="lg")
            upload_status = gr.Markdown("")
            context_box = gr.Textbox(
                label="Meeting Context (rút ra tự động từ tài liệu)",
                lines=6,
                interactive=False
            )

        # Ghi âm + Q&A
        with gr.Row():
            start_btn = gr.Button("▶️ Bắt đầu ghi âm", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ Dừng ghi âm", variant="stop", size="lg")
            status_box = gr.Markdown("_Chưa bắt đầu_")
        
        # Modal xác nhận tạo biên bản
        with gr.Group(visible=False) as minutes_modal:
            gr.Markdown("### 📝 Tạo biên bản cuộc họp?")
            gr.Markdown("Bạn có muốn tạo biên bản tổng hợp từ transcript của cuộc họp không?")
            with gr.Row():
                create_minutes_btn = gr.Button("✅ Có, tạo biên bản", variant="primary", size="lg")
                cancel_minutes_btn = gr.Button("❌ Không, bỏ qua", variant="secondary", size="lg")
            minutes_status = gr.Markdown("")
        
        # Box hiển thị biên bản
        with gr.Accordion("📜 Biên bản cuộc họp", open=False) as minutes_accordion:
            minutes_display = gr.Textbox(
                show_label=False,
                placeholder="Biên bản sẽ được hiển thị ở đây sau khi tạo...",
                lines=20,
                interactive=False,
                max_lines=30
            )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Transcript")
                transcript_display = gr.Textbox(
                    show_label=False,
                    placeholder="Transcript sẽ hiển thị ở đây khi bắt đầu ghi âm...",
                    lines=30,
                    interactive=False,
                    max_lines=30
                )

            with gr.Column(scale=3, elem_classes=["chat-container"]):
                gr.Markdown("### 💬 Hỏi đáp")
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=650,
                    bubble_full_width=False,
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png")
                )
                with gr.Row():
                    chat_msg = gr.Textbox(
                        show_label=False,
                        placeholder="💭 Đặt câu hỏi về cuộc họp hoặc tài liệu...",
                        lines=2,
                        scale=9
                    )
                    send_btn = gr.Button("📤", scale=1, variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Tóm tắt & Insights")
                summary_display = gr.Textbox(
                    show_label=False,
                    placeholder="Các đoạn tóm tắt từ AI sẽ xuất hiện ở đây...",
                    lines=30,
                    interactive=False,
                    max_lines=30
                )

    # ==================== EVENT HANDLERS ====================
    # Toggle xem/ẩn chi tiết
    detail_visible = gr.State(False)


    def toggle_details(current_visible):
        """Nếu đang ẩn thì hiện, nếu đang hiện thì ẩn."""
        if current_visible:
            # đang mở => ẩn lại
            return gr.update(visible=False), False, "ℹ️ Xem chi tiết"
        else:
            # đang ẩn => mở ra
            return gr.update(visible=True), True, "🔽 Ẩn chi tiết"


    demo.load(fn=load_meeting_cards, outputs=[meetings_grid])
    refresh_home_btn.click(fn=load_meeting_cards, outputs=[meetings_grid])


    # Toggle hiển thị box tạo cuộc họp
    def show_create_box():
        return gr.update(visible=True)


    def hide_create_box():
        return gr.update(visible=False)


    create_new_btn.click(fn=show_create_box, outputs=[create_meeting_box])
    cancel_create_btn.click(fn=hide_create_box, outputs=[create_meeting_box])

    create_btn.click(
        fn=create_meeting_and_go,
        inputs=[meeting_title, meeting_desc],
        outputs=[
            create_status,  # msg tạo cuộc họp
            meeting_header_box,  # header ngắn trên meeting page
            home_view,  # ẩn
            meeting_view,  # hiện
            meeting_detail_box,  # nội dung chi tiết (context + docs)
            meeting_detail_group  # group chi tiết (ẩn/hiện)
        ]
    )

    meetings_grid.select(
        fn=open_meeting_from_card,
        inputs=[meetings_grid],
        outputs=[home_view, meeting_view, meeting_header_box, status_box, meeting_detail_box, meeting_detail_group,
                transcript_display, summary_display, chatbot]
    )

    back_btn.click(
        fn=go_home,
        outputs=[home_view, meeting_view, status_box, transcript_display, summary_display, chatbot,
                 meeting_detail_group, meeting_detail_box]
    )

    upload_btn.click(
        fn=upload_documents,
        inputs=[file_input],
        outputs=[upload_status, context_box]
    )

    start_btn.click(fn=start_recording, outputs=[transcript_display, summary_display, status_box])
    stop_btn.click(fn=stop_recording, outputs=[status_box, minutes_modal])

    create_minutes_btn.click(
        fn=generate_meeting_minutes,
        outputs=[minutes_status, minutes_modal, minutes_display, minutes_accordion],
        show_progress="full"  # 👈 cái này sẽ bật màn hình loading của Gradio
    )

    cancel_minutes_btn.click(
        fn=cancel_meeting_minutes,
        outputs=[minutes_status, minutes_modal, minutes_display, minutes_accordion]
    )

    timer = gr.Timer(value=0.3, active=True)
    timer.tick(fn=poll_ui, outputs=[transcript_display, summary_display])

    send_btn.click(fn=chat_qa, inputs=[chatbot, chat_msg], outputs=[chatbot, chat_msg])
    chat_msg.submit(fn=chat_qa, inputs=[chatbot, chat_msg], outputs=[chatbot, chat_msg])

    detail_btn.click(
        fn=toggle_details,
        inputs=[detail_visible],
        outputs=[meeting_detail_group, detail_visible, detail_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
