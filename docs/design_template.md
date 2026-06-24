# Design Template

## Problem

Hệ thống nhận một câu hỏi nghiên cứu mở (ví dụ: *"Giải thích multi-agent systems và lợi ích so với single-agent"*) và phải trả về một câu trả lời có cấu trúc, dựa trên nguồn (source-grounded), phù hợp với đối tượng đọc (`audience`, mặc định "technical learners").

Đầu vào là `ResearchQuery` (`query`, `max_sources`, `audience`); đầu ra là `final_answer` cùng với trace đầy đủ và các metric benchmark để so sánh với baseline single-agent.

## Why multi-agent?

Một single-agent phải gộp tất cả công đoạn — tìm nguồn, phân tích, viết, tự đánh giá — vào một lượt suy luận duy nhất, dẫn đến:

- **Trộn lẫn trách nhiệm**: prompt phình to, mô hình dễ bỏ qua bước (ví dụ viết luôn mà chưa tra cứu nguồn).
- **Không có vòng kiểm soát chất lượng**: không có bước phản biện độc lập để bắt lỗi/yêu cầu sửa.
- **Khó quan sát**: không tách được latency/cost theo từng công đoạn.

Tách thành các agent chuyên biệt cho phép mỗi agent có prompt hẹp, chất lượng cao hơn, đồng thời thêm vòng **critic → revise** có giới hạn để cải thiện đáp án trước khi dừng.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Định tuyến: chọn worker kế tiếp và điều kiện dừng dựa trên trạng thái + verdict | `ResearchState` (iteration, các *_notes, agent_results) | `next_route` (`Route`), cập nhật `route_history`/`iteration`/`trace` | Vòng lặp vô hạn → chặn bằng `max_iterations` (trả `DONE`) |
| Researcher | Tra cứu nguồn qua `SearchClient`, tổng hợp ghi chú nghiên cứu | `request.query`, `max_sources` | `sources`, `research_notes`, `agent_results`, `trace` | Search lỗi/0 kết quả → `AgentExecutionError` sau retry ở service layer |
| Analyst | Phân tích & cô đọng nguồn thành luận điểm | `sources`, `research_notes` | `analysis_notes`, `agent_results`, `trace` | LLM lỗi → `AgentExecutionError`; thiếu input → supervisor route lại |
| Writer | Soạn câu trả lời cuối theo `audience` | `analysis_notes`, `request` | `final_answer`, `agent_results`, `trace` | Output rỗng → critic trả `revise` |
| Critic | Chấm điểm & verdict (`accept`/`revise`) cho `final_answer` | `final_answer` | `agent_results` (metadata `verdict`, `quality_score`), `trace` | Không parse được verdict → coi như cần thêm vòng (bounded by max_iterations) |

## Shared state

`ResearchState` (Pydantic) là nguồn sự thật duy nhất, được truyền qua mọi node:

- `request: ResearchQuery` — đầu vào gốc (query, max_sources, audience).
- `iteration: int` / `route_history: list[str]` / `next_route: str | None` — đếm bước và lịch sử định tuyến để supervisor ra quyết định + chống lặp.
- `sources: list[SourceDocument]` — nguồn researcher thu thập (để analyst/writer trích dẫn).
- `research_notes` / `analysis_notes` / `final_answer` — sản phẩm trung gian của từng worker; **sự hiện diện (None hay không) chính là tín hiệu định tuyến** cho supervisor.
- `agent_results: list[AgentResult]` — log có cấu trúc mỗi lượt agent (content, metadata gồm `cost_usd`, `verdict`, `quality_score`) → dùng cho benchmark & critic.
- `trace: list[dict]` — sự kiện quan sát (observability).
- `errors: list[str]` — gom lỗi không chặn luồng.
- Helper: `record_route()` (append history + tăng iteration), `add_trace_event()`.

## Routing policy

Supervisor là entry point; mỗi worker chạy xong quay lại supervisor (vòng lặp do supervisor điều khiển). Logic `decide()`:

```
supervisor
  ├─ iteration >= max_iterations ............. DONE
  ├─ verdict (sau lần writer cuối) == accept .. DONE
  ├─ research_notes is None .................. RESEARCHER
  ├─ analysis_notes is None .................. ANALYST
  ├─ final_answer is None .................... WRITER
  ├─ verdict is None ........................ CRITIC
  └─ verdict == revise ...................... WRITER  (vòng cải thiện)

worker (researcher/analyst/writer/critic) ──> supervisor
DONE ──> END
```

Luồng điển hình (offline, fake backend): `supervisor → researcher → analyst → writer → critic → (accept) → done`. Nếu critic trả `revise`, quay lại `writer` rồi `critic` cho tới khi `accept` hoặc chạm `max_iterations`.

## Guardrails

- **Max iterations**: `Settings.max_iterations` (mặc định 6, ràng buộc 1–20). Supervisor trả `DONE` khi `iteration >= max_iterations` — chặn vòng critic↔writer lặp vô hạn.
- **Timeout**: `Settings.timeout_seconds` (mặc định 60, ràng buộc 5–600) cho call mạng.
- **Retry**: thuộc service layer (`llm_client`/`search_client`), không nằm trong agent.
- **Fallback**: backend `fake` (mặc định) cho phép chạy hoàn toàn offline khi thiếu API key; `accept` của critic là điều kiện dừng "thành công sớm".
- **Validation**: mọi I/O qua schema Pydantic (`ResearchQuery.query` min_length=5, `quality_score` 0–10…); lỗi nghiệp vụ ném `AgentExecutionError`/`ValidationError`, stub chưa làm ném `StudentTodoError`.

## Benchmark plan

`run_benchmark()` đo từng run và trả `BenchmarkMetrics` (latency, estimated_cost_usd gộp từ `agent_results[*].metadata.cost_usd`, quality_score).

| Query | Metric | Expected outcome |
|---|---|---|
| "Giải thích multi-agent systems và lợi ích" | latency_seconds | multi-agent **chậm hơn** baseline (nhiều lượt LLM) |
| (cùng query) | estimated_cost_usd | multi-agent **tốn hơn** (nhiều token/call) |
| (cùng query) | quality_score (heuristic, hoặc `--judge` dùng LLM-judge) | multi-agent **điểm cao hơn** nhờ vòng research→analyze→critique |

Kết luận kỳ vọng: multi-agent đánh đổi latency/cost lấy chất lượng cao hơn — đúng động lực của thiết kế. Chạy: `malab benchmark --query "..."` (so sánh baseline vs multi-agent, xuất report qua `evaluation/report.py`).

### Kết quả thật (backend `openrouter` + `tavily`, model `gpt-4o-mini`)

| Run | Latency (s) | Tokens | Quality (0-10) | Citation cov. | Failure rate |
|---|---:|---:|---:|---:|---:|
| single | 11.2 | 803 | 5.0 | 0% | 0% |
| multi | 31.3 | 4 652 | 7.0 | 0% | 0% |

5 metric đo được: **latency** (wall-clock), **cost = token usage** (OpenRouter không trả USD nên dùng token làm proxy, theo gợi ý lab guide), **quality** (heuristic 0–10, có thể đổi sang LLM-judge bằng `--judge`), **citation coverage**, **failure rate**.

Quan sát: multi-agent tốn ~2.8× thời gian và ~5.8× token nhưng chất lượng cao hơn (7.0 vs 5.0). **Citation coverage = 0%** là số liệu trung thực, không phải bug: heuristic chỉ tính khi *title nguồn xuất hiện nguyên văn* trong câu trả lời, mà Writer diễn giải lại thay vì dán nguyên title → đây là hạn chế đã biết của metric, có thể cải thiện bằng cách yêu cầu Writer chèn citation tường minh hoặc đối khớp theo URL/fuzzy.

## Trace explanation (ví dụ thật)

Một lần chạy multi-agent thật (`reports/trace_sample.json`), supervisor điều phối theo đúng routing policy:

`route_history = [researcher → analyst → writer → critic → done]` (5 lượt định tuyến)

| # | Agent | Vào (làm gì) | Ra | Tokens (in/out) | Ghi chú |
|---|---|---|---|---:|---|
| 1 | researcher | Tavily search query | 5 nguồn + `research_notes` | 1164 / 418 | nguồn thật (SuperAnnotate, Medium, Reddit, Philschmid, Dataiku) |
| 2 | analyst | từ `research_notes` | `analysis_notes` | 454 / 367 | cô đọng luận điểm |
| 3 | writer | từ analysis + audience | `final_answer` | 851 / 605 | bản nháp v1 |
| 4 | critic | chấm `final_answer` | verdict=**accept**, score=8.0 | 674 / 119 | đạt ngưỡng (≥7.0) |
| 5 | supervisor | thấy verdict=accept | route=**done** | — | dừng đúng điều kiện |

**Đọc trace để debug:**
- *Ai làm gì*: cột Agent + `state.trace` (mỗi sự kiện có `name` + `payload`).
- *Tốn bao nhiêu*: `agent_results[*].metadata.input_tokens/output_tokens` → tổng 4 652 tokens.
- *Sai ở đâu*: nếu critic trả `revise`, supervisor route lại `writer` (vòng cải thiện) cho tới `accept` hoặc chạm `max_iterations=6`; lỗi agent được gom vào `state.errors` và `failure_rate` của benchmark. (Ở một lần chạy khác, critic trả `revise` (8.0) rồi `writer→critic` lần 2 mới `accept` (9.0) — minh chứng vòng critic hoạt động.)

## Exit ticket

**1. Case nào NÊN dùng multi-agent? Vì sao?**
Tác vụ research mở, nhiều bước dị nhất (tìm nguồn → phân tích → viết → tự phản biện) và cần *kiểm soát chất lượng*. Tách vai trò giúp mỗi agent có prompt hẹp, chính xác hơn, và vòng critic→revise bắt lỗi trước khi trả lời. Bằng chứng thật: quality 7.0 (multi) vs 5.0 (single) cho cùng câu hỏi — đáng để đánh đổi latency/cost khi chất lượng quan trọng hơn tốc độ/chi phí.

**2. Case nào KHÔNG nên dùng multi-agent? Vì sao?**
Tác vụ đơn giản, một bước, nhạy latency/cost (ví dụ Q&A ngắn, lookup, phân loại). Multi-agent tốn ~2.8× thời gian và ~5.8× token nhưng không cải thiện đáng kể những tác vụ vốn một single-agent đã làm tốt — overhead điều phối + nhiều lượt LLM là lãng phí. Khi đó single-agent baseline là lựa chọn đúng.
