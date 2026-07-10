#!/usr/bin/env bash

# Lấy một Session ID mới từ API
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | grep -o '"session_id":"[^"]*' | grep -o '[^"]*$')

echo "Đã khởi tạo Session ID: $SESSION_ID"
echo "Bắt đầu chạy 4 Test Cases..."

# Test 1: CNTT (Tiếng Việt)
echo -e "\n\n=========================================================="
echo "TEST 1: CÔNG NGHỆ THÔNG TIN (Hỏi tiếng Việt)"
echo "Câu hỏi: Những phần mềm nào bị cấm sử dụng trong công ty?"
echo "=========================================================="
curl -s -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "'"$SESSION_ID"'",
    "question": "Những phần mềm nào bị cấm sử dụng trong công ty?",
    "model": "auto",
    "mode": "research",
    "stream": false,
    "ui_language": "vi",
    "research_topic": "information_systems"
  }' | jq -r '{answer: .answer, sources: [.sources[].file]}'

echo "Đang nghỉ 10 giây để tránh quá tải API..."
sleep 10

# Test 2: Pháp chế (Tiếng Nhật)
echo -e "\n\n=========================================================="
echo "TEST 2: PHÁP CHẾ (Hỏi tiếng Nhật)"
echo "Câu hỏi: 3rdWATCHへのログイン方法を教えてください。"
echo "=========================================================="
curl -s -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "'"$SESSION_ID"'",
    "question": "3rdWATCHへのログイン方法を教えてください。",
    "model": "auto",
    "mode": "research",
    "stream": false,
    "ui_language": "ja",
    "research_topic": "legal_compliance"
  }' | jq -r '{answer: .answer, sources: [.sources[].file]}'

echo "Đang nghỉ 10 giây để tránh quá tải API..."
sleep 10

# Test 3: Kế toán (Tiếng Việt)
echo -e "\n\n=========================================================="
echo "TEST 3: KẾ TOÁN (Hỏi tiếng Việt)"
echo "Câu hỏi: Cách cài đặt ứng dụng Rakuraku Seisan trên smartphone?"
echo "=========================================================="
curl -s -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "'"$SESSION_ID"'",
    "question": "Cách cài đặt ứng dụng Rakuraku Seisan trên smartphone?",
    "model": "auto",
    "mode": "research",
    "stream": false,
    "ui_language": "vi",
    "research_topic": "accounting"
  }' | jq -r '{answer: .answer, sources: [.sources[].file]}'

echo "Đang nghỉ 10 giây để tránh quá tải API..."
sleep 10

# Test 4: Hành chính tổng hợp (Tiếng Việt)
echo -e "\n\n=========================================================="
echo "TEST 4: TỔNG VỤ (Hỏi tiếng Việt)"
echo "Câu hỏi: Báo cáo tai nạn lao động cần những biểu mẫu nào?"
echo "=========================================================="
curl -s -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "'"$SESSION_ID"'",
    "question": "Báo cáo tai nạn lao động cần những biểu mẫu nào?",
    "model": "auto",
    "mode": "research",
    "stream": false,
    "ui_language": "vi",
    "research_topic": "general_affairs"
  }' | jq -r '{answer: .answer, sources: [.sources[].file]}'

echo -e "\n\n=== HOÀN TẤT ==="
