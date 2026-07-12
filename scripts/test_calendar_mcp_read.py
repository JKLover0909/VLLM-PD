import asyncio
import json
from pathlib import Path
import subprocess

async def main():
    script_path = str(Path(__file__).parent.parent / "src" / "agent" / "run_calendar_mcp.js")
    print(f"Khởi chạy tiến trình Node con: node {script_path}")

    process = subprocess.Popen(
        ["node", script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Tắt buffer
    )

    # Gửi tin nhắn khởi tạo JSON-RPC chuẩn của MCP
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    print("👉 Gửi tin nhắn khởi tạo (initialize)...")
    process.stdin.write(json.dumps(init_msg) + "\n")
    process.stdin.flush()

    print("Đang đợi phản hồi từ stdout...")
    await asyncio.sleep(2)

    # Đọc dòng phản hồi đầu tiên từ stdout
    try:
        # Đọc 1 dòng từ stdout
        line = process.stdout.readline()
        print("\n✅ NHẬN PHẢN HỒI JSON-RPC TỪ SERVER:")
        print(line)

        # Gửi tiếp tin nhắn list_tools
        list_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        print("\n👉 Gửi tin nhắn lấy danh sách tools (tools/list)...")
        process.stdin.write(json.dumps(list_msg) + "\n")
        process.stdin.flush()

        # Đọc phản hồi list_tools
        line = process.stdout.readline()
        print("\n✅ DANH SÁCH TOOLS TRẢ VỀ:")
        data = json.loads(line)
        tools = data.get("result", {}).get("tools", [])
        print(f"Số lượng tools: {len(tools)}")

        # Sửa lỗi truy cập thuộc tính dict
        for t in tools:
            print(f"- {t['name']}: {t['description'][:100]}...")

        # Tìm tool list-events/get-events của package để chạy thử
        read_tool_name = next((t['name'] for t in tools if 'list' in t['name'] or 'get' in t['name']), None)
        if read_tool_name:
            print(f"\n👉 Sẽ gọi thử công cụ đọc lịch: {read_tool_name}")
            call_msg = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": read_tool_name,
                    "arguments": {
                        "calendarId": "primary",
                        "maxResults": 3
                    }
                }
            }
            process.stdin.write(json.dumps(call_msg) + "\n")
            process.stdin.flush()

            # Đọc kết quả thực thi
            line = process.stdout.readline()
            print("\n✅ KẾT QUẢ ĐỌC LỊCH THẬT TỪ GOOGLE CALENDAR:")
            res_data = json.loads(line)
            content = res_data.get("result", {}).get("content", [])
            if content:
                print(content[0].get("text", "")[:1200] + "..." if len(content[0].get("text", "")) > 1200 else content[0].get("text", ""))
            else:
                print("Không có nội dung trả về:", res_data)
        else:
            print("❌ Không tìm thấy công cụ đọc lịch phù hợp.")

    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        process.kill()

if __name__ == "__main__":
    asyncio.run(main())
