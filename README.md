# SRT Translate & Voice Workflow

Hướng dẫn này mô tả quy trình dịch phụ đề `.srt` sang tiếng Việt và tạo voice-over tiếng Việt từ phụ đề đã dịch.

## Yêu Cầu

Cần có:

- Python 3.11+
- `ffmpeg` và `ffprobe` trong PATH
- Các package Python đã cài trong môi trường hiện tại
- API OpenAI-compatible đang chạy ở `http://localhost:20128/v1` nếu dùng model `cx/gpt-5.5`

Nếu API cần key, đặt biến môi trường:

```powershell
$env:NINEROUTER_API_KEY="your_api_key"
```

## File Chính

- `srt_translate.py`: dịch SRT, giữ timeline và merge bản dịch về `.srt`.
- `srt_to_voice.py`: tạo voice-over từ `.srt`, có hỗ trợ đọc English/code terms bằng phát âm tiếng Việt.
- `*.srt`: phụ đề gốc tiếng Anh.
- `*.vi.srt`: phụ đề tiếng Việt.
- `*.english-map.py`: map phát âm cho từ tiếng Anh/code trong phụ đề Việt.

## Quy Ước Folder Output

Mặc định, khi không truyền `--work-dir` hoặc `--output`, `srt_translate.py` sẽ tạo một thư mục theo tên file SRT gốc.

Ví dụ file gốc:

```text
Factory Method Pattern_ Easy Guide for Beginners.srt
```

Sau khi chạy dịch, output mặc định sẽ là:

```text
Factory Method Pattern_ Easy Guide for Beginners\
  work\
    timeline.json
    text.jsonl
    translated.jsonl
    cache\
  Factory Method Pattern_ Easy Guide for Beginners.vi.srt
```

Khi tạo voice từ file `.vi.srt` nằm trong folder đó, output voice cũng nằm cùng folder:

```text
Factory Method Pattern_ Easy Guide for Beginners\
  Factory Method Pattern_ Easy Guide for Beginners.vi.english-map.py
  Factory Method Pattern_ Easy Guide for Beginners.vi.voice.wav
  Factory Method Pattern_ Easy Guide for Beginners.vi.voice-report.json
  voice_work\
```

## Dịch SRT

Ví dụ với file:

```text
Factory Method Pattern_ Easy Guide for Beginners.srt
```

### Cách Nhanh Nhất

```powershell
python .\srt_translate.py all ".\Factory Method Pattern_ Easy Guide for Beginners.srt" --models cx/gpt-5.5 --base-url http://localhost:20128/v1
```

File tiếng Việt sẽ được tạo tại:

```text
.\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.srt
```

### Chạy Từng Bước

Extract timeline và text:

```powershell
python .\srt_translate.py extract ".\Factory Method Pattern_ Easy Guide for Beginners.srt"
```

Dịch bằng model:

```powershell
python .\srt_translate.py translate --srt-file ".\Factory Method Pattern_ Easy Guide for Beginners.srt" --models cx/gpt-5.5 --base-url http://localhost:20128/v1
```

Merge thành file tiếng Việt:

```powershell
python .\srt_translate.py merge ".\Factory Method Pattern_ Easy Guide for Beginners.srt"
```

Kiểm tra bản dịch:

```powershell
python .\srt_translate.py validate ".\Factory Method Pattern_ Easy Guide for Beginners.srt"
```

Nếu báo `Missing translations: 0` là đủ dòng dịch.

## Tạo SRT Từ Video Bằng Web Upload

Nếu video chưa có file `.srt`, có thể chạy local web app để upload video hoặc audio rồi tự tạo `.srt` theo timeline, sau đó app sẽ tiếp tục tạo `.vi.srt` bằng pipeline dịch hiện có.

Cài thêm package cần thiết:

```powershell
pip install -r .\requirements.txt
```

Chạy web app local:

```powershell
python .\video_to_srt_web.py --host 127.0.0.1 --port 8765
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8765
```

App sẽ:

- nhận file video/audio upload
- trích audio bằng `ffmpeg`
- chia audio thành chunk 10 phút, overlap 1 giây
- gọi backend OpenAI-compatible local để lấy segment có timestamp
- ghép thành file `.srt` chuẩn timeline
- gọi `srt_translate.py all` để tạo thêm `.vi.srt`

Output mặc định:

```text
.\video_srt_outputs\<ten-video>\
  <ten-video>.srt
  <ten-video>.vi.srt
  job-report.json
  work\
```

Các tùy chọn hữu ích:

```powershell
python .\video_to_srt_web.py --output-dir .\my_outputs --transcribe-base-url http://localhost:20128/v1 --transcribe-model cx/gpt-5.5
python .\video_to_srt_web.py --translate-base-url http://localhost:20128/v1 --translate-model cx/gpt-5.5
```

Nếu dịch tiếng Việt lỗi, app vẫn giữ file `.srt` gốc và ghi lỗi vào `job-report.json`.
## Tạo Phonetic English Map

Trước khi tạo voice, nên tạo map phát âm cho các từ tiếng Anh/code như `Factory`, `Method`, `interface`, `object`.

Đặt biến path cho dễ chạy:

```powershell
$vi = ".\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.srt"
```

Tạo map và tự điền bằng model:

```powershell
python .\srt_to_voice.py $vi --scan-english-map --scan-lowercase-english --auto-fill-english-map
```

Lệnh này sẽ tạo file map mặc định nằm cạnh file `.vi.srt`:

```text
.\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.english-map.py
```

Ví dụ nội dung:

```python
PHONETIC_ENGLISH_MAP_UPDATE = {
    "Factory": "phác tơ ri",
    "method": "mé thợt",
    "interface": "in tờ phêis",
    "object": "ốp dếch",
}
```

## Tạo Voice Từ SRT

### Export TXT Cho J2TEAM TTS

Nếu muốn dùng J2TEAM TTS ở chế độ chuyển đổi hàng loạt, export mỗi cue thành một file `.txt` theo đúng thứ tự:

```powershell
python .\srt_to_voice.py $vi --export-txt
```

Với file `C# getters & setters 🔒.vi.srt`:

```powershell
python .\srt_to_voice.py ".\C# getters & setters 🔒\C# getters & setters 🔒.vi.srt" --export-txt
```

Output mặc định:

```text
.\C# getters & setters 🔒\C# getters & setters 🔒.vi.txt-batch\
  00001-cue-00001.txt
  00002-cue-00002.txt
  ...
  manifest.json
```

Trong J2TEAM TTS, bật chế độ chuyển đổi hàng loạt rồi chọn các file `.txt` trong folder này. `manifest.json` dùng để đối chiếu lại cue id, start time, end time nếu cần ghép audio về timeline sau.

Test vài cue đầu:

```powershell
python .\srt_to_voice.py ".\C# getters & setters 🔒\C# getters & setters 🔒.vi.srt" --export-txt --limit 5
```

Nếu muốn xuất cả cue rỗng hoặc cue dạng `[Music]`, dùng:

```powershell
python .\srt_to_voice.py $vi --export-txt --txt-include-skipped
```

### Ghép Audio Rời Từ J2TEAM

Sau khi export TXT, dùng J2TEAM TTS desktop để chuyển các file `.txt` thành audio rời trong cùng folder hoặc folder riêng:

```powershell
python .\srt_to_voice.py ".\C# getters & setters 🔒\C# getters & setters 🔒.vi.srt" --export-txt
```

Khi đã có các file audio như `00001-cue-00001.mp3`, `00002-cue-00002.wav`, hoặc tên bắt đầu bằng cùng prefix, ghép lại theo timeline trong `manifest.json`:

```powershell
python .\srt_to_voice.py ".\C# getters & setters 🔒\C# getters & setters 🔒.vi.srt" --assemble-audio --audio-input-dir ".\C# getters & setters 🔒\C# getters & setters 🔒.vi.txt-batch" --output ".\C# getters & setters 🔒\j2team.voice.wav"
```

Nếu không truyền `--audio-input-dir`, tool dùng mặc định folder `<tên srt>.txt-batch`. Nếu không truyền `--audio-manifest`, tool đọc `manifest.json` trong folder đó. Output mặc định của chế độ này là `<tên srt>.assembled.wav`, report mặc định là `<output>.voice-report.json`.

Các tùy chọn timeline hữu ích:

- `--max-speed 1.35`: cho phép tăng tốc audio dài tối đa 1.35 lần trước khi phải trim hoặc overflow.
- `--fit-mode trim`: giữ timeline chặt, nếu audio vẫn dài sau khi tăng tốc thì cắt đúng slot cue.
- `--fit-mode overflow`: giữ nguyên phần audio còn dài, nhưng report sẽ cảnh báo cue có thể lấn timeline sau.
- `--audio-missing silence`: nếu thiếu audio cho cue nào đó thì chèn silence đúng duration cue thay vì dừng.

### Tạo Voice Bằng Edge TTS

Mặc định tool dùng Edge TTS với giọng nam `vi-VN-NamMinhNeural`. Nếu file map nằm cạnh SRT theo tên mặc định, chỉ cần chạy:

```powershell
python .\srt_to_voice.py $vi --phonetic-english
```

Output mặc định:

```text
.\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.voice.wav
.\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.voice-report.json
```

Chỉ định map file thủ công:

```powershell
python .\srt_to_voice.py $vi --phonetic-english --phonetic-map-file ".\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.english-map.py"
```

Một số tùy chọn hữu ích:

```powershell
# Đổi sang giọng Edge khác nếu muốn
python .\srt_to_voice.py $vi --phonetic-english --voice "vi-VN-HoaiMyNeural"

# Tăng tốc giọng Edge TTS
python .\srt_to_voice.py $vi --phonetic-english --rate "+10%"

# Cho phép speed-up mạnh hơn khi câu dài
python .\srt_to_voice.py $vi --phonetic-english --max-speed 1.35

# Test vài cue đầu
python .\srt_to_voice.py $vi --phonetic-english --limit 5
```

## Quy Trình Từ Đầu Đến Cuối

```powershell
python .\srt_translate.py all ".\Factory Method Pattern_ Easy Guide for Beginners.srt" --models cx/gpt-5.5 --base-url http://localhost:20128/v1

$vi = ".\Factory Method Pattern_ Easy Guide for Beginners\Factory Method Pattern_ Easy Guide for Beginners.vi.srt"

python .\srt_to_voice.py $vi --scan-english-map --scan-lowercase-english --auto-fill-english-map

python .\srt_to_voice.py $vi --phonetic-english
```

## Lưu Ý Về Timeline

`srt_translate.py` hiện được cấu hình để:

- Dịch bám từng cue gốc.
- Không làm mượt bằng cách chuyển nội dung sang cue khác.
- Sửa overlap bằng cách cắt end time cue trước.

Nếu muốn giữ timeline y hệt file gốc, kể cả overlap, dùng:

```powershell
python .\srt_translate.py merge ".\Factory Method Pattern_ Easy Guide for Beginners.srt" --no-fix-overlaps
```

Nếu subtitle bị lệch đều so với video, dùng `--shift-ms`.

Ví dụ kéo subtitle sớm hơn 700ms:

```powershell
python .\srt_translate.py merge ".\Factory Method Pattern_ Easy Guide for Beginners.srt" --shift-ms -700
```

## Dọn File Tạm

Có thể xóa các thư mục/cache sau nếu muốn làm sạch:

```text
__pycache__
<tên file srt gốc>\work\cache
<tên file srt gốc>\voice_work
```

Không xóa `.venv` nếu vẫn cần môi trường Python.
