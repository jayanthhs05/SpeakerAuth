# Usage

Install the requirements:
```
pip install -r requirements.txt
```

Run:
```
python speaker_auth.py
```

Request Format (To register new user):
```
curl -X POST http://localhost:8000/enroll \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "audio_path": "user1_enroll.wav"}'

```

Request Format (To verify existing user):
```
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "audio_path": "user1_verify.wav"}'
```

Returns:
```
{
    "authenticated":true/false,
    "similarity_score":0.xxxxxx,
    "threshold":0.6,
    "transcription": "text goes here"
}
```

If there is some server error or file not found or any such error, the error will be returned with a status code != 200.

To change the threshold, edit the `THRESHOLD` parameter in the code file.
To change the whisper model, edit the `whisper.load_model(...)` to whatever size you need.