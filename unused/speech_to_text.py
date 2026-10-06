import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/samplefirst10.mp3",
                           language = "hi",
                           task = "translate",
                           word_timestamps = False)

print(result["segments"])
chunks = []
for segment in result["segments"]:
    chunks.append({"id":segment["id"],
                    "start":segment["start"],
                    "end":segment["end"],
                    "text":segment["text"]})


with open("output.json","w") as f:
    json.dump(chunks , f)
