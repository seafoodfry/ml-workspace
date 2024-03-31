# Whisper Experiments

This experiment makes use of [github.com/openai/whisper](https://github.com/openai/whisper).

This can readily be executed as follows:
```
whisper test-recording.m4a --model small --output_format txt --verbose True --task transcribe --language Spanish
```

If you want to segment an audio file into 300s (5m) segments, run:
```
ffmpeg -i yourfile.m4a -f segment -segment_time 300 -c copy part_%03d.m4a
```

We tried it out with 2 cores and 3gb ram and it processes about 10s of audio per minute.
If you run on Mac, and you are curious, then you want to watch the Virtual Machine Service For Docker.
