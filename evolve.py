'''
usage: python evolve.py [-c (compress)] [-d (decompress)] [-v video.mp4] [-i image.png]
- links to specific codevector file

for compression:
- if image: return encode(image)
- if video: for each frame in video, append encode(frame) to compressed video
- encode(image) should return time it takes to encode
- generate "filename".ev (random file extension, can be whatever we want)

for decompression:
- return decode(code)
- decode(code) should return time it takes to decode
- generate "filename".[jpg/mp4]
note: later we'll merge compress and decompress into one exectuable so that we can just run "evolve -c image.jpg" for compression and "evolve -d code.ev" for decompression on the command-line
'''
