from PIL import Image
import os.path
from os import mkdir

N = 12500

if not os.path.exists("processed"):
    os.mkdir("processed")

# cat
for i in xrange(N):
    path = "cat.{}.jpg".format(i)
    print "[*] processing {}...".format(path)
    f = Image.open("train/" + path)
    fo = f.resize((128, 128))
    fo.save("processed/" + path)

# dog
for i in xrange(N):
    path = "dog.{}.jpg".format(i)
    print "[*] processing {}...".format(path)
    f = Image.open("train/" + path)
    fo = f.resize((128, 128))
    fo.save("processed/" + path)

