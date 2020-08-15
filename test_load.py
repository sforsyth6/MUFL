import os

x = os.listdir("/data/cyber/ember2018/test/benign/")

if "00011da19f2e2cb2cf99e5f5f71b1b9b2a978bcf77254e06db22c1b7c8fefa66.gz" in x:
    print ("yes")

os.path.getsize("/data/cyber/ember2018/test/benign/" + "00011da19f2e2cb2cf99e5f5f71b1b9b2a978bcf77254e06db22c1b7c8fefa66.gz")