from __future__ import print_function
from pyspark import SparkContext
import sys

if __name__ == "__main__":
    print("In Step 1")
    if len(sys.argv) != 3:
        print("Usage: wordcount  ", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="WordCount")
    print("In Step 2")
    text_file = sc.textFile(sys.argv[1])
    print("In Step 3")
    counts = text_file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    print("In Step 4")
    counts.saveAsTextFile(sys.argv[2])
    print("In Step 5")
    sc.stop()
    print("In Step 6")
