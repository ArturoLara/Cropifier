import argparse

#read Args
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fit', type=bool, required=False)
    parser.add_argument('-predict', type=bool, required=False)

    parsed_args = parser.parse_args()

    videoPath = parsed_args.video
    outPath = parsed_args.out

    return videoPath, outPath
#save plot
#load CSV
#save CSV