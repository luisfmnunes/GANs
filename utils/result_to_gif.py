import os
import sys
import argparse
from datetime import datetime
import imageio

def create_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("images_dir",type=str,help="Directory containing the result images")

    return parser.parse_args(argv)

def main(args):
    output_dir = "data/gifs"
    create_dir(output_dir)

    with imageio.get_writer(os.path.join(output_dir, f'{datetime.now().strftime("%Y%m%d")}_results.gif'), mode='I') as writer:
        for file in sorted([os.path.join(args.images_dir,f) for f in os.listdir(args.images_dir) if not 'hori' in f],
                            key = lambda x: ( int(x.split('_')[2]), int(os.path.splitext(x)[0].split('_')[-1]) ))[::10]:
            image = imageio.imread(file)
            writer.append_data(image)

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
