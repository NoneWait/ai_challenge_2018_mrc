import os
from collections import Counter
import codecs
import json

def get_all_predict():
    datadir = os.path.join("log/answer")
    flist = os.listdir(datadir)  # 

    pp = {}
    for fdir in flist:
        path = os.path.join(datadir, fdir, "answer_old.json")
	print(path)
        with codecs.open(path, "r", encoding="utf-8") as fh:
            for line in fh.readlines():
                line = line.strip("\n").split("\t")
                assert (len(line) == 2)
                # if len(line)!=2:
                #     print(line)
                if line[0] in pp.keys():
                    pp[line[0]][line[1]] += 1
                else:
                    pp[line[0]] = Counter()
                    pp[line[0]][line[1]] += 1

    result = {}
    for key in pp:
        result[key] = pp[key].most_common()[0][0]
    
    print("init samples size", len(result))


    
    samples = []
    test_file = os.path.join("input/data")
    with codecs.open(test_file, "r", encoding="utf-8") as fh:
        for line in fh.readlines():
            samples.append(json.loads(line))

    cnt = 0
    for sample in samples:
        query_id = sample["query_id"]
        if str(query_id) not in result.keys():
            # print("the sample not" )
            result[str(query_id)] = sample['alternatives'].split("|")[0]
            cnt += 1

    print(cnt)

    return result


def save(answer_dict):
    with codecs.open("output/result", "w", encoding="utf-8") as fh:
        for key in answer_dict:
            fh.write(str(key) + "\t" + answer_dict[key] + "\n")


if __name__ == '__main__':
    answer_dict = get_all_predict()
    save(answer_dict)
    # print()
