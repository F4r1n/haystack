import json
from pprint import pprint

if __name__ == '__main__':
    # expected_version = '1.0'
    # parser = argparse.ArgumentParser(
    #     description='Evaluation for MLQA ' + expected_version)
    # parser.add_argument('dataset_file', help='Dataset file')
    # parser.add_argument('prediction_file', help='Prediction File')
    # parser.add_argument('answer_language', help='Language code of answer language')
    #
    # args = parser.parse_args()
    # with open(args.dataset_file) as dataset_file:
    #     dataset_json = json.load(dataset_file)
    #     if (str(dataset_json['version']) != expected_version):
    #         print('Evaluation expects v-' + expected_version +
    #               ', but got dataset with v-' + dataset_json['version'],
    #               file=sys.stderr)
    #     dataset = dataset_json['data']
    # with open(args.prediction_file) as prediction_file:
    #     predictions = json.load(prediction_file)
    # print(json.dumps(evaluate(dataset, predictions, args.answer_language)))

    # with open("/Users/PM/Desktop/MLQA_V1/test/test-context-de-question-de.json") as dataset_file:
    #     dataset_json = json.load(dataset_file)
    #     # dataset = dataset_json["data"]
    #     # print(dataset_json)
    #     # print(dataset_json["data"][0])
    #     # pprint(dataset_json["data"][0])
    #     for entry in dataset_json["data"]:
    #         print(entry["title"])
    #         print(len(entry["paragraphs"]))
    #         for paragraph in entry["paragraphs"]:
    #             print(len(paragraph["context"]))
    #             print(len(paragraph["qas"]))
    #             for qa in paragraph["qas"]:
    #                 print(qa["question"])
    #                 print(qa["answers"])
    #                 print(qa["id"])
    #         print("#" * 40)

    # MLQA json to multiple files converter
    # with open("/Users/PM/Desktop/MLQA_V1/test/test-context-de-question-de.json") as dataset_file:
    #     dataset_json = json.load(dataset_file)
    #     for entry in dataset_json["data"]:
    #         # print(entry["title"])
    #         with open("kbQA/data/MLQA_V1/" + entry["title"].replace(" ", "_").replace("/", "").replace("\\", "") + ".txt", "w", encoding="utf-8") as out_file:
    #             for paragraph in entry["paragraphs"]:
    #                 out_file.write(paragraph["context"] + "\n")

    #questions
    with open("/Users/PM/Desktop/MLQA_V1/test/test-context-de-question-de.json") as dataset_file:
        dataset_json = json.load(dataset_file)
        for entry in dataset_json["data"]:
            print(entry["title"])
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    print(qa["question"])
                    print(qa["answers"])
                    print(qa["id"])
            print("")
