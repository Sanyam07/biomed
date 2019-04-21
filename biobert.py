import ipdb
from ner import NER
from run_ner import main_funct
import csv


class BioBert(NER):

    def __init__(self):
        self.data_dir = 'BC4CHEMD'
        self.config_root = 'config_dir'
        self.output_dir = 'bio_output_dir'
        self.ground_truth_dict = dict()

    def convert_ground_truth(self, data, *args, **kwargs):
        ground_truth = list()
        for line in data:
            if line:
                temp = line.split()
                token = temp[0]
                label = temp[3]
                ground_truth.append([token, label])
                self.ground_truth_dict[token] = label
            else:
                continue
        return ground_truth

    def read_dataset(self, file_dict=None, dataset_name=None, *args, **kwargs):
        """
        :param file_dict: dictionary
                    {
                        "train": "location_of_train",
                        "test": "location_of_test",
                        "dev": "location_of_dev",
                    }

        :param args:
        :param kwargs:
        :return: dictionary of iterables
                    Format:
                    {
                        "train":[
                                    [ Line 1 tokenized],
                                    [Line 2 tokenized],
                                    ...
                                    [Line n tokenized]
                                ],
                        "test": same as train,
                        "dev": same as train
                    }

        """
        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        output_train_filename = self.data_dir+"/train.tsv"
        output_test_filename = self.data_dir+"/test.tsv"
        output_dev_filename = self.data_dir+"/train_dev.tsv"
        output_devel_filename = self.data_dir+"/devel.tsv"
        dev_length = len(data['dev'])
        train_lines = [[line.split()[0], line.split()[3]] for line in data['train']]
        test_lines = [[line.split()[0], line.split()[3]] for line in data['test']]
        dev_lines = [[line.split()[0], line.split()[3]] for line in data['dev'][dev_length//4:]]
        devel_lines = [[line.split()[0], line.split()[3]] for line in data['dev'][:dev_length//4]]
        with open(output_train_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in train_lines:
                tsv_writer.writerow(row)
        with open(output_test_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in test_lines:
                tsv_writer.writerow(row)
        with open(output_dev_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in dev_lines:
                tsv_writer.writerow(row)
        with open(output_devel_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in devel_lines:
                tsv_writer.writerow(row)
        del train_lines
        del test_lines
        del dev_lines
        del devel_lines
        return data

    def train(self, data=None, *args, **kwargs):
        data_dir = self.data_dir
        init_checkpoint = '/content/gdrive/My Drive/biobert_pubmed/biobert_model.ckpt'
        vocab_file = self.config_root+'/vocab.txt'
        bert_config_file = self.config_root+'/bert_config.json'
        output_dir = self.output_dir
        do_train = True
        do_eval = True
        do_predict = False
        main_funct(data_dir=data_dir, init_checkpoint=init_checkpoint, vocab_file=vocab_file,
                   bert_config_file=bert_config_file, output_dir=output_dir, do_train=do_train, do_eval=do_eval,
                   do_predict=do_predict)

    def predict(self, data=None, *args, **kwargs):
        output_predictions = main_funct(mode='eval', model_name='chunking_model', data_dir=self.data_dir, size=self.data_size,
                   gdrive_mounted=self.gdrive_mounted)
        output_file_path = self.data_dir + "/output.txt"
        with open(output_file_path, 'w') as pred_op_f:
            for val in output_predictions:
                for vall in val:
                    if vall[0] != '<missing>':
                        true_label = self.ground_truth_dict[vall[0]]
                        pred_op_f.write(vall[0] + ' ' + true_label + ' ' + vall[1] + '\n')
                pred_op_f.write('\n')

    def evaluate(self, predictions=None, ground_truths=None, *args, **kwargs):
        main_funct(mode='eval', model_name='chunking_model', data_dir=self.data_dir, size=self.data_size,
                   gdrive_mounted=self.gdrive_mounted)
