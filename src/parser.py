import numpy as np
import pickle
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from PIL import Image as imageio

class Parser():

    def __init__(self, config): 
        self.config = config

    def parsing(self):
        config = self.config

        data_type = config['MODEL']['data']
        data_path = config['DATA'][data_type]['path']
        output_classes = config['DATA'][data_type]['output']
        # frmat = config['DATA'][data_type]['format']

        if data_type == 'base':

            # Read training data
            with open( os.path.join(data_path, 'trainset.pickle'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            train_data = dic['data']
            train_labels = dic['label']

            with open( os.path.join(data_path, 'validset.pickle'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            valid_data = dic['data']
            valid_labels = dic['label']

            self.data = np.concatenate((train_data, valid_data), axis=0)
            self.data = self.data.reshape((-1, 3, 32, 32))
            self.data = self.data.transpose([0, 2, 3, 1])
            labels = train_labels + valid_labels
            labels = np.asarray(labels)
            self.labels = np.zeros((len(labels), np.max(labels)+1))
            self.labels[np.arange(len(labels)), labels] = 1

        if data_type == 'cifar10':

            # Read training data
            with open( os.path.join(data_path, 'data_batch_1'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            batch1_data = dic[b'data']
            batch1_labels = dic[b'labels']

            with open( os.path.join(data_path, 'data_batch_2'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            batch2_data = dic[b'data']
            batch2_labels = dic[b'labels']

            with open( os.path.join(data_path, 'data_batch_3'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            batch3_data = dic[b'data']
            batch3_labels = dic[b'labels']

            with open( os.path.join(data_path, 'data_batch_4'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            batch4_data = dic[b'data']
            batch4_labels = dic[b'labels']

            with open( os.path.join(data_path, 'data_batch_5'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            batch5_data = dic[b'data']
            batch5_labels = dic[b'labels']

            # Change data format to
            # (# of files, width, height, channel)
            self.data = np.concatenate((batch1_data, batch2_data, batch3_data, batch4_data, batch5_data), axis=0)
            self.data = self.data.reshape((-1, 3, 32, 32))
            self.data = self.data.transpose(0, 2, 3, 1)
            labels = batch1_labels + batch2_labels + batch3_labels + batch4_labels + batch5_labels
            labels = np.asarray(labels)
            self.labels = np.zeros((len(labels), np.max(labels)+1))
            self.labels[np.arange(len(labels)), labels] = 1

        if data_type == 'cifar100':

            # Read training data
            with open( os.path.join(data_path, 'train'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')

            train_data = dic[b'data']
            #train_coarse_labels = np.asarray(dic[b'coarse_labels'])
            train_fine_labels = dic[b'fine_labels']
            train_labels = train_fine_labels

            with open( os.path.join(data_path, 'test'), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            test_data = dic[b'data']
            # test_coarse_labels = dic[b'coarse_labels']
            test_fine_labels = dic[b'fine_labels']
            test_labels = test_fine_labels

            # Change data format to
            # (# of files, width, height, channel)
            self.data = np.concatenate((train_data, test_data), axis=0)
            self.data = self.data.reshape((-1, 3, 32, 32))
            self.data = self.data.transpose(0, 2, 3, 1)
            labels = train_labels + test_labels
            labels = np.asarray(labels)
            self.labels = np.zeros((len(labels), np.max(labels)+1))
            self.labels[np.arange(len(labels)), labels] = 1

        if data_type == 'intel':
            self.data, self.labels = self.parse_intel(data_path)

        self.data = self.data / 255.
        return self.data, self.labels


    def parse_intel(self, data_path):
        '''
            buildings: 0, forest: 1, glacier: 2, mountain: 3, sea: 4, street: 5
        '''

        # TODO intel.pickle exists check
        if os.path.exists(os.path.join(data_path, 'intel1.pickle')):
            pickle_file = os.path.join(data_path, 'intel1.pickle')
            with open(pickle_file, 'rb') as f:
                dic1 = pickle.load(f, encoding='bytes')
            with open(pickle_file, 'rb') as f:
                dic2 = pickle.load(f, encoding='bytes')

            data1 = dic1['data']
            labels1 = dic1['labels']
            data2 = dic2['data']
            labels2 = dic2['labels']
            data = np.concatenate((data1, data2), axis=0)
            labels = np.concatenate((labels1, labels2), axis=0)

            return data, labels

        data = []
        labels = []
        train_folder        = os.path.join(data_path, 'seg_train')
        buliding_folder     = os.path.join(train_folder, 'buildings')
        forest_folder       = os.path.join(train_folder, 'forest')
        glacier_folder      = os.path.join(train_folder, 'glacier')
        mountain_folder     = os.path.join(train_folder, 'mountain')
        sea_folder          = os.path.join(train_folder, 'sea')
        street_folder       = os.path.join(train_folder, 'street')

        folders = {
            'building': buliding_folder,
            'forest': forest_folder,
            'glacier': glacier_folder,
            'mountain': mountain_folder,
            'sea': sea_folder,
            'street': street_folder
        }
        train_data, train_labels = self.get_intel_item(folders)

        test_folder         = os.path.join(data_path, 'seg_test')
        building_folder     = os.path.join(test_folder, 'buildings')
        forest_folder       = os.path.join(test_folder, 'forest')
        glacier_folder      = os.path.join(test_folder, 'glacier')
        mountain_folder     = os.path.join(test_folder, 'mountain')
        sea_folder          = os.path.join(test_folder, 'sea')
        street_folder       = os.path.join(test_folder, 'street')
        folders = {
            'building': building_folder,
            'forest': forest_folder,
            'glacier': glacier_folder,
            'mountain': mountain_folder,
            'sea': sea_folder,
            'street': street_folder
        }
        test_data, test_labels = self.get_intel_item(folders)
        data = np.concatenate((train_data, test_data), axis=0)
        labels = np.concatenate((train_labels, test_labels), axis=0)

        pickle_file1 = os.path.join(data_path, 'intel1.pickle')
        pickle_file2 = os.path.join(data_path, 'intel2.pickle')
        num_file1 = len(data) // 2
        num_file2 = len(data) - num_file1

        dic1 = {
            'data': data[:num_file1],
            'labels': labels[:num_file1]
        }
        dic2 = {
            'data': data[num_file1:],
            'labels': labels[num_file1:]
        }

        with open(pickle_file1, 'wb') as f:
            pickle.dump(dic1, f)
        with open(pickle_file2, 'wb') as f:
            pickle.dump(dic2, f)

        return data, labels


    def get_intel_item(self, folders):

        building    = folders['building']
        forest      = folders['forest']
        glacier     = folders['glacier']
        mountain    = folders['mountain']
        sea         = folders['sea']
        street      = folders['street']

        folders = [building, forest, glacier, mountain, sea, street]
        data = []
        labels = []
        count = 0

        for idx, folder in enumerate(folders):
            for item in os.listdir(folder):
                img = imageio.open( os.path.join(folder, item))

                try:
                    img_list = list(img.getdata())
                    img_np = np.asarray(img_list).reshape((150, 150, 3))
                except:
                    img = img.resize((150, 150))
                    img_list = list(img.getdata())
                    img_np = np.asarray(img_list).reshape((150, 150, 3))
                    
                data.append(img_np)
                labels.append(idx)

        print (len(data))
        print (len(labels))
        print ('ignored item: ' + str(count))
        return data, labels

    def get_data_description(self):
        data = self.data 
        labels = self.labels

        if len(data) != len(labels):
            return None

        return data[0].shape, labels.shape[1]


    def print_data_description(self):
        data = self.data 
        labels = self.labels

        if len(data) != len(labels):
            print ('Error: # of data and # of labels is not equal')
            print ('# of data: ' + str(len(data)))
            print ('# of labels: ' + str(len(labels)))
            return

        print ('# of data: ' + str(len(labels)))
        print ('Image size and channel: ' + str(data[0].shape))
        print ('Labels: ' + str(labels.shape))


    def showimg(self):
        img = self.data[0]
        print (img.shape)
        plt.imshow(img)
        plt.show()



if __name__ == "__main__":

    config = None
    with open(os.path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if config is not None:
        np.random.seed(1)
        parser = Parser(config)

        parser.parsing()
        parser.print_data_description()
        parser.showimg()