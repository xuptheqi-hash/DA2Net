
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = r"/home/pod/shared-nvme/data/LEVIR_remake"
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir =  r"D:\master\PyCharm\PyCharm\Project\dataset\WHU-CD-256"
        elif data_name == 'SYSU':
            self.label_transform = "norm"
            self.root_dir = '/root/shared-nvme/data/SYSU'
        elif data_name == 'CAU':
            self.label_transform = "norm"
            self.root_dir = '/root/shared-nvme/data/CAU_CD'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

