
class TestFile:
    def __init__(self, test_file_name, test_file_caption):
        self.test_file_name = test_file_name
        self.test_file_caption = test_file_caption

    def write_test_file(self):
        test_file_dict = {
            'test_file_name':
                self.test_file_name,
            'test_file_caption':
                self.test_file_caption
        }
        return test_file_dict