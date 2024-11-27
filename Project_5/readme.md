Links/URLs to any videos you created and want to submit as part of your report.
    The Greek Letter dataset: 
        https://www.kaggle.com/datasets/sayangupta001/mnist-greek-letters
    The handwritten digits I created: 
        https://drive.google.com/drive/folders/1LfHkXGk0sNjQnSl2zgx3N13V9gTywv8-?usp=sharing
    The handwritten greek letters I created: 
        https://drive.google.com/drive/folders/1_FRO5c6sZ4b2kEGxNvJ-DoQov11vHwng?usp=sharing

What operating system and IDE you used to run and compile your code.
    Mac OS + VSCode

Instructions for running your executables.
    File                            Command Usage
    model_train.py                      [python3 model_train.py]
    evaluate.py                         [python3 evaluate.py path_to_model.pth]
    eval_handwritten.py                 [python3 eval_handwritten.py path_to_model.pth 
                                                 path_to_handwritten_digits_folder]
    exam_network.py                     [python3 exam_network.py path_to_model.pth]
    experience_model.py                 [python3 experience_model.py]
    greek_letters_recognizer.py         [python3 greek_letters_recognizer.py 
                                                 path_to_model.pth path_to_train_folder path_to_test_folder]
    full_greek_letters_recognizer.py    [python3 greek_letters_recognizer.py 
                                                 path_to_model.pth path_to_train_folder path_to_test_folder]


    For experience_model.py because each stage of experiment takes a lot time, I seperate them into four parts:
        exam_dropout_rates,  exam_num_epochs, exam_pooling_sizes and exam_dropout_rates_num_epochs
    To run the whole experience, simply comment/uncomment each function in order at line[113, 134, 155, 185]

Instructions for testing any extensions you completed.
    Download the dataset from Kaggle and run the executable
