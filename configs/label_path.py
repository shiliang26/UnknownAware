
def get_label_path(protocol):
    
    # This works as an example. Specify your own path of the dataset.
    if protocol == 'All_F2F':
        real_label_path = ['ff_270/train/real_train_label.json']
        fake_label_path = ['ff_270/train/df_train_label.json', \
                           'ff_270/train/fsw_train_label.json'\
                           'ff_270/train/nt_train_label.json']
        val_label_path = ['Faceforensics/excludes_hq/df_val_label.json', 'Faceforensics/excludes_hq/f2f_val_label.json', \
                          'Faceforensics/excludes_hq/fsw_val_label.json', 'Faceforensics/excludes_hq/nt_val_label.json']
        test_label_path = ['Faceforensics/excludes_hq/df_test_label.json', 'Faceforensics/excludes_hq/f2f_test_label.json', \
                          'Faceforensics/excludes_hq/fsw_test_label.json', 'Faceforensics/excludes_hq/nt_test_label.json']
        metrics = {'Train': ['Loss'], 'Test AUC': ['DF', 'F2F', 'FSW', 'NT'], 'Best Test AUC': ['DF', 'F2F', 'FSW', 'NT']}
    
    return real_label_path, fake_label_path, val_label_path, test_label_path, metrics
    