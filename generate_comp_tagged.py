import torch
import basic_model


# import advanced_model


def main():
    """reproduce competition files"""

    """LOAD DATA"""
    train_word_dict, train_pos_dict = basic_model.get_vocabs_counts(['train_5700_sentences.labeled'])
    train = basic_model.DependencyDataset(path='train_5700_sentences.labeled', word_dict=train_word_dict,
                                          pos_dict=train_pos_dict,
                                          word_embd_dim=100, pos_embd_dim=25, test=False,
                                          use_pre_trained=False, pre_trained_vectors_name='',
                                          min_freq=1,
                                          comp_mode=False)

    comp = basic_model.DependencyDataset(path='comp.unlabeled', word_dict=train_word_dict, pos_dict=train_pos_dict,
                                         test=[train.word_idx_mappings, train.pos_idx_mappings], comp_mode=True)
    comp_dataloader = basic_model.DataLoader(comp, shuffle=False)

    """BASIC MODEL"""
    basic_model_weights_path = None  # TODO CHANGE!!!
    trained_basic_model = basic_model.KiperwasserDependencyParser(dataset=train, hidden_dim=125, MLP_inner_dim=100,
                                                                  BiLSTM_layers=2,
                                                                  dropout_layers=0.0).cuda()
    trained_basic_model.load_state_dict(torch.load(basic_model_weights_path))
    basic_model.tag_file_save_output(model=trained_basic_model,
                                     dataloader=comp_dataloader,
                                     original_unlabeled_file='comp.unlabeled',
                                     result_path='comp_m1_308044296.labeled')

    """ADVANCED MODEL"""
    advanced_model_weights = None  # TODO CHANGE!!!
    trained_advanced_model = advanced_model.LOAD().cuda()  # TODO CHANGE!!!
    trained_advanced_model.load_state_dict(torch.load(advanced_model_weights))

    advanced_model.tag_file_save_output(model=trained_advanced_model,
                                        dataloader=comp_dataloader,
                                        original_unlabeled_file='comp.unlabeled',
                                        result_path='comp_m2_308044296.labeled')


if __name__ == "__main__":
    main()
