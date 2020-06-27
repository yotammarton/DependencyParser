import torch
import basic_model
import advanced_model


def main():
    """reproduce competition files"""
    """BASIC MODEL"""
    train_word_dict_basic, train_pos_dict_basic = basic_model.get_vocabs_counts(['train_5700_sentences.labeled'])
    train_basic = basic_model.DependencyDataset(path='train_5700_sentences.labeled', word_dict=train_word_dict_basic,
                                                pos_dict=train_pos_dict_basic,
                                                word_embd_dim=300, pos_embd_dim=75, test=False,
                                                use_pre_trained=False, pre_trained_vectors_name='',
                                                min_freq=2,
                                                comp_mode=False)
    comp_basic = basic_model.DependencyDataset(path='comp.unlabeled', word_dict=train_word_dict_basic,
                                               pos_dict=train_pos_dict_basic,
                                               test=[train_basic.word_idx_mappings, train_basic.pos_idx_mappings],
                                               comp_mode=True)
    comp_basic_dataloader = basic_model.DataLoader(comp_basic, shuffle=False)
    basic_model_weights_path = 'basic_model_comp_weights.pt'
    trained_basic_model = basic_model.KiperwasserDependencyParser(dataset=train_basic, hidden_dim=125,
                                                                  MLP_inner_dim=100,
                                                                  BiLSTM_layers=2,
                                                                  dropout_layers=0.0).cuda()
    trained_basic_model.load_state_dict(torch.load(basic_model_weights_path))
    basic_model.tag_file_save_output(model=trained_basic_model,
                                     dataloader=comp_basic_dataloader,
                                     original_unlabeled_file='comp.unlabeled',
                                     result_path='comp_m1_308044296.labeled')

    """ADVANCED MODEL"""
    train_word_dict_adv, train_pos_dict_adv = advanced_model.get_vocabs_counts(['train_5700_sentences.labeled'])
    train_advanced = advanced_model.DependencyDataset(path='train_5700_sentences.labeled',
                                                      word_dict=train_word_dict_adv,
                                                      pos_dict=train_pos_dict_adv,
                                                      word_embd_dim=300, pos_embd_dim=100, test=False,
                                                      use_pre_trained=False, pre_trained_vectors_name='',
                                                      min_freq=3,
                                                      comp_mode=False)
    comp_advanced = advanced_model.DependencyDataset(path='comp.unlabeled', word_dict=train_word_dict_adv,
                                                     pos_dict=train_pos_dict_adv,
                                                     test=[train_advanced.word_idx_mappings,
                                                           train_advanced.pos_idx_mappings],
                                                     comp_mode=True)
    comp_advanced_dataloader = advanced_model.DataLoader(comp_advanced, shuffle=False)
    advanced_model_weights_path = 'adv_model_comp_weights.pt'
    trained_advanced_model = advanced_model.GoldMartDependencyParser(dataset=train_advanced, word_hidden_dim=125,
                                                                     MLP_inner_dim=100,
                                                                     BiLSTM_layers=3,
                                                                     dropout_layers=0.0,
                                                                     char_emb_dim=80,
                                                                     char_hidden_dim=50).cuda()
    trained_advanced_model.load_state_dict(torch.load(advanced_model_weights_path))
    advanced_model.tag_file_save_output(model=trained_advanced_model,
                                        dataloader=comp_advanced_dataloader,
                                        original_unlabeled_file='comp.unlabeled',
                                        result_path='comp_m2_308044296.labeled')


if __name__ == "__main__":
    main()
