import json


def get_settings(filename, full_or_sub):
    with open(filename) as json_data_file:
        data = json.load(json_data_file)
    settings = data['settings']
    paths = data['paths']
    if full_or_sub == "full":
        working_folder = paths['data_identifier'] + '_full' + '_mb' + str(settings['minibatch_size'])
        if settings['ker_bracket']:
            working_folder = working_folder + '_bracket'
        if paths['init_hyper_file']:
            working_folder = working_folder + '_inithyper'
        working_folder = working_folder + paths['working_folder_suffix']
    else:
        working_folder = paths['data_identifier'] + '_subsets' + '_mb' + str(
            settings['minibatch_size']) + '_iternum' + str(settings['iter_num_subset'])
        if settings['ker_bracket']:
            working_folder = working_folder + '_bracket'
        working_folder = working_folder + paths['working_folder_suffix']
    # print('Results saved in:')
    # print(working_folder)
    if paths['init_hyper_file'] == '':
        paths['init_hyper_file'] = None
    return settings, paths, working_folder


# get_settings('../../config/air_cfg_test.json', 'sub')
