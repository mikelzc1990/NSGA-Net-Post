# -*- coding: utf-8 -*-
import os
import pandas as pd
from individual import Network
from scipy.ndimage import rotate
from scipy.misc import imread, imsave, imshow


def scan_design_folder(folder_pth):
    if not os.path.exists(folder_pth):
        return 0
    else:
        subfolders = [f.name for f in os.scandir(folder_pth) if f.is_dir()]
        max_id = 0
        for folder in subfolders:
            if 'Design' in folder:
                tmp = int(folder[folder.find('_') + 1:])
                if tmp > -1 and tmp > max_id:
                    max_id = tmp
        return max_id


def validate_design_completeness(design_pth):
    # check existence of all the relevant files
    if not os.path.exists(design_pth):
        return False
    else:
        if not os.path.exists(os.path.join(design_pth, 'args.txt')):
            return False
        if not os.path.exists(os.path.join(design_pth, 'PerformanceLogger.txt')):
            return False
        if not os.path.exists(os.path.join(design_pth, 'TrainLogger.txt')):
            return False
        if not os.path.exists(os.path.join(design_pth, 'TestLogger.txt')):
            return False
    return True


def get_design_genome_from_args(design_pth):
    with open(os.path.join(design_pth, 'args.txt'), 'r') as f:
        content = f.readlines()
    for line in content:
        if 'genome' in line:
            genome_str = line[int(line.find('=')+2):]
        if 'n_phases' in line:
            n_phases_str = line[int(line.find('=')+2):]
        if 'n_nodes' in line:
            n_nodes_str = line[int(line.find('=')+2):]
    param = {
        'genome': eval(genome_str),
        'n_phases': eval(n_phases_str),
        'n_nodes': eval(n_nodes_str),
    }
    return param


def get_design_performance(design_pth):
    df = pd.read_csv(os.path.join(design_pth, 'PerformanceLogger.txt'),
                     sep='\t', lineterminator='\n')
    return list(df['Accuracy'])[0], list(df['Params'])[0], list(df['FLOPs'])[0], list(df['Robustness'])[0]


def main(results_root):
    # create the folder for saving processed data
    save_pth = os.path.join(experiment_pth, 'visualization')
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    # loop through designs to collect statistics
    summary_csv_content = ['Design ID,Genome,Active Nodes,Accuracy,Params,FLOPs,Robustness']
    n_designs = scan_design_folder(results_root)
    for d in range(1, n_designs+1):
        design_pth = os.path.join(results_root, 'Design_{}'.format(d))
        if validate_design_completeness(design_pth):
            design_args = get_design_genome_from_args(design_pth)
            network = Network(nid=d,
                              genome=design_args['genome'],
                              n_phases=design_args['n_phases'],
                              n_nodes=design_args['n_nodes'])
            # render network
            network.render_networks(genome=network.genome,
                                    save_pth=os.path.join(save_pth,
                                                          'Design_{}'.format(d)))
            img = imread(os.path.join(save_pth,'Design_{}.png'.format(d)))
            os.remove(os.path.join(save_pth, 'Design_{}'.format(d)))
            # rotate the image by 90 degree for visualization
            rotate_img = rotate(img, 90)
            imsave(os.path.join(save_pth, 'Design_{}.png'.format(d)), rotate_img)
            # extract performance
            acc, params, flops, robustness = get_design_performance(design_pth)
            line = '{},{},{},{},{},{},{}'.format(d, '|'.join(map(str, network.key)),
                                                 sum(network.active_nodes), acc, params, flops, robustness)
            summary_csv_content.append(line)

    # save the results to a csv file
    with open(os.path.join(save_pth, 'summary.csv'), 'w') as file:
        for line in summary_csv_content:
            file.write(line)
            file.write('\n')
    return


if __name__ == "__main__":
    experiment_pth = '/Users/zhichao.lu/Documents/GitHub/results/Analysis_MNIST_Run_1'
    main(results_root=experiment_pth)