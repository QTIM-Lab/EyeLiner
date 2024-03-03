import os
from shutil import copy

root = 'trained_models'
models = [
    'coris',
    'coris_tps1000',
    'coris_tps100',
    'coris_tps10',
    'coris_tps',
    'coris_tps0.1',
    'coris_tps0.01',
    'coris_tps0',
    'sigf',
    'sigf_tps1000',
    'sigf_tps100',
    'sigf_tps10',
    'sigf_tps',
    'sigf_tps0.1',
    'sigf_tps0.01',
    'sigf_tps0',
    'fire',
    'fire_tps1000',
    'fire_tps100',
    'fire_tps10',
    'fire_tps',
    'fire_tps0.1',
    'fire_tps0.01',
    'fire_tps0',
]

# make results files
os.makedirs('results_files', exist_ok=True)

for m in models:

    # copy loftr_g
    copy(os.path.join(root, m, 'loftr_g/results.csv'), os.path.join('results_files', f'{m}_loftr_g_results.csv'))

    # copy loftr_v
    copy(os.path.join(root, m, 'loftr_v/results.csv'), os.path.join('results_files', f'{m}_loftr_v_results.csv'))

    # copy loftr_vm
    copy(os.path.join(root, m, 'loftr_vm/results.csv'), os.path.join('results_files', f'{m}_loftr_vm_results.csv'))

    # copy splg_g
    copy(os.path.join(root, m, 'splg_g/results.csv'), os.path.join('results_files', f'{m}_splg_g_results.csv'))

    # copy splg_v
    copy(os.path.join(root, m, 'splg_v/results.csv'), os.path.join('results_files', f'{m}_splg_v_results.csv'))

    # copy splg_vm
    copy(os.path.join(root, m, 'splg_vm/results.csv'), os.path.join('results_files', f'{m}_splg_vm_results.csv'))