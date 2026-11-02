"""Export measured category MAEs from the completed relative-energy benchmark."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    data = json.loads((ROOT/'summary.json').read_text())
    categories = ['reaction', 'barrier', 'noncovalent']
    titles = ['Reaction / isomerization energies', 'Barrier heights', 'Noncovalent binding energies']
    bases = ['minao', '3-21g', '6-31g(d,p)', 'def2-svp', 'cc-pvdz', 'NNAO-200-initial', 'NNAO-200-best']
    labels = ['MINAO', '3-21G', '6-31G(d,p)', 'def2-SVP', 'cc-pVDZ', 'NNAO initial', f'NNAO epoch {data["checkpoint"]["checkpoint_epoch"]}']
    colors = ['#a8afb6']*5+['#c7862c', '#2670a0']
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, reference in enumerate(['hf_ccpvtz', 'gmtkn55']):
        for col, category in enumerate(categories):
            ax = axes[row, col]
            records = {r['basis']: r for r in data['metrics'] if r['category'] == category}
            count = records[bases[0]]['common_complete_count']
            assert count > 0, 'Cannot plot absent category evidence.'
            values = [records[b][f'mae_vs_{reference}_kcal_mol'] for b in bases]
            ax.barh(range(len(bases)), values, color=colors, height=.68)
            ax.set_yticks(range(len(bases)), labels); ax.invert_yaxis()
            ax.set_xlim(0, max(values)*1.25)
            ax.set_title(titles[col]+f' ({count}/6 complete)')
            ax.set_xlabel('MAE (kcal/mol)')
            ax.set_axisbelow(True); ax.grid(axis='x', color='#e7e7e7', lw=.7)
            for y, value in enumerate(values):
                ax.text(value+max(values)*.015, y, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle(f'NNAO relative-energy benchmark | fixed epoch {data["checkpoint"]["checkpoint_epoch"]}', fontsize=16, y=.995)
    fig.text(.5, .955, 'Top: deviation from finite-basis HF/cc-pVTZ   |   Bottom: total error against GMTKN55 v2 references', ha='center', fontsize=11)
    fig.text(.5, .012, 'Selected 18-reaction panel; all methods DF-RHF. NCI energies have no counterpoise correction.\nGMTKN55 error includes HF method error; this is not a full GMTKN55 or CBS assessment.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .93), h_pad=3, w_pad=2)
    fig.savefig(ROOT/'relative-energy-mae.png', dpi=190)
    fig.savefig(ROOT/'relative-energy-mae.pdf')
    plt.close(fig)

if __name__ == '__main__':
    main()
