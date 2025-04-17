"""Title:Plot RDM for Multiple Arrangements tasks modified v1"""
"""ingredients: model, stims, neuro"""

from __future__ import annotations
from typing import TYPE_CHECKING
import rsatoolbox, numpy
from matplotlib import pyplot
if TYPE_CHECKING:
    from jobcontext import JobContext
import numpy as np
import nibabel as nib
from nilearn import plotting

def main(job: JobContext):
    print('you are here and this is done!')
    n = len(job.data)
    if n == 0:
        job.log('Zero participants')
        return
    for p, (pnick, pdata) in enumerate(job.data.items(), start=1):
        job.log(f'working on participant {p}/{n}', p/n)
        if 'tasks' not in pdata:
            job.log('missing tasks')
            continue
        for _, task in enumerate(pdata['tasks']):
            task_meta = task.get('task', {})
            if task_meta.get('task_type') == 'multiarrange':
                task_stimuli = [s['name'].split('.')[0] for s in task['stimuli']]
                rdms = rsatoolbox.rdm.rdms.RDMs(
                    numpy.atleast_2d(task['rdm']),
                    dissimilarity_measure='euclidean',
                    rdm_descriptors=dict(participation=[pnick]),
                    pattern_descriptors=dict(conds=task_stimuli),
                )
                pyplot.close('all')
                fig, _, _ = rsatoolbox.vis.rdm_plot.show_rdm(rdms)
                fpath = job.outputPath.joinpath(f'{pnick}.png')
                pyplot.savefig(fpath)
                job.addFile(fpath)
                break



def searchlight():

    # Loading brain mask
    brain_mask_nii = nib.load('./data/brainmask.nii.gz')
    brain_mask_data = brain_mask_nii.get_fdata()

    # Loading searchlight centers
    centers_linear = np.load('./data/subj04_searchlight_func1pt8mm_centers.npy')
    centers = np.array(np.unravel_index(centers_linear, brain_mask_data.shape)).T
    rdms = np.load('./data/subj04_searchlight_func1pt8mm_rdms.npy')
    volume = np.zeros(brain_mask_data.shape)

    for center, rdm in zip(centers, rdms):
        measure = np.mean(rdm)
        volume[tuple(center)] = measure

    # Create a new NIfTI image from the volume
    volume_nii = nib.Nifti1Image(volume, brain_mask_nii.affine, brain_mask_nii.header)
    nib.save(volume_nii, 'rdm_measure.nii.gz')


def visualize(path='rdm_measure.nii.gz'):

    nii = nib.load(path)
    plotting.plot_stat_map(nii, colorbar=True, cmap='jet')
    plotting.show()


if __name__ == '__main__':
    searchlight()
    visualize('rdm_measure.nii.gz')