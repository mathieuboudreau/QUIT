import unittest
from nipype.interfaces.base import CommandLine
from QUIT.core import NewImage, Diff
from QUIT.relaxometry import Multiecho, MultiechoSim, VFAPrep, VFAPrepSim

vb = True
CommandLine.terminal_output = 'allatonce'


class Relax(unittest.TestCase):
    def test_multiecho(self):
        me = {'MultiEcho': {'TR': 10, 'TE1': 0.01, 'ESP': 0.01,
                            'ETL': 5}}
        me_file = 'sim_me.nii.gz'
        img_sz = [32, 32, 32]
        noise = 0.001

        NewImage(img_size=img_sz, grad_dim=0, grad_vals=(0.8, 1.0),
                 out_file='PD.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=2, grad_vals=(0.04, 0.1),
                 out_file='T2.nii.gz', verbose=vb).run()

        MultiechoSim(sequence=me, in_file=me_file,
                     PD='PD.nii.gz', T2='T2.nii.gz',
                     noise=noise, verbose=vb).run()
        Multiecho(sequence=me, in_file=me_file, verbose=vb).run()

        diff_T2 = Diff(in_file='ME_T2.nii.gz', baseline='T2.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_PD = Diff(in_file='ME_PD.nii.gz', baseline='PD.nii.gz',
                       noise=noise, verbose=vb).run()
        self.assertLessEqual(diff_T2.outputs.out_diff, 3)
        self.assertLessEqual(diff_PD.outputs.out_diff, 2)

    def test_vfaprep_b1(self):
        seq = {'VFAPrep': {'TR': 0.002786, 'SPS': 128, 'adiabatic': False, 'ramp': 0.01,
                           'FA':     [1,    6,     1,     6,     1,     6],
                           'PrepFA': [0,    0,    30,    30,    60,    60],
                           'TE':     [0.0, 0.0, 0.015, 0.015, 0.015, 0.015],
                           'n180':   [0,    0,     2,     2,     2,     2]}}
        sim_file = 'sim_vfa_b1.nii.gz'
        img_sz = [32, 32, 16]
        noise = 0.1

        NewImage(img_size=img_sz, fill=100.0,
                 out_file='PD.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=0, grad_vals=(0.8, 1.5),
                 out_file='T1.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=1, grad_vals=(0.04, 0.1),
                 out_file='T2.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=2, grad_vals=(0.9, 1.5),
                 out_file='B1.nii.gz', verbose=vb).run()

        VFAPrepSim(sequence=seq, in_file=sim_file,
                   PD='PD.nii.gz', T1='T1.nii.gz', T2='T2.nii.gz', B1='B1.nii.gz',
                   noise=noise, verbose=vb).run()
        VFAPrep(sequence=seq, in_file=sim_file,
                prefix='b1', verbose=vb, threads=-1).run()

        diff_PD = Diff(in_file='b1VFAPrep_PD.nii.gz', baseline='PD.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_T1 = Diff(in_file='b1VFAPrep_T1.nii.gz', baseline='T1.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_T2 = Diff(in_file='b1VFAPrep_T2.nii.gz', baseline='T2.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_B1 = Diff(in_file='b1VFAPrep_B1.nii.gz', baseline='B1.nii.gz',
                       noise=noise, verbose=vb).run()

        self.assertLessEqual(diff_PD.outputs.out_diff, 1)
        self.assertLessEqual(diff_T1.outputs.out_diff, 1)
        self.assertLessEqual(diff_T2.outputs.out_diff, 1)
        self.assertLessEqual(diff_B1.outputs.out_diff, 1)

    def test_vfaprep_adiabatic(self):
        seq = {'VFAPrep': {'TR': 0.002786, 'SPS': 128, 'adiabatic': True, 'ramp': 0.01,
                           'FA':     [2,     2,     2,     2],
                           'PrepFA': [0,   180,    90,    90],
                           'TE':     [0, 100.0, 0.025, 0.075],
                           'n180':   [0,     0,     1,     1]}}
        sim_file = 'sim_vfa_adiabatic.nii.gz'
        img_sz = [32, 32, 16]
        noise = 0.01

        NewImage(img_size=img_sz, fill=100.0,
                 out_file='PD.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=0, grad_vals=(0.8, 1.5),
                 out_file='T1.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=1, grad_vals=(0.04, 0.1),
                 out_file='T2.nii.gz', verbose=vb).run()
        NewImage(img_size=img_sz, grad_dim=2, grad_vals=(0.9, 1.5),
                 out_file='B1.nii.gz', verbose=vb).run()

        VFAPrepSim(sequence=seq, in_file=sim_file,
                   PD='PD.nii.gz', T1='T1.nii.gz', T2='T2.nii.gz', B1='B1.nii.gz',
                   noise=noise, verbose=vb).run()
        VFAPrep(sequence=seq, in_file=sim_file, verbose=vb, threads=-1).run()

        diff_PD = Diff(in_file='VFAPrep_PD.nii.gz', baseline='PD.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_T1 = Diff(in_file='VFAPrep_T1.nii.gz', baseline='T1.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_T2 = Diff(in_file='VFAPrep_T2.nii.gz', baseline='T2.nii.gz',
                       noise=noise, verbose=vb).run()
        diff_B1 = Diff(in_file='VFAPrep_B1.nii.gz', baseline='B1.nii.gz',
                       noise=noise, verbose=vb).run()

        self.assertLessEqual(diff_PD.outputs.out_diff, 1)
        self.assertLessEqual(diff_T1.outputs.out_diff, 1)
        self.assertLessEqual(diff_T2.outputs.out_diff, 1)
        self.assertLessEqual(diff_B1.outputs.out_diff, 1)


if __name__ == '__main__':
    unittest.main()
