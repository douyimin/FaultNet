"""Copyright China University of Petroleum (East China), Yimin Dou, Kewen Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

# -*- coding:utf-8 -*-

import torch
import numpy as np
import argparse
import segyio
from utils import prediction, cubing_prediction, write_data, create_out_dir
from shutil import copyfile
import os

parser = argparse.ArgumentParser(description='Prediction management')
parser.add_argument('--input', type=str, default=r'data/F3.npy',
                    help='Input cuboid path (.npy or .segy or .sgy)')
parser.add_argument('--iline', type=int, default=189,
                    help='inline')  # 189 or 77, If none of them work, please fill the trace export in the commercial software and then read it.
parser.add_argument('--xline', type=int, default=193,
                    help='crossline')  # 193 or 73,If none of them work, please fill the trace export in the commercial software and then read it.
parser.add_argument('--gamma', type=float, default=0.7, help='Must 0.5,0.6,0.7')
parser.add_argument('--infer_size', type=int, nargs='+', default=None,
                    help='If None, the whole seismic volume is input.'
                         'If not None, the volume will be cut in blocks according to infer_size then input.'
                         'Shape = (tline,xline,iline) or (tline,iline,xline), must be divisible by 16')
parser.add_argument('--output_dir', type=str, default='output', help='Output dir')
parser.add_argument('--save_fault_cuboid', type=bool, default=False, help='')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    assert args.gamma == 0.5 or args.gamma == 0.6 or args.gamma == 0.7
    assert args.input[-3:] == 'npy' or args.input[-4:] == 'segy' or args.input[-3:] == 'sgy'
    if args.infer_size != None:
        assert args.infer_size[0] % 16 == 0 and args.infer_size[1] % 16 == 0 and args.infer_size[2] % 16 == 0
    if args.gamma == 0.5:
        model = torch.jit.load('network/FaultNet_Gamma0.5.pt').to(device)
    elif args.gamma == 0.6:
        model = torch.jit.load('network/FaultNet_Gamma0.6.pt').to(device)
    else:
        model = torch.jit.load('network/FaultNet_Gamma0.7.pt').to(device)

    if device.type != 'cpu': model = model.half()

    if args.input[-3:] == 'npy':
        data = np.load(args.input).transpose((2, 0, 1))  # Shape = (tline,xline,iline) or (tline,iline,xline),
    else:
        data = segyio.open(args.input, iline=args.iline, xline=args.xline)
        data = segyio.cube(data).transpose((2, 0, 1))  # Shape = (tline,xline,iline) or (tline,iline,xline),
    if args.infer_size == None:
        infer_size = data.shape
    else:
        infer_size = args.infer_size
    print('Load data successful.')
    print('Infer on', device)
    print(f'Data size is {tuple(data.shape)}, infer size is {infer_size}.')

    create_out_dir(args.output_dir, args.input)
    if args.infer_size == None:
        output = prediction(model, data, device)
    else:
        output = cubing_prediction(model, data, device, args.infer_size)
    print('Inference complete. Save results...')
    if args.save_fault_cuboid:
        if args.input[-3:] == 'npy':
            np.save(
                os.path.join(args.output_dir, os.path.split(args.input)[-1],
                             os.path.split(args.input)[-1] + '_Fault.npy'),
                output.transpose((1, 2, 0)))
            print(f'.npy fault cuboid file save in {args.output_dir}. Next save iline slices')
        else:
            output_file = os.path.join(args.output_dir, os.path.split(args.input)[-1],
                                       os.path.split(args.input)[-1] + '_Fault.segy')
            output = output.transpose((1, 2, 0))
            copyfile(args.input, output_file)
            with segyio.open(output_file, mode="r+", iline=args.iline, xline=args.xline) as src:
                for i, iline in enumerate(src.ilines):
                    src.iline[iline] = output[:, i, :]
            print(f'.segy fault cuboid file save in {args.output_dir}. Next save iline slices')
            output = output.transpose((2, 0, 1))
    write_data(output, data, args.output_dir, args.input, axis=2)
    print(f'iline slices save in {args.output_dir}. Next save tline slices')
    write_data(output, data, args.output_dir, args.input, axis=0)
    print(f'tline slices save in {args.output_dir}. Next save xline slices')
    write_data(output, data, args.output_dir, args.input, axis=1)
    print(f'xline slices save in {args.output_dir}. Slices save complete')
