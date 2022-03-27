"""Copyright China University of Petroleum East China, Yimin Dou, Kewen Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalization_tensor(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


def z_score(data):
    return (data - np.mean(data)) / np.std(data)


def cubing_prediction(model, data, device, infer_size):
    with torch.no_grad():
        ol = 1
        model.eval()
        n1, n2, n3 = infer_size
        input_tensor = torch.from_numpy(data)
        m1, m2, m3 = data.shape
        c1 = np.ceil((m1 + ol) / (n1 - ol)).astype(np.int)
        c2 = np.ceil((m2 + ol) / (n2 - ol)).astype(np.int)
        c3 = np.ceil((m3 + ol) / (n3 - ol)).astype(np.int)
        p1 = (n1 - ol) * c1 + ol
        p2 = (n2 - ol) * c2 + ol
        p3 = (n3 - ol) * c3 + ol
        gp = torch.zeros((p1, p2, p3)).float() + 0.5
        gy = np.zeros((p1, p2, p3), dtype=np.single)
        gp[:m1, :m2, :m3] = input_tensor
        if device != 'cpu': gp = gp.half()
        for k1 in range(c1):
            for k2 in range(c2):
                for k3 in range(c3):
                    b1 = k1 * n1 - k1 * ol
                    e1 = b1 + n1
                    b2 = k2 * n2 - k2 * ol
                    e2 = b2 + n2
                    b3 = k3 * n3 - k3 * ol
                    e3 = b3 + n3
                    gs = gp[b1:e1, b2:e2, b3:e3]
                    gs = normalization_tensor(gs[None, None, :, :, :]).to(device)
                    Y = model(gs).cpu().numpy()
                    gy[b1:e1, b2:e2, b3:e3] = gy[b1:e1, b2:e2, b3:e3] + Y[0, 0, :, :, :]
    return gy[:m1, :m2, :m3]


def prediction(model, data, device):
    model.eval()
    data = normalization(data)
    m1, m2, m3 = data.shape
    c1 = (np.ceil(m1 / 16) * 16).astype(np.int)
    c2 = (np.ceil(m2 / 16) * 16).astype(np.int)
    c3 = (np.ceil(m3 / 16) * 16).astype(np.int)
    input_tensor = np.zeros((c1, c2, c3), dtype=np.float32) + 0.5
    input_tensor[:m1, :m2, :m3] = data
    input_tensor = torch.from_numpy(input_tensor)[None, None, :, :, :].to(device)
    if device != 'cpu': input_tensor = input_tensor.half()
    with torch.no_grad():
        result = model(input_tensor).cpu().numpy()[0, 0, :m1, :m2, :m3]
    return result


def write_data(results, geo_cube, out_path, input_file, axis=0):
    file_name = os.path.split(input_file)[-1]
    geo_cube = normalization(geo_cube)
    assert axis == 0 or axis == 1 or axis == 2
    for i in range(geo_cube.shape[axis]):
        if axis == 0:
            result = results[i, :, :]
            geo = geo_cube[i, :, :]
        elif axis == 1:
            result = results[:, i, :]
            geo = geo_cube[:, i, :]
        else:
            result = results[:, :, i]
            geo = geo_cube[:, :, i]
        hm = plt.get_cmap('bone')(geo)[:, :, :-1]
        geo = plt.get_cmap('seismic')(geo)[:, :, :-1]
        logits = np.clip((result[:, :, None]), a_min=0, a_max=1)
        colormap = plt.get_cmap('jet')(logits[:, :, 0])[:, :, :-1]
        hm = np.where(logits > 0.5, colormap, hm)
        line = np.ones((geo.shape[0], 50, 3))
        result = np.concatenate((geo, line, hm), axis=1)
        result = (result * 255).astype(np.uint8)
        if axis == 0:
            cv2.imwrite(os.path.join(out_path, file_name, 'tline', f'{axis}_%05d_.png' % i),
                        cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if axis == 1:
            cv2.imwrite(os.path.join(out_path, file_name, 'xline', f'{axis}_%05d_.png' % i),
                        cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if axis == 2:
            cv2.imwrite(os.path.join(out_path, file_name, 'iline', f'{axis}_%05d_.png' % i),
                        cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def create_out_dir(output_dir, input_file):
    file_name = os.path.split(input_file)[-1]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, file_name))
        os.mkdir(os.path.join(output_dir, file_name, 'iline'))
        os.mkdir(os.path.join(output_dir, file_name, 'xline'))
        os.mkdir(os.path.join(output_dir, file_name, 'tline'))

    if not os.path.exists(os.path.join(output_dir, file_name)):
        os.mkdir(os.path.join(output_dir, file_name))

    if not os.path.exists(os.path.join(output_dir, file_name, 'iline')):
        os.mkdir(os.path.join(output_dir, file_name, 'iline'))

    if not os.path.exists(os.path.join(output_dir, file_name, 'xline')):
        os.mkdir(os.path.join(output_dir, file_name, 'xline'))

    if not os.path.exists(os.path.join(output_dir, file_name, 'tline')):
        os.mkdir(os.path.join(output_dir, file_name, 'tline'))
