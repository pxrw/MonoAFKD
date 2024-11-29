import torch, os, yaml, shutil
from tqdm import tqdm

from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader

def create_test_dataloader(cfg):
    cfg = cfg['dataset']
    _, _, test_dataloader = build_dataloader(cfg)

    return test_dataloader


def inference(cfg):
    test_dataloader = create_test_dataloader(cfg)
    mean_size = test_dataloader.dataset.cls_mean_size
    class_name = test_dataloader.dataset.class_name
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def create_model():
        weight = cfg['tester']['resume_model']
        model = build_model(cfg['model'], mean_size, flag='testing')
        model_state_dict = model.state_dict()
        ckpt = torch.load(weight)
        pretrained_state_dict = ckpt['model_state']
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        print(f'load pretrained weight from {weight}...')
        model.to(device)
        return model.eval()

    def save_results(results, output_dir):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                cls_name = class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(cls_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    torch.set_grad_enabled(False)
    model = create_model()

    results = {}
    progress_bar = tqdm(total=len(test_dataloader), leave=True, desc='Testing progress')
    for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(test_dataloader):
        if type(inputs) != dict:
            inputs = inputs.to(device)
        else:
            for key in inputs.keys(): inputs[key] = inputs[key].to(device)

        calibs = calibs.to(device)
        coord_ranges = coord_ranges.to(device)
        _, outputs, _ = model(inputs, coord_ranges, calibs, K=50, mode='test')
        dets = extract_dets_from_outputs(outputs=outputs, K=50)
        dets = dets.detach().cpu().numpy()

        calibs = [test_dataloader.dataset.get_calib(index) for index in info['img_id']]
        info = {key: val.detach().cpu().numpy() for key, val in info.items()}

        dets = decode_detections(dets=dets,
                                 info=info,
                                 calibs=calibs,
                                 cls_mean_size=mean_size,
                                 threshold=cfg['tester']['threshold'])
        results.update(dets)
        progress_bar.update()

    output_dir = r'/home/pxr/pxrProject/3Ddetection/MonoLSS/test_result'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    save_results(results, output_dir)
    progress_bar.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cfg_path = r'/home/pxr/pxrProject/3Ddetection/MonoLSS/configs/distill.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.Loader)
    inference(cfg)







