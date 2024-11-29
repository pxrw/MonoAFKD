from lib.models.MonoLSS import MonoLSS
from lib.models.distill_model import MonoAFKD

def build_model(cfg, mean_size, flag):
    if cfg['type'] != 'distill':
        return MonoLSS(backbone=cfg['backbone'],
                       neck=cfg['neck'],
                       mean_size=mean_size,
                       model_type=cfg['type'])
    elif cfg['type'] == 'distill':
        return MonoAFKD(backbone=cfg['backbone'],
                            neck=cfg['neck'],
                            flag=flag,
                            mean_size=mean_size,
                            model_type='distill',
                            cfg=cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
